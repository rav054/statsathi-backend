"""Formulas for calculated columns, evaluated safely over a table.

A researcher types something like

    [Plot yield (kg)] / [Plot area (m2)] * 10000
    if_else(Treatment == "Control", 0, log10(Dose))
    row_mean(Rep1, Rep2, Rep3)

and gets a new column. The formula is parsed with Python's ``ast`` module and only a small, explicit set of nodes is
accepted - numbers, text, column names, arithmetic, comparisons, and/or/not, and the functions listed in FUNCTIONS.
Nothing is ever passed to ``eval``, so a formula cannot reach files, the network or the interpreter.

Column names that are not plain identifiers (spaces, units, brackets) are written inside square brackets.
"""

import ast
import re
import warnings

import numpy as np
import pandas as pd

MAX_LENGTH = 500


class FormulaError(ValueError):
    """A formula that cannot be read or evaluated, with a message meant for the user."""


def _numeric(value, label="value"):
    if isinstance(value, pd.Series):
        if pd.api.types.is_bool_dtype(value):
            return value.astype(float)
        if not pd.api.types.is_numeric_dtype(value):
            converted = pd.to_numeric(value, errors="coerce")
            if converted.notna().sum() == 0 and value.notna().sum() > 0:
                raise FormulaError(f"\"{label}\" is not numeric.")
            return converted.astype(float)
        return value.astype(float)
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.number)):
        return float(value)
    raise FormulaError(f"Expected a number but got \"{value}\".")


def _row_stat(reducer):
    def compute(*values):
        if not values:
            raise FormulaError("Give at least one column.")
        frame = pd.concat([_numeric(v) if isinstance(v, pd.Series) else pd.Series(_numeric(v), index=_index(values))
                           for v in values], axis=1)
        return reducer(frame)
    return compute


def _index(values):
    for value in values:
        if isinstance(value, pd.Series):
            return value.index
    raise FormulaError("Row functions need at least one column.")


def _if_else(condition, when_true, when_false):
    if isinstance(condition, pd.Series):
        mask = condition.fillna(False).astype(bool)
        index = condition.index
        true_values = when_true if isinstance(when_true, pd.Series) else pd.Series(when_true, index=index)
        false_values = when_false if isinstance(when_false, pd.Series) else pd.Series(when_false, index=index)
        return true_values.where(mask, false_values)
    return when_true if condition else when_false


def _round(value, digits=0):
    return np.round(_numeric(value), int(digits))


FUNCTIONS = {
    # element-wise maths
    "log": lambda x: np.log(_numeric(x)),
    "ln": lambda x: np.log(_numeric(x)),
    "log10": lambda x: np.log10(_numeric(x)),
    "log2": lambda x: np.log2(_numeric(x)),
    "sqrt": lambda x: np.sqrt(_numeric(x)),
    "exp": lambda x: np.exp(_numeric(x)),
    "abs": lambda x: np.abs(_numeric(x)),
    "round": _round,
    "floor": lambda x: np.floor(_numeric(x)),
    "ceil": lambda x: np.ceil(_numeric(x)),
    "clip": lambda x, low, high: np.clip(_numeric(x), _numeric(low), _numeric(high)),
    # arcsine square-root, the classic transformation for proportions
    "arcsin_sqrt": lambda x: np.arcsin(np.sqrt(_numeric(x))),
    # choices and gaps
    "if_else": _if_else,
    "is_missing": lambda x: x.isna() if isinstance(x, pd.Series) else pd.isna(x),
    "fill": lambda x, value: x.fillna(value) if isinstance(x, pd.Series) else (value if pd.isna(x) else x),
    # across several columns in the same row
    "row_mean": _row_stat(lambda frame: frame.mean(axis=1)),
    "row_sum": _row_stat(lambda frame: frame.sum(axis=1, min_count=1)),
    "row_min": _row_stat(lambda frame: frame.min(axis=1)),
    "row_max": _row_stat(lambda frame: frame.max(axis=1)),
    # summaries of a whole column, usable inside a row formula (e.g. relative yield)
    "mean": lambda x: float(_numeric(x).mean()),
    "median": lambda x: float(_numeric(x).median()),
    "sd": lambda x: float(_numeric(x).std(ddof=1)),
    "min": lambda x: float(_numeric(x).min()),
    "max": lambda x: float(_numeric(x).max()),
    "sum": lambda x: float(_numeric(x).sum()),
    "zscore": lambda x: (_numeric(x) - _numeric(x).mean()) / _numeric(x).std(ddof=1),
    "rank": lambda x: _numeric(x).rank(method="average"),
}

FUNCTION_HELP = {
    "log / ln, log10, log2, sqrt, exp, abs": "element-wise maths",
    "round(x, digits), floor, ceil, clip(x, low, high)": "rounding and limits",
    "arcsin_sqrt(p)": "arcsine square-root of a proportion",
    "if_else(condition, a, b)": "a where the condition holds, otherwise b",
    "is_missing(x), fill(x, value)": "gaps",
    "row_mean, row_sum, row_min, row_max": "across columns in the same row",
    "mean, median, sd, min, max, sum": "one number summarising a whole column",
    "zscore(x), rank(x)": "standardise or rank a column",
}

BINARY = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
    ast.Mod: lambda a, b: a % b,
    ast.FloorDiv: lambda a, b: a // b,
}

COMPARE = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
}

BRACKETED = re.compile(r"\[([^\[\]]+)\]")


def _substitute_columns(formula, columns):
    """Replace [Column name] with safe placeholder identifiers, returning the new text and the lookup."""
    lookup = {}

    def replace(match):
        name = match.group(1).strip()
        if name not in columns:
            raise FormulaError(f"There is no column called \"{name}\".")
        key = f"__col{len(lookup)}__"
        lookup[key] = name
        return key

    return BRACKETED.sub(replace, formula), lookup


class _Evaluator:
    def __init__(self, df, lookup):
        self.df = df
        self.lookup = lookup

    def column(self, name):
        return self.df[name]

    def visit(self, node):
        method = getattr(self, f"visit_{type(node).__name__}", None)
        if method is None:
            raise FormulaError(f"\"{ast.unparse(node) if hasattr(ast, 'unparse') else type(node).__name__}\" is not allowed in a formula.")
        return method(node)

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float, str, bool)):
            return node.value
        raise FormulaError("Only numbers and text in quotes can be written in a formula.")

    def visit_Name(self, node):
        if node.id in self.lookup:
            return self.column(self.lookup[node.id])
        if node.id in self.df.columns:
            return self.column(node.id)
        if node.id in ("True", "False"):
            return node.id == "True"
        if node.id in FUNCTIONS:
            raise FormulaError(f"\"{node.id}\" is a function: write {node.id}(...).")
        raise FormulaError(f"There is no column called \"{node.id}\". Put names with spaces in [square brackets].")

    def visit_BinOp(self, node):
        operation = BINARY.get(type(node.op))
        if operation is None:
            raise FormulaError("Only + - * / ** % and // can be used for arithmetic.")
        left, right = self.visit(node.left), self.visit(node.right)
        if isinstance(left, str) or isinstance(right, str):
            raise FormulaError("Text can only be compared (==, !=), not used in arithmetic.")
        try:
            return operation(_numeric(left), _numeric(right))
        except (ZeroDivisionError, OverflowError):
            # Plain numbers (not columns) raise instead of giving inf; either way the cell is undefined
            return np.nan

    def visit_UnaryOp(self, node):
        value = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -_numeric(value)
        if isinstance(node.op, ast.UAdd):
            return _numeric(value)
        if isinstance(node.op, ast.Not):
            return ~value.fillna(False).astype(bool) if isinstance(value, pd.Series) else not value
        raise FormulaError("That operator is not allowed.")

    def visit_Compare(self, node):
        left = self.visit(node.left)
        result = None
        for op, comparator in zip(node.ops, node.comparators):
            operation = COMPARE.get(type(op))
            if operation is None:
                raise FormulaError("Comparisons use == != < <= > >=.")
            right = self.visit(comparator)
            if isinstance(left, str) or isinstance(right, str):
                if type(op) not in (ast.Eq, ast.NotEq):
                    raise FormulaError("Text can only be compared with == or !=.")
                a = left.astype(str).where(left.notna()) if isinstance(left, pd.Series) else left
                b = right.astype(str).where(right.notna()) if isinstance(right, pd.Series) else right
                step = operation(a, b)
            else:
                step = operation(_numeric(left), _numeric(right))
            result = step if result is None else (result & step)
            left = right
        return result

    def visit_BoolOp(self, node):
        values = [self.visit(v) for v in node.values]
        as_mask = [v.fillna(False).astype(bool) if isinstance(v, pd.Series) else bool(v) for v in values]
        combined = as_mask[0]
        for value in as_mask[1:]:
            combined = (combined & value) if isinstance(node.op, ast.And) else (combined | value)
        return combined

    def visit_IfExp(self, node):
        return _if_else(self.visit(node.test), self.visit(node.body), self.visit(node.orelse))

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            name = node.func.id if isinstance(node.func, ast.Name) else "that"
            raise FormulaError(f"\"{name}\" is not a function StatSathi knows. Available: {', '.join(sorted(FUNCTIONS))}.")
        if node.keywords:
            raise FormulaError("Write function arguments in order, without names.")
        arguments = [self.visit(arg) for arg in node.args]
        try:
            return FUNCTIONS[node.func.id](*arguments)
        except TypeError:
            raise FormulaError(f"{node.func.id}() was given the wrong number of values.")


def evaluate(df, formula):
    """The formula's value for every row, as a Series (a single number is repeated down the column)."""
    text = (formula or "").strip()
    if not text:
        raise FormulaError("Write a formula.")
    if len(text) > MAX_LENGTH:
        raise FormulaError(f"Formulas can be up to {MAX_LENGTH} characters.")
    substituted, lookup = _substitute_columns(text, set(map(str, df.columns)))
    try:
        tree = ast.parse(substituted, mode="eval")
    except SyntaxError:
        raise FormulaError("The formula could not be read - check the brackets, quotes and operators.")

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        value = _Evaluator(df, lookup).visit(tree)

    if not isinstance(value, pd.Series):
        value = pd.Series(value, index=df.index)
    if pd.api.types.is_numeric_dtype(value) and not pd.api.types.is_bool_dtype(value):
        value = value.astype(float).replace([np.inf, -np.inf], np.nan)
    return value
