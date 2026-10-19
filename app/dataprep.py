"""Data preparation: what is wrong with a dataset, and the steps that fix it.

Two halves:

* ``profile`` looks at a table and reports what a researcher would otherwise find the hard way - missing values,
  duplicated rows, numbers stored as text ("1,200 kg"), category labels that differ only by spacing or case,
  constant columns and outliers.
* ``apply_steps`` runs an ordered recipe of cleaning steps over the table. Every step reports in words what it
  did ("Filled 7 missing values in Yield with the median (4.35)"), so the cleaning can be written up, checked by a
  supervisor, and run again on next season's data.

The steps are plain dictionaries, which is what the API receives and what a saved recipe stores.
"""

import re

import numpy as np
import pandas as pd

from . import expressions, imputation

MAX_PREVIEW_ROWS = 50
MAX_TOP_VALUES = 5
MAX_EXAMPLES = 3

# Values that mean "missing" in field and laboratory sheets but arrive as text
MISSING_TOKENS = ("", "na", "n/a", "n.a.", "nan", "null", "none", "nil", "-", "--", "?", ".", "missing", "not recorded")

# Text that commonly travels with a number: thousands separators, units, stray symbols
SEPARATORS = re.compile(r"[,\s]")
UNITS = re.compile(r"(?<=\d)(kg|g|ha|cm|mm|m|%|°c|ppm|t)$", re.IGNORECASE)


class StepError(ValueError):
    """A recipe step that cannot be carried out (the API turns this into a 400)."""


# ----------------------------------------------------------------------------------------------- describing a table

def column_kind(series):
    """How a column behaves: 'numeric', 'datetime', 'boolean' or 'categorical'."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "categorical"


def as_number(series):
    """The column read as numbers, with thousands separators and units stripped first."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    text = series.astype(str).str.strip()
    # Separators go first, so "12 kg" becomes "12kg" and the unit can then be recognised after the digits
    text = text.str.replace(SEPARATORS, "", regex=True).str.replace(UNITS, "", regex=True)
    return pd.to_numeric(text, errors="coerce")


def outlier_mask(values, method="iqr", threshold=None):
    """Which values a rule considers extreme.

    iqr      - outside Q1 - k·IQR … Q3 + k·IQR (Tukey, k = 1.5 by default)
    zscore   - |x - mean| / sd above the threshold (3 by default)
    mad      - |0.6745·(x - median)| / MAD above the threshold (3.5 by default, Iglewicz & Hoaglin)
    """
    numbers = pd.to_numeric(values, errors="coerce")
    present = numbers.dropna()
    mask = pd.Series(False, index=numbers.index)
    if len(present) < 3:
        return mask
    if method == "iqr":
        k = 1.5 if threshold is None else float(threshold)
        q1, q3 = present.quantile(0.25), present.quantile(0.75)
        spread = q3 - q1
        if spread == 0:
            return mask
        return (numbers < q1 - k * spread) | (numbers > q3 + k * spread)
    if method == "zscore":
        limit = 3.0 if threshold is None else float(threshold)
        sd = present.std(ddof=1)
        if not sd or np.isnan(sd):
            return mask
        return (numbers - present.mean()).abs() / sd > limit
    if method == "mad":
        limit = 3.5 if threshold is None else float(threshold)
        median = present.median()
        mad = (present - median).abs().median()
        if mad == 0:
            return mask
        return (0.6745 * (numbers - median)).abs() / mad > limit
    raise StepError(f"Unknown outlier rule: {method}. Use iqr, zscore or mad.")


def _text_variants(series):
    """Labels that are the same word written differently ('RBD ', 'rbd', 'Rbd')."""
    text = series.dropna().astype(str)
    if text.empty:
        return []
    normalized = text.str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    groups = {}
    for original, key in zip(text, normalized):
        groups.setdefault(key, set()).add(original)
    return [sorted(spellings) for spellings in groups.values() if len(spellings) > 1]


def _blank_like(series):
    """Cells holding text that means "missing" but is not recognised as such."""
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
        return 0
    text = series.dropna().astype(str).str.strip().str.lower()
    return int(text.isin(MISSING_TOKENS).sum())


def describe_column(series):
    """One column's shape and the problems worth flagging."""
    n = len(series)
    missing = int(series.isna().sum())
    kind = column_kind(series)
    info = {
        "name": str(series.name),
        "kind": kind,
        "dtype": str(series.dtype),
        "missing": missing,
        "missing_pct": round(100 * missing / n, 2) if n else 0.0,
        "unique": int(series.nunique(dropna=True)),
        "examples": [str(v) for v in series.dropna().unique()[:MAX_EXAMPLES]],
        "issues": [],
    }

    present = n - missing
    if present == 0:
        info["issues"].append({"code": "all_missing", "message": "Every value is missing."})
    elif info["unique"] == 1:
        info["issues"].append({"code": "constant", "message": "Every row has the same value."})

    blanks = _blank_like(series)
    if blanks:
        info["blank_like"] = blanks
        info["issues"].append({
            "code": "missing_as_text",
            "message": f"{blanks} cell{'s' if blanks != 1 else ''} say \"NA\", \"-\" or similar instead of being empty.",
        })

    if kind == "numeric":
        numbers = series.astype(float)
        present_values = numbers.dropna()
        if not present_values.empty:
            info.update({
                "min": float(present_values.min()),
                "max": float(present_values.max()),
                "mean": float(present_values.mean()),
                "median": float(present_values.median()),
                "sd": float(present_values.std(ddof=1)) if len(present_values) > 1 else 0.0,
                "zeros": int((present_values == 0).sum()),
                "negatives": int((present_values < 0).sum()),
            })
            info["outliers"] = {
                rule: int(outlier_mask(numbers, rule).sum()) for rule in ("iqr", "zscore", "mad")
            }
            if info["outliers"]["iqr"]:
                info["issues"].append({
                    "code": "outliers",
                    "message": f"{info['outliers']['iqr']} value(s) lie outside 1.5 × IQR of the quartiles.",
                })
    else:
        counts = series.dropna().astype(str).value_counts()
        info["top_values"] = [[str(value), int(count)] for value, count in counts.head(MAX_TOP_VALUES).items()]

        numeric_view = as_number(series)
        readable = int(numeric_view.notna().sum())
        if present and readable >= 0.8 * present and info["unique"] > 1:
            info["numeric_ready"] = readable
            info["issues"].append({
                "code": "numeric_as_text",
                "message": f"{readable} of {present} values are numbers stored as text - this column can be converted.",
            })

        variants = _text_variants(series)
        if variants:
            info["variants"] = variants[:MAX_TOP_VALUES]
            info["issues"].append({
                "code": "label_variants",
                "message": f"{len(variants)} label(s) are written in more than one way, for example {variants[0][:2]}.",
            })

    return info


def profile(df):
    """Everything the preparation screen shows about a dataset before anything is changed."""
    rows, columns = df.shape
    duplicates = int(df.duplicated().sum())
    empty_rows = int(df.isna().all(axis=1).sum())
    column_reports = [describe_column(df[name]) for name in df.columns]
    total_cells = rows * columns
    return {
        "n_rows": int(rows),
        "n_columns": int(columns),
        "duplicate_rows": duplicates,
        "empty_rows": empty_rows,
        "missing_cells": int(df.isna().sum().sum()),
        "missing_pct": round(100 * float(df.isna().sum().sum()) / total_cells, 2) if total_cells else 0.0,
        "complete_rows": int(df.notna().all(axis=1).sum()),
        "columns": column_reports,
        "issue_count": sum(len(c["issues"]) for c in column_reports) + (1 if duplicates else 0),
    }


def preview(df, rows=MAX_PREVIEW_ROWS):
    """The first rows as plain JSON, with NaN as null."""
    head = df.head(rows).replace({np.nan: None})
    return {
        "columns": [str(c) for c in df.columns],
        "rows": head.astype(object).where(head.notna(), None).to_dict(orient="records"),
        "showing": int(min(rows, len(df))),
        "total": int(len(df)),
    }


# ------------------------------------------------------------------------------------------------- recipe steps

def _columns_of(df, step, key="columns", default_all=False, kind=None):
    """The columns a step names, checked against the table."""
    named = step.get(key)
    if named in (None, [], ""):
        if not default_all:
            raise StepError(f"This step needs at least one column in \"{key}\".")
        named = [c for c in df.columns if kind is None or column_kind(df[c]) == kind]
    if isinstance(named, str):
        named = [named]
    missing = [c for c in named if c not in df.columns]
    if missing:
        raise StepError(f"Column(s) not in the dataset: {', '.join(map(str, missing))}.")
    return list(named)


def _plural(count, word, plural=None):
    return f"{count} {word if count == 1 else (plural or word + 's')}"


def step_drop_missing(df, step):
    """Remove rows (or columns) with missing values."""
    axis = step.get("axis", "rows")
    how = step.get("how", "any")
    if how not in ("any", "all"):
        raise StepError("\"how\" must be any or all.")
    if axis == "rows":
        subset = _columns_of(df, step, default_all=True) if step.get("columns") else None
        before = len(df)
        result = df.dropna(axis=0, how=how, subset=subset)
        removed = before - len(result)
        where = f" (looking at {', '.join(subset)})" if subset else ""
        return result, f"Removed {_plural(removed, 'row')} with {'any' if how == 'any' else 'only'} missing values{where}."
    if axis == "columns":
        limit = step.get("max_missing_fraction")
        if limit is None:
            result = df.dropna(axis=1, how=how)
        else:
            limit = float(limit)
            if not 0 <= limit <= 1:
                raise StepError("\"max_missing_fraction\" must be between 0 and 1.")
            keep = [c for c in df.columns if df[c].isna().mean() <= limit]
            result = df[keep]
        dropped = [str(c) for c in df.columns if c not in result.columns]
        detail = f" ({', '.join(dropped)})" if dropped else ""
        return result, f"Removed {_plural(len(dropped), 'column')} with too many missing values{detail}."
    raise StepError("\"axis\" must be rows or columns.")


def step_fill_missing(df, step):
    """Fill the gaps in the named columns."""
    columns = _columns_of(df, step)
    method = step.get("method", "mean")
    group_by = step.get("group_by") or None
    if group_by and group_by not in df.columns:
        raise StepError(f"Grouping column not in the dataset: {group_by}.")

    result = df.copy()
    filled_total = 0
    described = []
    for name in columns:
        column = result[name]
        if method in ("mean", "median", "interpolate") and not pd.api.types.is_numeric_dtype(column):
            raise StepError(f"\"{name}\" is not numeric, so it cannot be filled with the {method}.")
        gaps = int(column.isna().sum())
        if gaps == 0:
            continue

        if method in ("mean", "median", "mode") and group_by:
            def group_fill(values):
                if values.dropna().empty:
                    return values
                if method == "mean":
                    return values.fillna(values.mean())
                if method == "median":
                    return values.fillna(values.median())
                return values.fillna(values.mode().iloc[0])
            result[name] = result.groupby(group_by, dropna=False)[name].transform(group_fill)
            described.append(f"{name} ({method} of its {group_by} group)")
        elif method == "mean":
            value = column.mean()
            result[name] = column.fillna(value)
            described.append(f"{name} (mean {value:.4g})")
        elif method == "median":
            value = column.median()
            result[name] = column.fillna(value)
            described.append(f"{name} (median {value:.4g})")
        elif method == "mode":
            modes = column.mode()
            if modes.empty:
                continue
            result[name] = column.fillna(modes.iloc[0])
            described.append(f"{name} (most common value {modes.iloc[0]})")
        elif method == "constant":
            if "value" not in step:
                raise StepError("Filling with a constant needs a \"value\".")
            result[name] = column.fillna(step["value"])
            described.append(f"{name} ({step['value']})")
        elif method in ("ffill", "bfill"):
            result[name] = column.ffill() if method == "ffill" else column.bfill()
            described.append(f"{name} ({'previous' if method == 'ffill' else 'next'} value)")
        elif method == "interpolate":
            if not pd.api.types.is_numeric_dtype(column):
                raise StepError(f"\"{name}\" is not numeric, so it cannot be interpolated.")
            result[name] = column.interpolate(limit_direction="both")
            described.append(f"{name} (interpolated)")
        else:
            raise StepError(f"Unknown fill method: {method}.")
        filled_total += gaps - int(result[name].isna().sum())

    if not described:
        return result, "No missing values to fill in the chosen columns."
    return result, f"Filled {_plural(filled_total, 'missing value')} in {', '.join(described)}."


def step_drop_duplicates(df, step):
    """Remove repeated rows."""
    subset = _columns_of(df, step) if step.get("columns") else None
    keep = step.get("keep", "first")
    if keep not in ("first", "last", "none"):
        raise StepError("\"keep\" must be first, last or none.")
    before = len(df)
    result = df.drop_duplicates(subset=subset, keep=False if keep == "none" else keep)
    where = f" on {', '.join(subset)}" if subset else ""
    return result, f"Removed {_plural(before - len(result), 'duplicate row')}{where}."


COMPARISONS = {
    "eq": lambda s, v, _: s == v,
    "ne": lambda s, v, _: s != v,
    "lt": lambda s, v, _: pd.to_numeric(s, errors="coerce") < float(v),
    "le": lambda s, v, _: pd.to_numeric(s, errors="coerce") <= float(v),
    "gt": lambda s, v, _: pd.to_numeric(s, errors="coerce") > float(v),
    "ge": lambda s, v, _: pd.to_numeric(s, errors="coerce") >= float(v),
    "between": lambda s, v, v2: pd.to_numeric(s, errors="coerce").between(float(v), float(v2)),
    "in": lambda s, v, _: s.astype(str).isin([str(x) for x in v]),
    "not_in": lambda s, v, _: ~s.astype(str).isin([str(x) for x in v]),
    "contains": lambda s, v, _: s.astype(str).str.contains(str(v), case=False, na=False),
    "is_missing": lambda s, v, _: s.isna(),
    "not_missing": lambda s, v, _: s.notna(),
}

COMPARISON_WORDS = {
    "eq": "is", "ne": "is not", "lt": "is below", "le": "is at most", "gt": "is above", "ge": "is at least",
    "between": "is between", "in": "is one of", "not_in": "is not one of", "contains": "contains",
    "is_missing": "is missing", "not_missing": "is present",
}


def step_filter_rows(df, step):
    """Keep (or drop) the rows that match a condition."""
    column = _columns_of(df, step, key="column")[0]
    op = step.get("op", "eq")
    if op not in COMPARISONS:
        raise StepError(f"Unknown comparison: {op}.")
    value, value2 = step.get("value"), step.get("value2")
    if op in ("in", "not_in") and not isinstance(value, (list, tuple)):
        value = [value]
    matches = COMPARISONS[op](df[column], value, value2).fillna(False)

    action = step.get("action", "keep")
    if action not in ("keep", "drop"):
        raise StepError("\"action\" must be keep or drop.")
    result = df[matches] if action == "keep" else df[~matches]
    shown = f" {value} and {value2}" if op == "between" else ("" if op in ("is_missing", "not_missing") else f" {value}")
    return result, (f"{'Kept' if action == 'keep' else 'Removed'} {_plural(int(matches.sum()) if action == 'keep' else int(matches.sum()), 'row')} "
                    f"where {column} {COMPARISON_WORDS[op]}{shown} ({len(result)} of {len(df)} rows remain).")


def step_select_columns(df, step):
    """Keep only some columns, or drop some."""
    columns = _columns_of(df, step)
    action = step.get("action", "keep")
    if action == "keep":
        result = df[columns]
        return result, f"Kept {_plural(len(columns), 'column')}: {', '.join(map(str, columns))}."
    if action == "drop":
        result = df.drop(columns=columns)
        return result, f"Removed {_plural(len(columns), 'column')}: {', '.join(map(str, columns))}."
    raise StepError("\"action\" must be keep or drop.")


def step_rename_columns(df, step):
    """Give columns clearer names."""
    mapping = step.get("mapping") or {}
    if not mapping:
        raise StepError("This step needs a \"mapping\" of old to new column names.")
    unknown = [old for old in mapping if old not in df.columns]
    if unknown:
        raise StepError(f"Column(s) not in the dataset: {', '.join(map(str, unknown))}.")
    result = df.rename(columns=mapping)
    pairs = ", ".join(f"{old} → {new}" for old, new in mapping.items())
    return result, f"Renamed {_plural(len(mapping), 'column')}: {pairs}."


def step_cast_type(df, step):
    """Read columns as numbers, text, dates or categories."""
    columns = _columns_of(df, step)
    target = step.get("to", "numeric")
    result = df.copy()
    notes = []
    for name in columns:
        before_missing = int(result[name].isna().sum())
        if target == "numeric":
            result[name] = as_number(result[name])
        elif target == "integer":
            numbers = as_number(result[name])
            result[name] = numbers.round().astype("Int64")
        elif target == "text":
            result[name] = result[name].astype(str).where(result[name].notna())
        elif target == "category":
            result[name] = result[name].astype("category")
        elif target == "datetime":
            result[name] = pd.to_datetime(result[name], errors="coerce", dayfirst=bool(step.get("dayfirst", False)))
        else:
            raise StepError(f"Unknown target type: {target}.")
        lost = int(result[name].isna().sum()) - before_missing
        notes.append(f"{name}" + (f" ({lost} value(s) could not be read and are now missing)" if lost > 0 else ""))
    return result, f"Converted {_plural(len(columns), 'column')} to {target}: {', '.join(notes)}."


def step_clean_text(df, step):
    """Tidy label columns: trim spaces, collapse runs of spaces, unify capitalisation."""
    columns = _columns_of(df, step, default_all=True, kind="categorical")
    case = step.get("case", "none")
    if case not in ("none", "lower", "upper", "title"):
        raise StepError("\"case\" must be none, lower, upper or title.")
    result = df.copy()
    changed = 0
    for name in columns:
        original = result[name]
        text = original.astype(str)
        if step.get("trim", True):
            text = text.str.strip()
        if step.get("collapse_spaces", True):
            text = text.str.replace(r"\s+", " ", regex=True)
        if case == "lower":
            text = text.str.lower()
        elif case == "upper":
            text = text.str.upper()
        elif case == "title":
            text = text.str.title()
        if step.get("blanks_to_missing", True):
            text = text.mask(text.str.strip().str.lower().isin(MISSING_TOKENS))
        text = text.where(original.notna())
        changed += int((text.fillna("\x00") != original.astype(str).where(original.notna()).fillna("\x00")).sum())
        result[name] = text
    return result, f"Tidied the labels in {_plural(len(columns), 'column')}; {_plural(changed, 'value')} changed."


def step_recode_values(df, step):
    """Replace values in a column, optionally into a new column."""
    column = _columns_of(df, step, key="column")[0]
    mapping = step.get("mapping") or {}
    if not mapping:
        raise StepError("This step needs a \"mapping\" of old to new values.")
    target = step.get("new_column") or column
    result = df.copy()
    keys = {str(k): v for k, v in mapping.items()}
    recoded = result[column].astype(str).map(keys)
    default = step.get("default", "keep")
    recoded = recoded.where(recoded.notna(), result[column] if default == "keep" else default)
    result[target] = recoded.where(result[column].notna())
    hit = int(result[column].astype(str).isin(keys).sum())
    where = f" into a new column {target}" if target != column else ""
    return result, f"Recoded {_plural(hit, 'value')} in {column}{where} ({len(keys)} rule(s))."


def step_bin_numeric(df, step):
    """Turn a measurement into classes (low/medium/high, quartiles, or your own edges)."""
    column = _columns_of(df, step, key="column")[0]
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise StepError(f"\"{column}\" must be numeric to be grouped into classes.")
    method = step.get("method", "equal_width")
    labels = step.get("labels") or None
    target = step.get("new_column") or f"{column}_class"
    values = df[column].astype(float)
    result = df.copy()

    if method == "equal_width":
        bins = int(step.get("bins", 3))
        classes = pd.cut(values, bins=bins, labels=labels, include_lowest=True)
        how = f"{bins} equal-width classes"
    elif method == "quantile":
        bins = int(step.get("bins", 4))
        classes = pd.qcut(values, q=bins, labels=labels, duplicates="drop")
        how = f"{bins} equal-sized groups"
    elif method == "custom":
        edges = step.get("edges")
        if not edges or len(edges) < 2:
            raise StepError("Custom classes need at least two edges.")
        classes = pd.cut(values, bins=[float(e) for e in edges], labels=labels, include_lowest=True)
        how = f"the edges {', '.join(str(e) for e in edges)}"
    else:
        raise StepError(f"Unknown binning method: {method}.")

    result[target] = classes.astype(object).where(classes.notna())
    return result, f"Grouped {column} into {target} using {how}."


def step_handle_outliers(df, step):
    """Deal with extreme values: remove the rows, blank them, or pull them back to the fence."""
    columns = _columns_of(df, step, default_all=True, kind="numeric")
    method = step.get("method", "iqr")
    threshold = step.get("threshold")
    action = step.get("action", "flag")
    group_by = step.get("group_by") or None
    if group_by and group_by not in df.columns:
        raise StepError(f"Grouping column not in the dataset: {group_by}.")

    result = df.copy()
    flagged = pd.Series(False, index=df.index)
    per_column = {}
    for name in columns:
        if not pd.api.types.is_numeric_dtype(result[name]):
            raise StepError(f"\"{name}\" is not numeric.")
        if group_by:
            mask = result.groupby(group_by, dropna=False)[name].transform(
                lambda values: outlier_mask(values, method, threshold))
            mask = mask.fillna(False).astype(bool)
        else:
            mask = outlier_mask(result[name], method, threshold)
        per_column[name] = int(mask.sum())
        flagged |= mask

        if action == "to_missing":
            result.loc[mask, name] = np.nan
        elif action == "winsorize":
            numbers = result[name].astype(float)
            inside = numbers[~mask].dropna()
            if not inside.empty:
                result[name] = numbers.clip(lower=inside.min(), upper=inside.max())
        elif action == "flag":
            result[f"{name}_is_outlier"] = mask
        elif action != "remove":
            raise StepError("\"action\" must be flag, remove, to_missing or winsorize.")

    if action == "remove":
        result = result[~flagged]

    total = int(sum(per_column.values()))
    detail = ", ".join(f"{name}: {count}" for name, count in per_column.items() if count)
    done = {
        "remove": f"removed {_plural(int(flagged.sum()), 'row')}",
        "to_missing": "set them to missing",
        "winsorize": "pulled them back to the nearest value inside the fence",
        "flag": "marked them in a new column per variable",
    }[action]
    where = f" ({detail})" if detail else ""
    rule = {"iqr": "1.5 × IQR", "zscore": "z-score", "mad": "modified z-score"}[method]
    grouped = f" within each {group_by}" if group_by else ""
    return result, f"Found {_plural(total, 'extreme value')} by the {rule} rule{grouped}{where} and {done}."


def step_sort_rows(df, step):
    """Put the rows in order."""
    columns = _columns_of(df, step)
    ascending = step.get("ascending", True)
    result = df.sort_values(by=columns, ascending=ascending, kind="mergesort")
    return result, f"Sorted the rows by {', '.join(map(str, columns))} ({'ascending' if ascending else 'descending'})."


def step_to_long(df, step):
    """Wide to long: one row per observation, as most analyses expect."""
    id_columns = _columns_of(df, step, key="id_columns", default_all=False)
    value_columns = _columns_of(df, step, key="value_columns", default_all=True) if step.get("value_columns") \
        else [c for c in df.columns if c not in id_columns]
    if not value_columns:
        raise StepError("There are no columns left to stack into long format.")
    variable_name = step.get("variable_name") or "variable"
    value_name = step.get("value_name") or "value"
    result = df.melt(id_vars=id_columns, value_vars=value_columns,
                     var_name=variable_name, value_name=value_name)
    if step.get("drop_missing", False):
        result = result.dropna(subset=[value_name])
    return result, (f"Stacked {_plural(len(value_columns), 'column')} into {variable_name}/{value_name}: "
                    f"{len(df)} rows became {len(result)}.")


def step_to_wide(df, step):
    """Long to wide: one row per unit, one column per level."""
    index = _columns_of(df, step, key="index")
    spread = _columns_of(df, step, key="columns_from", default_all=False)[0]
    values = _columns_of(df, step, key="values_from", default_all=False)[0]
    aggfunc = step.get("aggfunc", "mean")
    if aggfunc not in ("mean", "median", "sum", "count", "first", "max", "min"):
        raise StepError(f"Unknown way to combine repeats: {aggfunc}.")
    try:
        table = pd.pivot_table(df, index=index, columns=spread, values=values, aggfunc=aggfunc, observed=False)
    except Exception as error:  # pandas raises for non-numeric values with a numeric aggregate
        raise StepError(f"The table could not be spread: {error}")
    result = table.reset_index()
    result.columns = [str(c) for c in result.columns]
    return result, (f"Spread {values} across one column per {spread} ({aggfunc} of repeats): "
                    f"{len(df)} rows became {len(result)}.")


def step_merge(df, step, extra=None):
    """Join a second table onto this one (plot list, weather, laboratory results)."""
    if extra is None:
        raise StepError("This step needs a second file to join.")
    on = _columns_of(df, step, key="on")
    missing = [c for c in on if c not in extra.columns]
    if missing:
        raise StepError(f"The second file has no column(s): {', '.join(map(str, missing))}.")
    how = step.get("how", "left")
    if how not in ("left", "inner", "outer", "right"):
        raise StepError("\"how\" must be left, inner, outer or right.")
    before = len(df)
    result = df.merge(extra, on=on, how=how, suffixes=("", step.get("suffix", "_2")))
    added = [str(c) for c in result.columns if c not in df.columns]
    matched = int(result[added[0]].notna().sum()) if added else len(result)
    return result, (f"Joined the second file on {', '.join(map(str, on))} ({how} join): "
                    f"{_plural(len(added), 'column')} added, {matched} of {before} rows matched.")


def step_compute_column(df, step):
    """Add (or overwrite) a column from a formula, such as yield per hectare from plot yield and plot area."""
    name = str(step.get("name") or "").strip()
    if not name:
        raise StepError("Give the new column a name.")
    try:
        values = expressions.evaluate(df, step.get("formula"))
    except expressions.FormulaError as error:
        raise StepError(str(error))
    result = df.copy()
    existed = name in df.columns
    result[name] = values
    undefined = int(values.isna().sum())
    note = (f" {_plural(undefined, 'row')} {'has' if undefined == 1 else 'have'} no value (a missing input, division by zero or the log of a "
            f"non-positive number)." if undefined else "")
    verb = "Replaced" if existed else "Added"
    return result, f"{verb} {name} = {str(step.get('formula')).strip()}.{note}"


def step_impute_model(df, step):
    """Fill gaps with a model's prediction (chained regressions or nearest neighbours) - a single imputation."""
    columns = _columns_of(df, step)
    if step.get("predictors"):
        predictors, skipped = _columns_of(df, step, key="predictors"), []
    else:
        predictors, skipped = imputation.default_predictors(df, columns)
    method = step.get("method", "iterative")
    try:
        result = imputation.impute(
            df, columns, predictors, method=method, seed=int(step.get("seed", 2026)),
            max_iter=int(step.get("max_iter", 10)), n_neighbors=int(step.get("n_neighbors", 5)),
            bounds=step.get("bounds", "none"),
        )
    except imputation.ImputationError as error:
        raise StepError(str(error))
    filled = int(df[columns].isna().sum().sum() - result[columns].isna().sum().sum())
    how = ("chained regressions (IterativeImputer, Bayesian ridge)" if method == "iterative"
           else f"the {int(step.get('n_neighbors', 5))} nearest complete rows")
    using = f" using {', '.join(map(str, predictors))}" if predictors else ""
    if skipped:
        using += f" ({', '.join(map(str, skipped))} left out: a different whole number on every row looks like an identifier)"
    return result, (f"Filled {_plural(filled, 'missing value')} in {', '.join(map(str, columns))} with {how}{using}. "
                    f"This is a single imputation: analyses of the filled table will understate uncertainty.")


STEPS = {
    "compute_column": step_compute_column,
    "impute_model": step_impute_model,
    "drop_missing": step_drop_missing,
    "fill_missing": step_fill_missing,
    "drop_duplicates": step_drop_duplicates,
    "filter_rows": step_filter_rows,
    "select_columns": step_select_columns,
    "rename_columns": step_rename_columns,
    "cast_type": step_cast_type,
    "clean_text": step_clean_text,
    "recode_values": step_recode_values,
    "bin_numeric": step_bin_numeric,
    "handle_outliers": step_handle_outliers,
    "sort_rows": step_sort_rows,
    "to_long": step_to_long,
    "to_wide": step_to_wide,
    "merge": step_merge,
}


def apply_steps(df, steps, extra=None):
    """Run a recipe over the table, returning the result and what each step did.

    `extra` is the second table a merge step joins on. A step that fails stops the recipe: the caller sees which
    step failed and why, with the log of everything that had already been done.
    """
    result = df.copy()
    log = []
    for position, step in enumerate(steps, start=1):
        kind = (step or {}).get("type")
        if kind not in STEPS:
            raise StepError(f"Step {position}: unknown step \"{kind}\".")
        rows_before, columns_before = result.shape
        try:
            if kind == "merge":
                result, message = step_merge(result, step, extra)
            else:
                result, message = STEPS[kind](result, step)
        except StepError as error:
            raise StepError(f"Step {position} ({kind}): {error}")
        except Exception as error:
            raise StepError(f"Step {position} ({kind}) could not be carried out: {error}")
        if result.empty:
            raise StepError(f"Step {position} ({kind}) would leave no rows: {message}")
        log.append({
            "step": position,
            "type": kind,
            "message": message,
            "rows_before": int(rows_before),
            "rows_after": int(len(result)),
            "columns_before": int(columns_before),
            "columns_after": int(result.shape[1]),
        })
    return result.reset_index(drop=True), log
