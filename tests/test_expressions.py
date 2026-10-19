"""Calculated-column formulas: arithmetic, functions, conditions, and the things a formula must not be able to do.

Run from backend/:  python -m unittest discover -s tests -t .
"""
import json
import unittest

from tests.support import post_csv  # must come before other app imports

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app import dataprep as dp  # noqa: E402
from app.expressions import FormulaError, evaluate  # noqa: E402


def plots():
    return pd.DataFrame({
        "Plot": [1, 2, 3, 4],
        "Treatment": ["Control", "N60", "N120", "Control"],
        "Plot yield (kg)": [4.2, 5.1, 6.3, np.nan],
        "Plot area (m2)": [20.0, 20.0, 25.0, 20.0],
        "Rep1": [10.0, 12.0, 14.0, 9.0],
        "Rep2": [11.0, np.nan, 15.0, 10.0],
        "Germinated": [45, 38, 50, 0],
        "Sown": [50, 50, 50, 50],
    })


class ArithmeticTests(unittest.TestCase):
    def test_yield_per_hectare_with_bracketed_names(self):
        df = plots()
        got = evaluate(df, "[Plot yield (kg)] / [Plot area (m2)] * 10000")
        expected = df["Plot yield (kg)"] / df["Plot area (m2)"] * 10000
        pd.testing.assert_series_equal(got, expected, check_names=False)
        self.assertTrue(np.isnan(got[3]))  # a missing input gives a missing result

    def test_precedence_powers_and_unary_minus(self):
        df = plots()
        pd.testing.assert_series_equal(evaluate(df, "-Rep1 + 2 * Rep1 ** 2"), -df["Rep1"] + 2 * df["Rep1"] ** 2,
                                       check_names=False)
        self.assertEqual(list(evaluate(df, "(Rep1 + 2) % 3")), list((df["Rep1"] + 2) % 3))

    def test_division_by_zero_and_log_of_zero_become_missing(self):
        df = plots()
        ratio = evaluate(df, "Sown / Germinated")
        self.assertTrue(np.isnan(ratio[3]))
        self.assertTrue(np.isnan(evaluate(df, "log(Germinated)")[3]))
        self.assertTrue(evaluate(df, "1 / 0").isna().all())
        self.assertTrue(evaluate(df, "9.0 ** 9 ** 9").isna().all())

    def test_a_single_number_fills_the_column(self):
        self.assertEqual(list(evaluate(plots(), "2.5")), [2.5] * 4)


class FunctionTests(unittest.TestCase):
    def test_maths_functions_match_numpy(self):
        df = plots()
        rep = df["Rep1"]
        for formula, expected in [
            ("log10(Rep1)", np.log10(rep)), ("ln(Rep1)", np.log(rep)), ("sqrt(Rep1)", np.sqrt(rep)),
            ("round(Rep1 / 3, 2)", np.round(rep / 3, 2)), ("clip(Rep1, 10, 12)", np.clip(rep, 10, 12)),
            ("arcsin_sqrt(Germinated / Sown)", np.arcsin(np.sqrt(df["Germinated"] / df["Sown"]))),
        ]:
            pd.testing.assert_series_equal(evaluate(df, formula), expected.astype(float), check_names=False, obj=formula)

    def test_row_functions_skip_gaps_like_a_spreadsheet(self):
        df = plots()
        self.assertEqual(list(evaluate(df, "row_mean(Rep1, Rep2)")), [10.5, 12.0, 14.5, 9.5])
        self.assertEqual(list(evaluate(df, "row_sum(Rep1, Rep2)")), [21.0, 12.0, 29.0, 19.0])
        self.assertEqual(list(evaluate(df, "row_max(Rep1, Rep2, 13)")), [13.0, 13.0, 15.0, 13.0])

    def test_column_summaries_inside_a_row_formula(self):
        df = plots()
        relative = evaluate(df, "100 * Rep1 / mean(Rep1)")
        pd.testing.assert_series_equal(relative, 100 * df["Rep1"] / df["Rep1"].mean(), check_names=False)
        z = evaluate(df, "zscore(Rep1)")
        self.assertAlmostEqual(z.mean(), 0.0)
        self.assertAlmostEqual(z.std(ddof=1), 1.0)

    def test_conditions_and_text_comparisons(self):
        df = plots()
        self.assertEqual(list(evaluate(df, 'if_else(Treatment == "Control", 0, Rep1)')), [0.0, 12.0, 14.0, 0.0])
        self.assertEqual(list(evaluate(df, "Rep1 if Rep1 > 11 else -1")), [-1.0, 12.0, 14.0, -1.0])
        self.assertEqual(list(evaluate(df, "(Rep1 >= 10) and (Rep1 < 14)")), [True, True, False, False])
        self.assertEqual(list(evaluate(df, "fill(Rep2, 0)")), [11.0, 0.0, 15.0, 10.0])
        self.assertEqual(list(evaluate(df, "is_missing(Rep2)")), [False, True, False, False])


class SafetyTests(unittest.TestCase):
    def test_nothing_outside_the_formula_language_runs(self):
        df = plots()
        for formula in [
            "__import__('os').system('echo hi')",
            "open('secrets.txt')",
            "Rep1.__class__",
            "[x for x in Rep1]",
            "lambda: 1",
            "eval('1')",
            "Rep1; Rep2",
            "globals()",
        ]:
            with self.assertRaises(FormulaError, msg=formula):
                evaluate(df, formula)

    def test_helpful_messages(self):
        df = plots()
        cases = {
            "Yeild * 2": "no column called \"Yeild\"",
            "[Plot yeild] * 2": "no column called \"Plot yeild\"",
            "median": "is a function",
            "Rep1 +": "could not be read",
            'Treatment + 1': "not numeric",
            'Treatment > "N60"': "only be compared with == or !=",
            "power(Rep1, 2)": "not a function",
            "round()": "wrong number",
            "": "Write a formula",
        }
        for formula, message in cases.items():
            with self.assertRaises(FormulaError, msg=formula) as caught:
                evaluate(df, formula)
            self.assertIn(message, str(caught.exception), formula)


class StepTests(unittest.TestCase):
    def test_compute_step_adds_a_column_and_explains_gaps(self):
        out, log = dp.apply_steps(plots(), [{"type": "compute_column", "name": "Yield (kg/ha)",
                                             "formula": "[Plot yield (kg)] / [Plot area (m2)] * 10000"}])
        self.assertAlmostEqual(out["Yield (kg/ha)"][0], 2100.0)
        self.assertIn("Added Yield (kg/ha) =", log[0]["message"])
        self.assertIn("1 row has no value", log[0]["message"])

    def test_later_steps_can_use_the_new_column(self):
        out, log = dp.apply_steps(plots(), [
            {"type": "compute_column", "name": "Germination", "formula": "100 * Germinated / Sown"},
            {"type": "filter_rows", "column": "Germination", "op": "ge", "value": 80},
        ])
        # germination is 90, 76, 100 and 0 %
        self.assertEqual(list(out["Plot"]), [1, 3])

    def test_bad_formula_names_the_step(self):
        with self.assertRaisesRegex(dp.StepError, r"Step 1 \(compute_column\): There is no column"):
            dp.apply_steps(plots(), [{"type": "compute_column", "name": "x", "formula": "Nitrogen * 2"}])

    def test_through_the_api(self):
        steps = [{"type": "compute_column", "name": "Mean", "formula": "row_mean(Rep1, Rep2)"}]
        code, body = post_csv("/analyze/prep/apply", plots(), steps=json.dumps(steps))
        self.assertEqual(code, 200, body)
        self.assertIn("Mean", body["preview"]["columns"])
        code, body = post_csv("/analyze/prep/apply", plots(),
                              steps=json.dumps([{"type": "compute_column", "name": "x", "formula": "open('a')"}]))
        self.assertEqual(code, 400)

    def test_formula_check_endpoint(self):
        code, body = post_csv("/analyze/prep/formula", plots(), formula="[Plot yield (kg)] * 2")
        self.assertEqual(code, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(body["sample"], [8.4, 10.2, 12.6, None])
        self.assertEqual(body["missing"], 1)
        # a column created by an earlier step can be used
        earlier = [{"type": "rename_columns", "mapping": {"Rep1": "First"}}]
        code, body = post_csv("/analyze/prep/formula", plots(), formula="First + 1", steps=json.dumps(earlier))
        self.assertEqual(body["sample"][:2], [11.0, 13.0])
        code, body = post_csv("/analyze/prep/formula", plots(), formula="Rep9 + 1")
        self.assertFalse(body["ok"])
        self.assertIn("Rep9", body["message"])


if __name__ == "__main__":
    unittest.main()
