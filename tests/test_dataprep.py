"""Data preparation: the dataset profile, every recipe step, and the /prep endpoints.

Each step is checked against the pandas operation it stands for, or against values worked out by hand.
Run from backend/:  python -m unittest discover -s tests -t .
"""
import io
import json
import unittest

from tests.support import client_and_headers, post_csv  # must come before other app imports

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app import dataprep as dp  # noqa: E402


def field_sheet():
    """A typical messy field sheet: stray spaces, NA as text, a comma in a number, a repeated row."""
    return pd.DataFrame({
        "Plot": [1, 2, 3, 4, 5, 6, 6],
        "Treatment": ["T1 ", "t1", "T2", "T2", "NA", "T3", "T3"],
        "Yield": ["1,200", "1250", "1300", None, "1310", "1280", "1280"],
        "Height": [88.0, 90.0, np.nan, 92.0, 95.0, 91.0, 91.0],
    })


def run(df, *steps, extra=None):
    return dp.apply_steps(df, list(steps), extra=extra)


class ProfileTests(unittest.TestCase):
    def test_counts_and_flags_the_usual_problems(self):
        report = dp.profile(field_sheet())
        self.assertEqual(report["n_rows"], 7)
        self.assertEqual(report["duplicate_rows"], 1)
        self.assertEqual(report["missing_cells"], 2)
        columns = {c["name"]: c for c in report["columns"]}

        treatment = {i["code"] for i in columns["Treatment"]["issues"]}
        self.assertIn("missing_as_text", treatment)
        self.assertIn("label_variants", treatment)
        self.assertIn(["T1 ", "t1"], columns["Treatment"]["variants"])

        self.assertEqual(columns["Yield"]["kind"], "categorical")
        self.assertIn("numeric_as_text", {i["code"] for i in columns["Yield"]["issues"]})
        self.assertEqual(columns["Yield"]["numeric_ready"], 6)

        height = columns["Height"]
        self.assertEqual(height["kind"], "numeric")
        self.assertAlmostEqual(height["median"], 91.0)
        self.assertAlmostEqual(height["sd"], pd.Series([88, 90, 92, 95, 91, 91.0]).std(ddof=1))

    def test_constant_and_empty_columns(self):
        df = pd.DataFrame({"Site": ["A"] * 4, "Notes": [np.nan] * 4, "x": [1, 2, 3, 4]})
        codes = {c["name"]: {i["code"] for i in c["issues"]} for c in dp.profile(df)["columns"]}
        self.assertEqual(codes["Site"], {"constant"})
        self.assertEqual(codes["Notes"], {"all_missing"})
        self.assertEqual(codes["x"], set())

    def test_outlier_rules_match_their_definitions(self):
        values = pd.Series([10, 11, 12, 11, 10, 12, 11, 40.0])
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = (values < q1 - 1.5 * (q3 - q1)) | (values > q3 + 1.5 * (q3 - q1))
        self.assertTrue((dp.outlier_mask(values, "iqr") == iqr).all())

        z = (values - values.mean()).abs() / values.std(ddof=1)
        self.assertTrue((dp.outlier_mask(values, "zscore", 2) == (z > 2)).all())

        median = values.median()
        mad = (values - median).abs().median()
        modified = (0.6745 * (values - median)).abs() / mad
        self.assertTrue((dp.outlier_mask(values, "mad") == (modified > 3.5)).all())
        self.assertEqual(list(values[dp.outlier_mask(values, "mad")]), [40.0])

    def test_preview_turns_missing_into_null(self):
        shown = dp.preview(field_sheet(), rows=4)
        self.assertEqual(shown["total"], 7)
        self.assertEqual(shown["showing"], 4)
        self.assertIsNone(shown["rows"][2]["Height"])
        json.dumps(shown)  # must be plain JSON


class MissingValueTests(unittest.TestCase):
    def test_drop_rows_with_any_or_selected_gaps(self):
        df = field_sheet()
        out, log = run(df, {"type": "drop_missing", "axis": "rows"})
        self.assertEqual(len(out), len(df.dropna()))
        self.assertEqual(log[0]["message"], "Removed 2 rows with any missing values.")
        out, log = run(df, {"type": "drop_missing", "axis": "rows", "columns": ["Height"]})
        self.assertEqual(len(out), 6)
        self.assertEqual(log[0]["message"], "Removed 1 row with any missing values (looking at Height).")

    def test_drop_columns_over_a_missing_share(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, np.nan, np.nan, np.nan], "c": [1, np.nan, 3, 4]})
        out, log = run(df, {"type": "drop_missing", "axis": "columns", "max_missing_fraction": 0.5})
        self.assertEqual(list(out.columns), ["a", "c"])
        self.assertIn("(b)", log[0]["message"])

    def test_fill_with_mean_median_mode_and_constant(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0, 10.0], "g": ["a", None, "a", "b"]})
        self.assertAlmostEqual(run(df, {"type": "fill_missing", "columns": ["x"], "method": "mean"})[0]["x"][1],
                               np.mean([1, 3, 10]))
        self.assertAlmostEqual(run(df, {"type": "fill_missing", "columns": ["x"], "method": "median"})[0]["x"][1], 3.0)
        self.assertEqual(run(df, {"type": "fill_missing", "columns": ["g"], "method": "mode"})[0]["g"][1], "a")
        self.assertEqual(run(df, {"type": "fill_missing", "columns": ["x"], "method": "constant", "value": 0})[0]["x"][1], 0)

    def test_fill_by_group_uses_each_treatment_mean(self):
        df = pd.DataFrame({"Treatment": ["T1", "T1", "T1", "T2", "T2", "T2"],
                           "Yield": [4.0, np.nan, 6.0, 9.0, 11.0, np.nan]})
        out, log = run(df, {"type": "fill_missing", "columns": ["Yield"], "method": "mean", "group_by": "Treatment"})
        self.assertEqual(list(out["Yield"]), [4.0, 5.0, 6.0, 9.0, 11.0, 10.0])
        self.assertIn("Treatment group", log[0]["message"])

    def test_interpolate_and_carry_forward(self):
        df = pd.DataFrame({"t": [1.0, np.nan, np.nan, 4.0]})
        self.assertEqual(list(run(df, {"type": "fill_missing", "columns": ["t"], "method": "interpolate"})[0]["t"]),
                         [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(list(run(df, {"type": "fill_missing", "columns": ["t"], "method": "ffill"})[0]["t"]),
                         [1.0, 1.0, 1.0, 4.0])

    def test_mean_of_text_is_refused(self):
        with self.assertRaisesRegex(dp.StepError, "not numeric"):
            run(field_sheet(), {"type": "fill_missing", "columns": ["Treatment"], "method": "mean"})


class RowAndColumnTests(unittest.TestCase):
    def test_duplicates(self):
        df = field_sheet()
        out, log = run(df, {"type": "drop_duplicates"})
        self.assertEqual(len(out), len(df.drop_duplicates()))
        self.assertEqual(log[0]["message"], "Removed 1 duplicate row.")
        out, _ = run(df, {"type": "drop_duplicates", "columns": ["Plot"], "keep": "none"})
        self.assertNotIn(6, list(out["Plot"]))

    def test_filters(self):
        df = field_sheet()
        cases = [
            ({"op": "gt", "value": 90}, df["Height"] > 90),
            ({"op": "between", "value": 89, "value2": 92}, df["Height"].between(89, 92)),
            ({"op": "in", "value": ["T2", "T3"]}, df["Treatment"].isin(["T2", "T3"])),
            ({"op": "is_missing"}, df["Height"].isna()),
        ]
        for spec, expected in cases:
            column = "Treatment" if spec["op"] == "in" else "Height"
            out, _ = run(df, {"type": "filter_rows", "column": column, **spec})
            self.assertEqual(len(out), int(expected.sum()), spec)
        out, log = run(df, {"type": "filter_rows", "column": "Treatment", "op": "contains", "value": "t1", "action": "drop"})
        self.assertEqual(len(out), 5)
        self.assertIn("Removed 2 rows", log[0]["message"])

    def test_select_rename_sort(self):
        df = field_sheet()
        out, _ = run(df,
                     {"type": "select_columns", "columns": ["Plot", "Height"]},
                     {"type": "rename_columns", "mapping": {"Height": "Plant height (cm)"}},
                     {"type": "sort_rows", "columns": ["Plant height (cm)"], "ascending": False})
        self.assertEqual(list(out.columns), ["Plot", "Plant height (cm)"])
        self.assertEqual(out["Plant height (cm)"].iloc[0], 95.0)

    def test_unknown_column_names_the_step(self):
        with self.assertRaisesRegex(dp.StepError, r"Step 2 \(select_columns\).*Moisture"):
            run(field_sheet(), {"type": "sort_rows", "columns": ["Plot"]},
                {"type": "select_columns", "columns": ["Moisture"]})

    def test_a_step_that_empties_the_table_is_refused(self):
        with self.assertRaisesRegex(dp.StepError, "no rows"):
            run(field_sheet(), {"type": "filter_rows", "column": "Height", "op": "gt", "value": 1000})


class CleaningTests(unittest.TestCase):
    def test_numbers_stored_as_text_become_numbers(self):
        out, log = run(field_sheet(), {"type": "cast_type", "columns": ["Yield"], "to": "numeric"})
        self.assertEqual(list(out["Yield"].dropna()), [1200.0, 1250.0, 1300.0, 1310.0, 1280.0, 1280.0])
        df = pd.DataFrame({"w": ["12 kg", "7.5kg", "abc"]})
        out, log = run(df, {"type": "cast_type", "columns": ["w"], "to": "numeric"})
        self.assertEqual(list(out["w"][:2]), [12.0, 7.5])
        self.assertIn("1 value(s) could not be read", log[0]["message"])

    def test_label_tidying_merges_variants_and_blanks(self):
        out, _ = run(field_sheet(), {"type": "clean_text", "columns": ["Treatment"], "case": "upper"})
        self.assertEqual(list(out["Treatment"][:4]), ["T1", "T1", "T2", "T2"])
        self.assertTrue(pd.isna(out["Treatment"][4]))  # "NA" as text is now missing

    def test_recode_into_a_new_column(self):
        df = pd.DataFrame({"Irrigation": ["I1", "I2", "I3", None]})
        out, log = run(df, {"type": "recode_values", "column": "Irrigation",
                            "mapping": {"I1": "Rainfed", "I2": "Irrigated"}, "new_column": "Water"})
        self.assertEqual(list(out["Water"][:3]), ["Rainfed", "Irrigated", "I3"])
        self.assertTrue(pd.isna(out["Water"][3]))
        self.assertIn("Recoded 2 values", log[0]["message"])

    def test_classes_match_pandas_cut_and_qcut(self):
        df = pd.DataFrame({"pH": [4.8, 5.5, 6.1, 6.8, 7.2, 7.9, 8.4, 8.9]})
        out, _ = run(df, {"type": "bin_numeric", "column": "pH", "method": "custom",
                          "edges": [0, 6.5, 7.5, 14], "labels": ["Acidic", "Neutral", "Alkaline"]})
        self.assertEqual(list(out["pH_class"]), ["Acidic"] * 3 + ["Neutral"] * 2 + ["Alkaline"] * 3)
        out, _ = run(df, {"type": "bin_numeric", "column": "pH", "method": "quantile", "bins": 4})
        expected = pd.qcut(df["pH"], 4).astype(object)
        self.assertEqual(list(out["pH_class"].astype(str)), list(expected.astype(str)))

    def test_outlier_actions(self):
        df = pd.DataFrame({"Rep": list("AAAABBBB"), "y": [10, 11, 12, 11, 20, 21, 22, 90.0]})
        removed, _ = run(df, {"type": "handle_outliers", "columns": ["y"], "method": "iqr", "action": "remove"})
        self.assertNotIn(90.0, list(removed["y"]))
        blanked, _ = run(df, {"type": "handle_outliers", "columns": ["y"], "method": "iqr", "action": "to_missing"})
        self.assertEqual(int(blanked["y"].isna().sum()), 1)
        capped, _ = run(df, {"type": "handle_outliers", "columns": ["y"], "method": "iqr", "action": "winsorize"})
        self.assertEqual(capped["y"].max(), 22.0)
        flagged, _ = run(df, {"type": "handle_outliers", "columns": ["y"], "method": "iqr", "action": "flag"})
        self.assertEqual(int(flagged["y_is_outlier"].sum()), 1)

    def test_outliers_within_groups(self):
        # 20 is ordinary among B's values but extreme for A's
        df = pd.DataFrame({"Rep": ["A"] * 6 + ["B"] * 6,
                           "y": [10, 11, 10, 11, 10, 20, 19, 20, 21, 20, 19, 21.0]})
        pooled, _ = run(df, {"type": "handle_outliers", "columns": ["y"], "method": "iqr", "action": "flag"})
        grouped, log = run(df, {"type": "handle_outliers", "columns": ["y"], "method": "iqr", "action": "flag",
                                "group_by": "Rep"})
        self.assertEqual(int(pooled["y_is_outlier"].sum()), 0)
        self.assertEqual(list(grouped.index[grouped["y_is_outlier"]]), [5])
        self.assertIn("within each Rep", log[0]["message"])


class ReshapeTests(unittest.TestCase):
    def test_wide_to_long_and_back(self):
        wide = pd.DataFrame({"Genotype": ["G1", "G2"], "Env1": [4.1, 5.0], "Env2": [3.8, 4.6]})
        long, log = run(wide, {"type": "to_long", "id_columns": ["Genotype"], "variable_name": "Environment",
                               "value_name": "Yield"})
        self.assertEqual(len(long), 4)
        self.assertEqual(set(long["Environment"]), {"Env1", "Env2"})
        again, _ = run(long, {"type": "to_wide", "index": ["Genotype"], "columns_from": "Environment",
                              "values_from": "Yield", "aggfunc": "first"})
        pd.testing.assert_frame_equal(again[["Genotype", "Env1", "Env2"]], wide, check_names=False)
        self.assertIn("2 rows became 4", log[0]["message"])

    def test_spreading_averages_replicates(self):
        long = pd.DataFrame({"G": ["G1", "G1", "G2", "G2"], "Rep": ["R1", "R1", "R1", "R1"], "y": [4.0, 6.0, 1.0, 3.0]})
        wide, _ = run(long, {"type": "to_wide", "index": ["G"], "columns_from": "Rep", "values_from": "y"})
        self.assertEqual(list(wide["R1"]), [5.0, 2.0])

    def test_merge_a_second_table(self):
        plots = pd.DataFrame({"Plot": [1, 2, 3], "Yield": [4.0, 5.0, 6.0]})
        soil = pd.DataFrame({"Plot": [1, 3], "OC": [0.52, 0.61]})
        out, log = run(plots, {"type": "merge", "on": ["Plot"], "how": "left"}, extra=soil)
        self.assertEqual(list(out.columns), ["Plot", "Yield", "OC"])
        self.assertTrue(pd.isna(out["OC"][1]))
        self.assertIn("2 of 3 rows matched", log[0]["message"])
        with self.assertRaisesRegex(dp.StepError, "second file"):
            run(plots, {"type": "merge", "on": ["Plot"]})


class EndpointTests(unittest.TestCase):
    def test_profile_endpoint(self):
        code, body = post_csv("/analyze/prep/profile", field_sheet())
        self.assertEqual(code, 200)
        self.assertEqual(body["profile"]["duplicate_rows"], 1)
        self.assertEqual(body["preview"]["total"], 7)

    def test_apply_endpoint_returns_log_profiles_and_csv(self):
        steps = [{"type": "drop_duplicates"},
                 {"type": "cast_type", "columns": ["Yield"], "to": "numeric"},
                 {"type": "fill_missing", "columns": ["Yield", "Height"], "method": "median"}]
        code, body = post_csv("/analyze/prep/apply", field_sheet(), steps=json.dumps(steps))
        self.assertEqual(code, 200, body)
        self.assertEqual(len(body["log"]), 3)
        self.assertEqual(body["before"]["n_rows"], 7)
        self.assertEqual(body["after"]["n_rows"], 6)
        # "NA" in Treatment is read as missing by the CSV reader and is left alone by these steps
        after = {c["name"]: c for c in body["after"]["columns"]}
        self.assertEqual(after["Yield"]["missing"] + after["Height"]["missing"], 0)
        cleaned = pd.read_csv(io.StringIO(body["csv_content"]))
        self.assertEqual(cleaned["Yield"].dtype.kind, "f")

    def test_apply_endpoint_with_a_merge_file(self):
        client, headers = client_and_headers()
        plots = pd.DataFrame({"Plot": [1, 2], "Yield": [4.0, 5.0]}).to_csv(index=False).encode()
        soil = pd.DataFrame({"Plot": [1, 2], "OC": [0.5, 0.6]}).to_csv(index=False).encode()
        response = client.post("/analyze/prep/apply", headers=headers,
                               files={"file": ("plots.csv", plots, "text/csv"),
                                      "merge_file": ("soil.csv", soil, "text/csv")},
                               data={"steps": json.dumps([{"type": "merge", "on": ["Plot"]}])})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["preview"]["columns"], ["Plot", "Yield", "OC"])

    def test_bad_recipes_are_400(self):
        for steps in ("not json", "[]", json.dumps([{"type": "teleport"}])):
            code, body = post_csv("/analyze/prep/apply", field_sheet(), steps=steps)
            self.assertEqual(code, 400, steps)
            self.assertIn("detail", body)


if __name__ == "__main__":
    unittest.main()
