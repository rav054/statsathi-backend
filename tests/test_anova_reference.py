"""ANOVA endpoint validated against statsmodels / scipy reference implementations.

Runs the real FastAPI app in-process against a throwaway SQLite database.
Run from backend/:  python -m unittest discover -s tests -t .
"""
import unittest
from itertools import combinations

from tests.support import post_csv  # must come before other app imports

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.anova import anova_lm  # noqa: E402
from statsmodels.stats.oneway import anova_oneway  # noqa: E402


def run_anova(df, **form):
    return post_csv("/analyze/anova", df, **form)


# ---------------------------------------------------------------------------
# Synthetic experiments (fixed seeds)
# ---------------------------------------------------------------------------
def factorial_crd(a=3, b=4, reps=3, seed=1):
    rng = np.random.default_rng(seed)
    rows = [{"A": f"A{i + 1}", "B": f"B{j + 1}",
             "Y": 40 + 3 * i + 1.5 * j + (2.0 if (i, j) == (a - 1, b - 1) else 0.0) + rng.normal(0, 1.5)}
            for i in range(a) for j in range(b) for _ in range(reps)]
    return pd.DataFrame(rows)


def three_factor_crd(reps=3, seed=2):
    rng = np.random.default_rng(seed)
    # third factor is "D": a column named "C" would shadow patsy's C() in reference formulas
    rows = [{"A": f"A{i + 1}", "B": f"B{j + 1}", "D": f"D{c + 1}",
             "Y": 10 + 4 * i + 2 * j + c + 3 * (i == 1 and j == 2) + rng.normal(0, 1)}
            for i in range(2) for j in range(3) for c in range(2) for _ in range(reps)]
    return pd.DataFrame(rows)


def rbd_factorial(a=2, b=3, r=4, seed=4):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(r):
        block = rng.normal(0, 2)
        for i in range(a):
            for j in range(b):
                rows.append({"Rep": f"R{k + 1}", "A": f"A{i + 1}", "B": f"B{j + 1}",
                             "Y": 25 + block + 2 * i + 1.2 * j + 1.5 * (i == 1 and j == 2) + rng.normal(0, 1)})
    return pd.DataFrame(rows)


def rbd_oneway(t=5, r=4, seed=3):
    rng = np.random.default_rng(seed)
    rows = [{"Rep": f"R{k + 1}", "T": f"T{i + 1}", "Y": 50 + 1.8 * i + 1.5 * k + rng.normal(0, 1)}
            for k in range(r) for i in range(t)]
    return pd.DataFrame(rows)


def split_plot(a=3, b=4, r=3, seed=5):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(r):
        block = rng.normal(0, 1)
        for i in range(a):
            whole_plot = rng.normal(0, 1.2)
            for j in range(b):
                rows.append({"Rep": f"R{k + 1}", "A": f"I{i + 1}", "B": f"N{j + 1}",
                             "Y": 30 + block + whole_plot + 1.4 * i + 1.1 * j + rng.normal(0, 1)})
    return pd.DataFrame(rows)


def latin_square(t=5, seed=6):
    rng = np.random.default_rng(seed)
    rows = [{"Row": f"R{i + 1}", "Col": f"C{j + 1}", "T": f"T{(i + j) % t + 1}",
             "Y": 20 + 0.8 * i + 0.5 * j + 1.3 * ((i + j) % t) + rng.normal(0, 0.8)}
            for i in range(t) for j in range(t)]
    return pd.DataFrame(rows)


def one_way(sizes=(4, 6, 5, 7, 5), sds=None, seed=7):
    rng = np.random.default_rng(seed)
    sds = sds or [1.0] * len(sizes)
    rows = [{"G": f"G{i + 1}", "Y": v}
            for i, (n, sd) in enumerate(zip(sizes, sds)) for v in rng.normal(10 + 0.7 * i, sd, n)]
    return pd.DataFrame(rows)


def type3(formula, df):
    return anova_lm(smf.ols(formula, df).fit(), typ=3)


class ReferenceAssertions(unittest.TestCase):
    def assertEffect(self, app_row, ref, term):
        self.assertAlmostEqual(app_row["ss"], ref.loc[term, "sum_sq"], places=6, msg=term)
        self.assertEqual(app_row["df"], int(ref.loc[term, "df"]), msg=term)
        self.assertAlmostEqual(app_row["f_statistic"], ref.loc[term, "F"], places=6, msg=term)
        self.assertAlmostEqual(app_row["p_value"], ref.loc[term, "PR(>F)"], places=9, msg=term)


class FactorialCrdTests(ReferenceAssertions):
    def check_two_way(self, df):
        status, res = run_anova(df, test_type="twoway", dep_var="Y", ind_var1="A", ind_var2="B")
        self.assertEqual(status, 200, res)
        ref = type3("Y ~ C(A, Sum) * C(B, Sum)", df)
        table = res["anova_table"]
        self.assertEffect(table["factorA"], ref, "C(A, Sum)")
        self.assertEffect(table["factorB"], ref, "C(B, Sum)")
        self.assertEffect(table["interaction"], ref, "C(A, Sum):C(B, Sum)")
        self.assertAlmostEqual(table["error"]["ss"], ref.loc["Residual", "sum_sq"], places=6)

    def test_two_way_balanced(self):
        self.check_two_way(factorial_crd())

    def test_two_way_unbalanced_type_iii(self):
        self.check_two_way(factorial_crd().drop(index=[0, 4, 13, 22]).reset_index(drop=True))

    def check_three_factor(self, df):
        status, res = run_anova(df, test_type="crd_multifactor", dep_var="Y", factors="A,B,D")
        self.assertEqual(status, 200, res)
        ref = type3("Y ~ C(A, Sum) * C(B, Sum) * C(D, Sum)", df)
        rows = {e["source"]: e for e in res["anova_table"]["effects"]}
        self.assertEqual(len(rows), 7)
        for source, row in rows.items():
            self.assertEffect(row, ref, ":".join(f"C({f}, Sum)" for f in source.split(":")))

    def test_three_factor_balanced(self):
        self.check_three_factor(three_factor_crd())

    def test_three_factor_unbalanced_type_iii(self):
        self.check_three_factor(three_factor_crd().drop(index=[1, 8, 20]).reset_index(drop=True))


class BlockDesignTests(ReferenceAssertions):
    def test_rbd_one_factor(self):
        df = rbd_oneway()
        status, res = run_anova(df, test_type="rbd_oneway", dep_var="Y", ind_var1="T", rep_var="Rep")
        self.assertEqual(status, 200, res)
        ref = anova_lm(smf.ols("Y ~ C(Rep) + C(T)", df).fit(), typ=2)
        t = res["anova_table"]
        self.assertAlmostEqual(t["ss_between"], ref.loc["C(T)", "sum_sq"], places=6)
        self.assertAlmostEqual(t["ss_rep"], ref.loc["C(Rep)", "sum_sq"], places=6)
        self.assertAlmostEqual(t["ss_within"], ref.loc["Residual", "sum_sq"], places=6)
        self.assertAlmostEqual(t["p_value"], ref.loc["C(T)", "PR(>F)"], places=9)

    def test_rbd_two_factor(self):
        df = rbd_factorial()
        status, res = run_anova(df, test_type="rbd_twoway", dep_var="Y", ind_var1="A", ind_var2="B", rep_var="Rep")
        self.assertEqual(status, 200, res)
        ref = anova_lm(smf.ols("Y ~ C(Rep) + C(A) * C(B)", df).fit(), typ=2)
        t = res["anova_table"]
        self.assertEffect(t["factorA"], ref, "C(A)")
        self.assertEffect(t["factorB"], ref, "C(B)")
        self.assertEffect(t["interaction"], ref, "C(A):C(B)")
        self.assertAlmostEqual(t["ss_rep"], ref.loc["C(Rep)", "sum_sq"], places=6)

    def test_multi_factor_rbd(self):
        df = rbd_factorial()
        status, res = run_anova(df, test_type="rbd_multifactor", dep_var="Y", factors="A,B", rep_var="Rep")
        self.assertEqual(status, 200, res)
        ref = type3("Y ~ C(Rep, Sum) + C(A, Sum) * C(B, Sum)", df)
        rows = {e["source"]: e for e in res["anova_table"]["effects"]}
        self.assertEffect(rows["Rep"], ref, "C(Rep, Sum)")
        self.assertEffect(rows["A"], ref, "C(A, Sum)")
        self.assertEffect(rows["B"], ref, "C(B, Sum)")
        self.assertEffect(rows["A:B"], ref, "C(A, Sum):C(B, Sum)")

    def test_latin_square(self):
        df = latin_square()
        status, res = run_anova(df, test_type="lsd", dep_var="Y", ind_var1="T", ind_var2="Col", rep_var="Row")
        self.assertEqual(status, 200, res)
        ref = anova_lm(smf.ols("Y ~ C(Row) + C(Col) + C(T)", df).fit(), typ=2)
        t = res["anova_table"]
        self.assertAlmostEqual(t["ss_row"], ref.loc["C(Row)", "sum_sq"], places=6)
        self.assertAlmostEqual(t["ss_col"], ref.loc["C(Col)", "sum_sq"], places=6)
        self.assertAlmostEqual(t["ss_between"], ref.loc["C(T)", "sum_sq"], places=6)
        self.assertAlmostEqual(t["ss_within"], ref.loc["Residual", "sum_sq"], places=6)
        self.assertAlmostEqual(t["p_value"], ref.loc["C(T)", "PR(>F)"], places=9)

    def test_split_plot_error_terms(self):
        df = split_plot()
        status, res = run_anova(df, test_type="splitplot", dep_var="Y", ind_var1="A", ind_var2="B", rep_var="Rep")
        self.assertEqual(status, 200, res)
        ref = anova_lm(smf.ols("Y ~ C(Rep) + C(A) + C(Rep):C(A) + C(B) + C(A):C(B)", df).fit(), typ=1)
        t = res["anova_table"]
        ms_ea = ref.loc["C(Rep):C(A)", "sum_sq"] / ref.loc["C(Rep):C(A)", "df"]
        self.assertAlmostEqual(t["error_a"]["ss"], ref.loc["C(Rep):C(A)", "sum_sq"], places=6)
        self.assertAlmostEqual(t["error_b"]["ss"], ref.loc["Residual", "sum_sq"], places=6)
        self.assertAlmostEqual(t["factorA"]["ss"], ref.loc["C(A)", "sum_sq"], places=6)
        f_a = (ref.loc["C(A)", "sum_sq"] / ref.loc["C(A)", "df"]) / ms_ea
        self.assertAlmostEqual(t["factorA"]["f_statistic"], f_a, places=6)
        self.assertAlmostEqual(t["factorA"]["p_value"], stats.f.sf(f_a, ref.loc["C(A)", "df"], ref.loc["C(Rep):C(A)", "df"]), places=9)
        self.assertAlmostEqual(t["interaction"]["p_value"], ref.loc["C(A):C(B)", "PR(>F)"], places=9)

    def test_split_plot_lsd_letters_match_cd_within_main_plot(self):
        df = split_plot()
        status, res = run_anova(df, test_type="splitplot", dep_var="Y", ind_var1="A", ind_var2="B",
                                rep_var="Rep", posthoc_method="lsd")
        self.assertEqual(status, 200, res)
        cd = next(c["cd_5"] for c in res["cd_results"] if c["parameter"] == "Sub Plot B at same Main Plot A")
        for comp in res["posthoc_results"]:
            if comp["group1"].split(" / ")[0] == comp["group2"].split(" / ")[0]:
                self.assertEqual(comp["significant"], abs(comp["mean_diff"]) > cd, comp)


class OneWayTests(ReferenceAssertions):
    def test_classic_anova_and_tukey_kramer(self):
        df = one_way()
        status, res = run_anova(df, test_type="oneway", dep_var="Y", ind_var1="G", posthoc_method="tukey")
        self.assertEqual(status, 200, res)
        groups = [g["Y"].values for _, g in df.groupby("G")]
        f, p = stats.f_oneway(*groups)
        self.assertAlmostEqual(res["anova_table"]["f_statistic"], f, places=8)
        self.assertAlmostEqual(res["anova_table"]["p_value"], p, places=10)
        exact = stats.tukey_hsd(*groups)
        for n, (i, j) in enumerate(combinations(range(len(groups)), 2)):
            self.assertAlmostEqual(res["posthoc_results"][n]["p_value"], exact.pvalue[i, j], places=7)

    def test_unequal_variances_reports_welch_without_switching(self):
        df = one_way(sizes=(8, 8, 8, 8), sds=[0.3, 0.5, 3.0, 6.0], seed=8)
        status, res = run_anova(df, test_type="oneway", dep_var="Y", ind_var1="G", posthoc_method="tukey")
        self.assertEqual(status, 200, res)
        self.assertEqual(res["posthoc_method"], "tukey")
        self.assertFalse(res["levene_results"]["equal_var"])
        welch = anova_oneway(df["Y"], df["G"], use_var="unequal")
        self.assertAlmostEqual(res["anova_table"]["welch_anova"]["f_statistic"], welch.statistic, places=6)
        self.assertAlmostEqual(res["anova_table"]["welch_anova"]["p_value"], welch.pvalue, places=8)
        self.assertTrue(any("Welch" in note for note in res["notes"]))


class LetterConsistencyTests(unittest.TestCase):
    CASES = [
        (one_way, dict(test_type="oneway", dep_var="Y", ind_var1="G")),
        (factorial_crd, dict(test_type="twoway", dep_var="Y", ind_var1="A", ind_var2="B")),
        (rbd_oneway, dict(test_type="rbd_oneway", dep_var="Y", ind_var1="T", rep_var="Rep")),
        (latin_square, dict(test_type="lsd", dep_var="Y", ind_var1="T", ind_var2="Col", rep_var="Row")),
        (split_plot, dict(test_type="splitplot", dep_var="Y", ind_var1="A", ind_var2="B", rep_var="Rep")),
        (three_factor_crd, dict(test_type="crd_multifactor", dep_var="Y", factors="A,B,D")),
    ]

    def test_letters_agree_with_reported_pairwise_tests(self):
        for make_data, form in self.CASES:
            methods = ["tukey", "lsd", "duncan", "bonferroni"] + (["games_howell"] if form["test_type"] == "oneway" else [])
            for method in methods:
                with self.subTest(design=form["test_type"], method=method):
                    status, res = run_anova(make_data(), posthoc_method=method, **form)
                    self.assertEqual(status, 200, res)
                    self.assertEqual(res["posthoc_method"], method)
                    letters, pairs = res["posthoc_letters"], res["posthoc_results"]
                    k = len(res["descriptives"])
                    self.assertEqual(len(pairs), k * (k - 1) // 2)
                    for comp in pairs:
                        share = bool(set(letters[comp["group1"]]) & set(letters[comp["group2"]]))
                        self.assertEqual(share, not comp["significant"], comp)


class InputValidationTests(unittest.TestCase):
    def assertRejected(self, status, res, fragment):
        self.assertEqual(status, 400, res)
        self.assertIn(fragment, res["detail"])

    def test_rbd_missing_plot_is_rejected(self):
        df = rbd_oneway().drop(index=7)
        self.assertRejected(*run_anova(df, test_type="rbd_oneway", dep_var="Y", ind_var1="T", rep_var="Rep"), "missing")

    def test_split_plot_missing_plot_is_rejected(self):
        df = split_plot().drop(index=5)
        self.assertRejected(*run_anova(df, test_type="splitplot", dep_var="Y", ind_var1="A", ind_var2="B", rep_var="Rep"), "missing")

    def test_non_latin_layout_is_rejected(self):
        df = latin_square()
        df["T"] = df["Row"].str.replace("R", "T")
        self.assertRejected(*run_anova(df, test_type="lsd", dep_var="Y", ind_var1="T", ind_var2="Col", rep_var="Row"), "not a Latin square")

    def test_two_way_empty_cell_is_rejected(self):
        df = factorial_crd()
        df = df[~((df["A"] == "A1") & (df["B"] == "B2"))]
        self.assertRejected(*run_anova(df, test_type="twoway", dep_var="Y", ind_var1="A", ind_var2="B"), "Empty")

    def test_games_howell_only_for_one_way(self):
        self.assertRejected(*run_anova(factorial_crd(), test_type="twoway", dep_var="Y", ind_var1="A", ind_var2="B",
                                       posthoc_method="games_howell"), "only available for One-Way")

    def test_unknown_method_is_rejected(self):
        self.assertRejected(*run_anova(one_way(), test_type="oneway", dep_var="Y", ind_var1="G",
                                       posthoc_method="scheffe"), "Unknown post-hoc method")


if __name__ == "__main__":
    unittest.main()
