"""Analyses pooled over multiple imputations: two-group comparisons, correlations and ANOVA (D1 Wald tests).

With m identical "imputations" every pooled result must reduce to the complete-data analysis, which is checked against
scipy and statsmodels; the D1 statistic and its degrees of freedom against Li, Raghunathan & Rubin (1991) worked by
hand; and the whole pipeline against data simulated with a known treatment effect.
Run from backend/:  python -m unittest discover -s tests -t .
"""
import json
import math
import unittest

from tests.support import post_csv  # must come before other app imports

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.stats as stats  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from statsmodels.stats.anova import anova_lm  # noqa: E402

from app import imputation as imp  # noqa: E402
from app import pooled_analyses as pa  # noqa: E402


def factorial_trial(seed=3, missing=0.0, reps=4):
    """Irrigation (3) × Nitrogen (2) in `reps` blocks, with a real nitrogen and irrigation effect."""
    rng = np.random.default_rng(seed)
    rows = []
    for irrigation, bonus in (("I1", 0.0), ("I2", 1.5), ("I3", 0.8)):
        for nitrogen, n_bonus in (("N0", 0.0), ("N60", 1.0)):
            for rep in range(1, reps + 1):
                rows.append({"Irrigation": irrigation, "Nitrogen": nitrogen, "Rep": f"R{rep}",
                             "Tillers": 30 + 4 * bonus + rng.normal(0, 2),
                             "Yield": 10 + bonus + n_bonus + 0.1 * rep + rng.normal(0, 0.6)})
    df = pd.DataFrame(rows)
    if missing:
        df.loc[rng.random(len(df)) < missing, "Yield"] = np.nan
    return df


def copies(df, m=5):
    return [df.copy() for _ in range(m)]


class GroupComparisonTests(unittest.TestCase):
    def test_identical_imputations_give_welch_on_the_complete_data(self):
        df = factorial_trial()
        pooled = pa.pooled_group_comparison(copies(df), "Yield", "Nitrogen")
        a = df.loc[df["Nitrogen"] == "N0", "Yield"]
        b = df.loc[df["Nitrogen"] == "N60", "Yield"]
        self.assertAlmostEqual(pooled["estimate"], a.mean() - b.mean())
        self.assertAlmostEqual(pooled["se"], math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
        self.assertEqual(pooled["levels"], ["N0", "N60"])
        self.assertAlmostEqual(pooled["between"], 0.0)
        welch = stats.ttest_ind(a, b, equal_var=False)
        self.assertAlmostEqual(abs(pooled["statistic"]), abs(welch.statistic))

    def test_needs_two_groups(self):
        with self.assertRaisesRegex(imp.ImputationError, "exactly two groups"):
            pa.pooled_group_comparison(copies(factorial_trial()), "Yield", "Irrigation")


class CorrelationTests(unittest.TestCase):
    def test_identical_imputations_give_the_complete_data_coefficient(self):
        df = factorial_trial()
        pooled = pa.pooled_correlations(copies(df), ["Yield", "Tillers"])
        pair = pooled["pairs"][0]
        r = df["Yield"].corr(df["Tillers"])
        self.assertAlmostEqual(pair["r"], r)
        self.assertLess(pair["ci_low"], r)
        self.assertGreater(pair["ci_high"], r)
        # the interval is Fisher's: tanh(atanh(r) ± t * 1/sqrt(n - 3))
        half = math.atanh(pair["ci_high"]) - math.atanh(r)
        self.assertAlmostEqual(half / (1 / math.sqrt(len(df) - 3)), stats.t.ppf(0.975, pair["df"]), places=6)

    def test_spearman_uses_the_inflated_variance(self):
        df = factorial_trial()
        pearson = pa.pooled_correlations(copies(df), ["Yield", "Tillers"], "pearson")["pairs"][0]
        spearman = pa.pooled_correlations(copies(df), ["Yield", "Tillers"], "spearman")["pairs"][0]
        self.assertAlmostEqual(spearman["r"], df["Yield"].corr(df["Tillers"], method="spearman"))
        width_p = math.atanh(pearson["ci_high"]) - math.atanh(pearson["r"])
        width_s = math.atanh(spearman["ci_high"]) - math.atanh(spearman["r"])
        self.assertAlmostEqual(width_s / width_p, math.sqrt(1.06), delta=0.01)

    def test_all_pairs(self):
        df = factorial_trial()
        df["Height"] = df["Tillers"] * 0.5 + 3
        pooled = pa.pooled_correlations(copies(df), ["Yield", "Tillers", "Height"])
        self.assertEqual([(p["x"], p["y"]) for p in pooled["pairs"]],
                         [("Yield", "Tillers"), ("Yield", "Height"), ("Tillers", "Height")])


class D1Tests(unittest.TestCase):
    def test_hand_worked_statistic_and_degrees_of_freedom(self):
        q = [[1.0, 2.0], [1.4, 1.6], [0.9, 2.3]]
        u = [np.array([[0.2, 0.05], [0.05, 0.3]])] * 3
        m, k = 3, 2
        q_bar = np.mean(q, axis=0)
        b = np.cov(np.array(q), rowvar=False, ddof=1)
        u_inv = np.linalg.inv(u[0])
        r1 = (1 + 1 / m) * np.trace(b @ u_inv) / k
        d1 = q_bar @ u_inv @ q_bar / (k * (1 + r1))
        t = k * (m - 1)  # 4, so the small-t formula applies
        df2 = t * (1 + 1 / k) * (1 + 1 / r1) ** 2 / 2
        pooled = pa.d1_wald(q, u)
        self.assertAlmostEqual(pooled["F"], d1)
        self.assertAlmostEqual(pooled["riv"], r1)
        self.assertAlmostEqual(pooled["df2"], df2)
        self.assertAlmostEqual(pooled["p_value"], stats.f.sf(d1, k, df2))

        many = pa.d1_wald(q * 3, u * 3)  # m = 9, t = 16 > 4
        r1_many = (1 + 1 / 9) * np.trace(np.cov(np.array(q * 3), rowvar=False, ddof=1) @ u_inv) / k
        self.assertAlmostEqual(many["df2"], 4 + (16 - 4) * (1 + (1 - 2 / 16) / r1_many) ** 2)

    def test_no_between_variance_caps_df_at_the_complete_data_df(self):
        pooled = pa.d1_wald([[1.0, 2.0]] * 4, [np.eye(2) * 0.5] * 4, df_complete=20)
        self.assertAlmostEqual(pooled["F"], (1 + 4) / 0.5 / 2)
        self.assertEqual(pooled["df2"], 20)


class AnovaTests(unittest.TestCase):
    def test_identical_imputations_reproduce_the_type_iii_table(self):
        df = factorial_trial()
        pooled = pa.pooled_anova(copies(df, 4), "Yield", ["Irrigation", "Nitrogen"], block="Rep", interaction=True)
        model = smf.ols("Yield ~ C(Irrigation, Sum) * C(Nitrogen, Sum) + C(Rep, Sum)", data=df).fit()
        table = anova_lm(model, typ=3)
        expected = {
            "Irrigation": table.loc["C(Irrigation, Sum)"],
            "Nitrogen": table.loc["C(Nitrogen, Sum)"],
            "Rep": table.loc["C(Rep, Sum)"],
            "Irrigation × Nitrogen": table.loc["C(Irrigation, Sum):C(Nitrogen, Sum)"],
        }
        terms = {row["term"]: row for row in pooled["terms"]}
        self.assertEqual(set(terms), set(expected))
        for name, row in expected.items():
            self.assertAlmostEqual(terms[name]["F"], row["F"], places=6, msg=name)
            self.assertEqual(terms[name]["df1"], int(row["df"]))
            self.assertAlmostEqual(terms[name]["p_value"], row["PR(>F)"], places=8, msg=name)
        self.assertEqual(pooled["df_residual"], model.df_resid)

    def test_one_way_matches_the_classical_f_test(self):
        df = factorial_trial()
        pooled = pa.pooled_anova(copies(df, 3), "Yield", ["Irrigation"], interaction=False)
        groups = [g["Yield"] for _, g in df.groupby("Irrigation")]
        self.assertAlmostEqual(pooled["terms"][0]["F"], stats.f_oneway(*groups).statistic, places=6)

    def test_level_means_pairwise_differences_and_letters(self):
        df = factorial_trial()
        pooled = pa.pooled_anova(copies(df, 3), "Yield", ["Irrigation"], block="Rep", interaction=False, comparison="lsd")
        model = smf.ols("Yield ~ C(Irrigation, Sum) + C(Rep, Sum)", data=df).fit()
        means = {row["level"]: row for row in pooled["comparisons"]["Irrigation"]["means"]}
        for level, group in df.groupby("Irrigation"):
            self.assertAlmostEqual(means[level]["mean"], group["Yield"].mean())
            self.assertAlmostEqual(means[level]["se"], math.sqrt(model.mse_resid / len(group)))
        pairs = pooled["comparisons"]["Irrigation"]["pairs"]
        for pair in pairs:
            expected_diff = means[pair["group2"]]["mean"] - means[pair["group1"]]["mean"]
            self.assertAlmostEqual(pair["mean_diff"], expected_diff)
            share = set(means[pair["group1"]]["group"]) & set(means[pair["group2"]]["group"])
            self.assertEqual(bool(share), not pair["significant"], pair)  # letters agree with the tests

        bonferroni = pa.pooled_anova(copies(df, 3), "Yield", ["Irrigation"], block="Rep", interaction=False)
        for lsd_pair, bon_pair in zip(pairs, bonferroni["comparisons"]["Irrigation"]["pairs"]):
            self.assertAlmostEqual(bon_pair["p_value"], min(1.0, lsd_pair["p_value"] * 3))

    def test_imputed_trial_detects_the_effects_and_carries_the_uncertainty(self):
        # 48 plots, a quarter of the yields missing: large enough for the simulated effects to be clear
        df = factorial_trial(seed=8, missing=0.25, reps=8)
        completed = imp.multiple_impute(df, ["Yield"], ["Tillers", "Irrigation", "Nitrogen"], m=20, seed=5)
        pooled = pa.pooled_anova(completed, "Yield", ["Irrigation", "Nitrogen"], block="Rep", interaction=True)
        terms = {row["term"]: row for row in pooled["terms"]}
        self.assertLess(terms["Irrigation"]["p_value"], 0.05)
        self.assertGreater(terms["Irrigation × Nitrogen"]["p_value"], 0.05)  # no interaction was simulated
        self.assertLess(terms["Nitrogen"]["p_value"], 0.05)
        self.assertGreater(terms["Irrigation"]["riv"], 0)
        self.assertLessEqual(terms["Irrigation"]["df2"], pooled["df_residual"])
        # I2 was simulated as the best irrigation level
        best = max(pooled["comparisons"]["Irrigation"]["means"], key=lambda row: row["mean"])
        self.assertEqual(best["level"], "I2")
        self.assertIn("a", best["group"])

    def test_refusals(self):
        df = factorial_trial()
        with self.assertRaisesRegex(imp.ImputationError, "one or two treatment factors"):
            pa.pooled_anova(copies(df), "Yield", [])
        with self.assertRaises(imp.ImputationError):
            pa.run_pooled_analyses(copies(df), {"anova": {"response": "Irrigation", "factors": ["Nitrogen"]}})


class EndpointTests(unittest.TestCase):
    def test_pooled_analyses_through_the_api(self):
        df = factorial_trial(seed=4, missing=0.2)
        analyses = {
            "group_comparison": {"outcome": "Yield", "group": "Nitrogen"},
            "correlation": {"variables": ["Yield", "Tillers"], "method": "pearson"},
            "anova": {"response": "Yield", "factors": ["Irrigation", "Nitrogen"], "block": "Rep", "interaction": True},
        }
        code, body = post_csv("/analyze/prep/multiple-imputation", df, columns_str="Yield", predictors_str="Tillers,Irrigation,Nitrogen",
                              m="5", seed="1", analyses=json.dumps(analyses))
        self.assertEqual(code, 200, body)
        self.assertEqual(body["group_comparison"]["levels"], ["N0", "N60"])
        self.assertEqual(body["correlation"]["pairs"][0]["x"], "Yield")
        self.assertEqual([t["term"] for t in body["anova"]["terms"]], ["Irrigation", "Nitrogen", "Rep", "Irrigation × Nitrogen"])
        json.dumps(body)

    def test_bad_analysis_requests(self):
        df = factorial_trial(missing=0.2)
        for analyses, message in [
            ("not json", "JSON"),
            (json.dumps({"correlation": {"variables": ["Yield", "Rainfall"]}}), "not in the dataset"),
            (json.dumps({"group_comparison": {"outcome": "Yield", "group": "Irrigation"}}), "exactly two groups"),
        ]:
            code, body = post_csv("/analyze/prep/multiple-imputation", df, columns_str="Yield", m="3", analyses=analyses)
            self.assertEqual(code, 400, analyses)
            self.assertIn(message, body["detail"])


if __name__ == "__main__":
    unittest.main()
