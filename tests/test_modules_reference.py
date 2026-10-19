"""Endpoint tests for the parametric, non-parametric, correlation, regression, PCA,
descriptive, clustering and transformation modules.

Run from backend/:  python -m unittest discover -s tests -t .
"""
import unittest

from tests.support import post_csv  # must come before other app imports

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy import stats  # noqa: E402


def rng(seed):
    return np.random.default_rng(seed)


class ParametricEndpointTests(unittest.TestCase):
    def test_independent_t_reports_both_tests_and_effect_sizes(self):
        r = rng(1)
        df = pd.DataFrame({"A": r.normal(10, 1, 20), "B": np.r_[r.normal(12, 1.2, 16), [np.nan] * 4]})
        status, res = post_csv("/analyze/parametric", df, test_type="independent_t", col1="A", col2="B")
        self.assertEqual(status, 200, res)
        s = res["statistics"]
        chosen = s["student"] if res["test_used"] == "Student's t-Test" else s["welch"]
        self.assertEqual(s["p_value"], chosen["p_value"])
        self.assertEqual(s["ci_difference"], chosen["ci_difference"])
        ref = stats.ttest_ind(df["A"], df["B"].dropna(), equal_var=False)
        self.assertAlmostEqual(s["welch"]["p_value"], ref.pvalue, places=12)
        self.assertLess(s["cohens_d"], -1.0)

    def test_paired_t_reports_cohens_dz(self):
        r = rng(2)
        pre = r.normal(20, 3, 15)
        df = pd.DataFrame({"pre": pre, "post": pre + r.normal(1.5, 1, 15)})
        status, res = post_csv("/analyze/parametric", df, test_type="paired_t", col1="pre", col2="post")
        self.assertEqual(status, 200, res)
        diff = df["pre"] - df["post"]
        self.assertAlmostEqual(res["statistics"]["cohens_d"], diff.mean() / diff.std(ddof=1), places=12)


class NonParametricEndpointTests(unittest.TestCase):
    def test_mann_whitney_rank_biserial(self):
        r = rng(3)
        df = pd.DataFrame({"A": r.integers(0, 10, 14), "B": r.integers(3, 13, 14)})
        status, res = post_csv("/analyze/nonparametric", df, test_type="mann_whitney", cols_str="A,B")
        self.assertEqual(status, 200, res)
        greater = sum(a > b for a in df.A for b in df.B)
        less = sum(a < b for a in df.A for b in df.B)
        self.assertAlmostEqual(res["statistics"]["effect_size"], (greater - less) / 196, places=12)
        self.assertEqual(res["statistics"]["effect_size_name"], "Rank-biserial correlation")

    def test_wilcoxon_excludes_zero_differences(self):
        df = pd.DataFrame({"pre": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                           "post": [5, 6, 7, 9, 11, 12, 11, 15, 16, 18, 15, 20]})
        status, res = post_csv("/analyze/nonparametric", df, test_type="wilcoxon", cols_str="pre,post")
        self.assertEqual(status, 200, res)
        self.assertEqual(res["statistics"]["n_nonzero"], 7)
        self.assertAlmostEqual(res["statistics"]["effect_size"], -1.0, places=12)  # every non-zero pair increased

    def test_chi_square_uses_pearson_statistic_for_cramers_v(self):
        r = rng(4)
        df = pd.DataFrame({"A": r.choice(["yes", "no"], 60), "B": r.choice(["hi", "lo"], 60, p=[0.65, 0.35])})
        status, res = post_csv("/analyze/nonparametric", df, test_type="chi_square", cols_str="A,B")
        self.assertEqual(status, 200, res)
        table = pd.crosstab(df.A, df.B)
        chi2 = stats.chi2_contingency(table, correction=False)[0]
        self.assertAlmostEqual(res["statistics"]["statistic"], chi2, places=10)
        self.assertAlmostEqual(res["statistics"]["effect_size"], np.sqrt(chi2 / 60), places=10)
        self.assertIsNotNone(res["statistics"]["fisher_exact_p"])
        self.assertEqual(len(res["statistics"]["contingency_table"]["expected"]), 2)


class CorrelationEndpointTests(unittest.TestCase):
    def test_spearman_with_p_values(self):
        r = rng(5)
        x = r.normal(size=30)
        df = pd.DataFrame({"x": x, "y": np.exp(x) + r.normal(scale=0.3, size=30), "z": r.normal(size=30)})
        status, res = post_csv("/analyze/correlation", df, method="spearman")
        self.assertEqual(status, 200, res)
        ref = stats.spearmanr(df.x, df.y)
        self.assertEqual(res["method"], "spearman")
        self.assertAlmostEqual(res["matrix"]["values"][0][1], ref.statistic, places=12)
        self.assertAlmostEqual(res["p_values"][0][1], ref.pvalue, places=12)
        self.assertEqual(res["n"][0][1], 30)

    def test_unknown_method_is_rejected(self):
        status, _ = post_csv("/analyze/correlation", pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 1, 4, 3]}), method="kendall")
        self.assertEqual(status, 400)


class RegressionEndpointTests(unittest.TestCase):
    def setUp(self):
        r = rng(6)
        n = 40
        self.df = pd.DataFrame({"X1": r.normal(10, 2, n), "X2": r.normal(5, 1, n), "X3": r.normal(0, 3, n)})
        self.df["Y"] = 3 + 0.8 * self.df.X1 - 1.2 * self.df.X2 + 0.1 * self.df.X3 + r.normal(0, 1, n)
        self.df["Y2"] = 1 + 0.3 * self.df.X1 + 0.5 * self.df.X3 + r.normal(0, 1, n)

    def test_ols_matches_statsmodels_including_confidence_intervals(self):
        status, res = post_csv("/analyze/regression", self.df, regression_type="multiple", dep_vars_str="Y", ind_vars_str="X1,X2,X3")
        self.assertEqual(status, 200, res)
        ols = sm.OLS(self.df.Y, sm.add_constant(self.df[["X1", "X2", "X3"]])).fit()
        ci = ols.conf_int().values
        for i, coef in enumerate(res["coefficients"]):
            self.assertAlmostEqual(coef["coefficient"], ols.params.iloc[i], places=10)
            self.assertAlmostEqual(coef["p_value"], ols.pvalues.iloc[i], places=12)
            self.assertAlmostEqual(coef["ci_lower"], ci[i, 0], places=10)
            self.assertAlmostEqual(coef["ci_upper"], ci[i, 1], places=10)
        self.assertAlmostEqual(res["f_pvalue"], ols.f_pvalue, places=14)

    def test_plsr_variance_vip_and_intercepts(self):
        status, res = post_csv("/analyze/regression", self.df, regression_type="plsr", dep_vars_str="Y,Y2",
                               ind_vars_str="X1,X2,X3", n_components="3")
        self.assertEqual(status, 200, res)
        self.assertAlmostEqual(sum(res["explained_variance_x"]), 1.0, places=8)
        self.assertAlmostEqual(float(np.mean([v["vip"] ** 2 for v in res["vip"]])), 1.0, places=10)
        x = self.df[["X1", "X2", "X3"]].values
        for j, dep in enumerate(["Y", "Y2"]):
            slopes = np.array([row["coefficients"][dep] for row in res["coefficients"]])
            predicted = x @ slopes + res["intercepts"][dep]
            np.testing.assert_allclose(predicted, np.array(res["predicted"])[:, j], atol=1e-8)


class PcaEndpointTests(unittest.TestCase):
    def test_eigenvalues_are_those_of_the_correlation_matrix(self):
        r = rng(7)
        df = pd.DataFrame(r.normal(size=(25, 4)) @ [[1, 0.5, 0, 0], [0, 1, 0.3, 0], [0, 0, 1, 0.8], [0, 0, 0, 1]], columns=list("ABCD"))
        status, res = post_csv("/analyze/pca", df, columns_str="A,B,C,D")
        self.assertEqual(status, 200, res)
        expected = np.sort(np.linalg.eigvalsh(np.corrcoef(df.values, rowvar=False)))[::-1]
        np.testing.assert_allclose(res["eigenvalues"], expected, atol=1e-10)
        self.assertEqual(res["kaiser_retained"], int(np.sum(expected >= 1)))
        self.assertEqual(len(res["correlation_loadings"]["A"]), 4)


class DescriptiveEndpointTests(unittest.TestCase):
    def test_undefined_values_are_null_not_zero(self):
        df = pd.DataFrame({"v": [-2.5, -1.0, 0.0, 1.0, 2.5, -0.4, 0.4, -1.7, 1.7]})
        status, res = post_csv("/analyze/descriptive", df, columns_str="v")
        self.assertEqual(status, 200, res)
        overall = res["variables"]["v"]["overall"]
        self.assertIsNone(overall["cv"])  # mean is zero
        self.assertIsNone(overall["mode"])  # no repeated value
        self.assertAlmostEqual(overall["se_skewness"], np.sqrt(6 * 9 * 8 / (7 * 10 * 12)), places=12)
        self.assertIsNotNone(overall["lilliefors"])


class ClusterEndpointTests(unittest.TestCase):
    def setUp(self):
        r = rng(8)
        self.df = pd.DataFrame(np.vstack([r.normal(0, 1, (12, 2)), r.normal(6, 1, (12, 2))]), columns=["P", "K"])

    def test_quality_measures(self):
        status, res = post_csv("/analyze/cluster", self.df, variables_str="P,K", method="kmeans", k="2")
        self.assertEqual(status, 200, res)
        quality = res["quality"]
        self.assertGreater(quality["cophenetic_correlation"], 0.5)
        self.assertGreater(quality["kmeans_silhouette"], 0.5)
        self.assertIsNotNone(quality["kmeans_inertia"])

    def test_too_many_clusters_is_rejected(self):
        status, _ = post_csv("/analyze/cluster", self.df, variables_str="P,K", method="kmeans", k="24")
        self.assertEqual(status, 400)


class TransformEndpointTests(unittest.TestCase):
    def test_zscore_uses_sample_sd_and_boxcox_reports_lambda(self):
        r = rng(9)
        df = pd.DataFrame({"yield": r.gamma(3.0, 2.0, 30)})
        status, res = post_csv("/analyze/transform", df, columns="yield", method="zscore")
        self.assertEqual(status, 200, res)
        z = np.array([row["yield_ZScore"] for row in res["records"]])
        self.assertAlmostEqual(z.std(ddof=1), 1.0, places=12)

        status, res = post_csv("/analyze/transform", df, columns="yield", method="boxcox")
        self.assertEqual(status, 200, res)
        # lambda is found by numerical optimization; agreement to 1e-6 is the meaningful tolerance
        self.assertAlmostEqual(res["lambdas"]["yield"], stats.boxcox(df["yield"].values)[1], places=6)
        self.assertLess(abs(res["diagnostics"]["yield"]["after"]["skewness"]), abs(res["diagnostics"]["yield"]["before"]["skewness"]))


if __name__ == "__main__":
    unittest.main()
