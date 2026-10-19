"""Reference tests for app/measures.py against scipy, statsmodels, sklearn and published values.

Run from backend/:  python -m unittest discover -s tests -t .
"""
import math
import unittest
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster import hierarchy
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

from app.measures import (
    chi_square_independence,
    cluster_quality,
    correlation_tests,
    distribution_extras,
    pca_decomposition,
    pls_summary,
    rank_biserial_independent,
    rank_biserial_paired,
    shape_summary,
    skewness_kurtosis_se,
    two_sample_t,
)


class TwoSampleTTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(10)
        self.x = rng.normal(10, 2, 18)
        self.y = rng.normal(11.5, 3.5, 25)

    def test_student_and_welch_match_scipy_and_statsmodels(self):
        result = two_sample_t(self.x, self.y)
        compare = CompareMeans(DescrStatsW(self.x), DescrStatsW(self.y))
        for key, equal_var, usevar in (("student", True, "pooled"), ("welch", False, "unequal")):
            ref = stats.ttest_ind(self.x, self.y, equal_var=equal_var)
            self.assertAlmostEqual(result[key]["t_statistic"], ref.statistic, places=10)
            self.assertAlmostEqual(result[key]["p_value"], ref.pvalue, places=12)
            self.assertAlmostEqual(result[key]["degrees_of_freedom"], ref.df, places=8)
            low, high = compare.tconfint_diff(usevar=usevar)
            self.assertAlmostEqual(result[key]["ci_difference"][0], low, places=10)
            self.assertAlmostEqual(result[key]["ci_difference"][1], high, places=10)

    def test_cohens_d_and_hedges_g(self):
        n1, n2 = len(self.x), len(self.y)
        pooled_sd = math.sqrt(((n1 - 1) * self.x.var(ddof=1) + (n2 - 1) * self.y.var(ddof=1)) / (n1 + n2 - 2))
        d = (self.x.mean() - self.y.mean()) / pooled_sd
        result = two_sample_t(self.x, self.y)
        self.assertAlmostEqual(result["cohens_d"], d, places=12)
        self.assertAlmostEqual(result["hedges_g"], d * (1 - 3 / (4 * (n1 + n2) - 9)), places=12)


class RankBiserialTests(unittest.TestCase):
    def test_independent_equals_probability_of_superiority_difference(self):
        rng = np.random.default_rng(11)
        x = rng.integers(0, 10, 15).astype(float)
        y = rng.integers(2, 12, 12).astype(float)  # includes ties
        u = stats.mannwhitneyu(x, y, alternative="two-sided").statistic
        greater = sum(a > b for a in x for b in y)
        less = sum(a < b for a in x for b in y)
        self.assertAlmostEqual(rank_biserial_independent(u, len(x), len(y)), (greater - less) / (len(x) * len(y)), places=12)

    def test_paired_excludes_zero_differences(self):
        d = np.array([0, 0, 1.5, -0.5, 2, 3, -1, 0, 4, 2.5])
        result = rank_biserial_paired(d)
        nonzero = d[d != 0]
        ranks = stats.rankdata(np.abs(nonzero))
        m = len(nonzero)
        self.assertEqual(result["n_nonzero"], 7)
        self.assertAlmostEqual(result["rank_biserial"], (ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / (m * (m + 1) / 2), places=12)
        self.assertAlmostEqual(min(result["w_plus"], result["w_minus"]), stats.wilcoxon(d).statistic, places=12)


class ChiSquareTests(unittest.TestCase):
    def test_matches_scipy_and_cramers_v(self):
        table = np.array([[12, 5, 9], [7, 14, 6]])
        result = chi_square_independence(table)
        chi2, p_value, dof, expected = stats.chi2_contingency(table, correction=False)
        self.assertAlmostEqual(result["statistic"], chi2, places=12)
        self.assertAlmostEqual(result["p_value"], p_value, places=12)
        self.assertEqual(result["degrees_of_freedom"], dof)
        np.testing.assert_allclose(result["expected"], expected)
        self.assertAlmostEqual(result["cramers_v"], stats.contingency.association(table, method="cramer"), places=12)
        self.assertNotIn("yates", result)

    def test_two_by_two_adds_yates_and_fisher(self):
        table = np.array([[8, 2], [1, 5]])
        result = chi_square_independence(table)
        chi2_yates, p_yates, _, _ = stats.chi2_contingency(table, correction=True)
        self.assertAlmostEqual(result["yates"]["statistic"], chi2_yates, places=12)
        self.assertAlmostEqual(result["yates"]["p_value"], p_yates, places=12)
        self.assertAlmostEqual(result["fisher_exact_p"], stats.fisher_exact(table)[1], places=12)


class CorrelationTests(unittest.TestCase):
    def test_pairwise_complete_pearson_and_spearman(self):
        rng = np.random.default_rng(12)
        base = rng.normal(size=(40, 1))
        df = pd.DataFrame(base + rng.normal(scale=[0.5, 1.0, 2.0, 3.0], size=(40, 4)), columns=list("ABCD"))
        df.iloc[[1, 5, 9], 0] = np.nan
        df.iloc[[2, 5], 2] = np.nan
        for method, test in (("pearson", stats.pearsonr), ("spearman", stats.spearmanr)):
            r, p, n = correlation_tests(df, method)
            for i, j in combinations(range(4), 2):
                pair = df.iloc[:, [i, j]].dropna()
                ref = test(pair.iloc[:, 0], pair.iloc[:, 1])
                self.assertAlmostEqual(r[i, j], ref.statistic, places=12)
                self.assertAlmostEqual(p[i, j], ref.pvalue, places=12)
                self.assertEqual(n[i, j], len(pair))
            if method == "pearson":
                np.testing.assert_allclose(r, df.corr().values, atol=1e-12)


class PcaTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(13)
        latent = rng.normal(size=(40, 1))
        self.x = latent @ [[1.0, 0.8, -0.5, 0.2]] + rng.normal(size=(40, 4)) * [0.5, 1.0, 1.5, 3.0] + [10, 20, 5, 0]

    def test_correlation_matrix_pca(self):
        result = pca_decomposition(self.x, scale=True)
        expected = np.sort(np.linalg.eigvalsh(np.corrcoef(self.x, rowvar=False)))[::-1]
        np.testing.assert_allclose(result["eigenvalues"], expected, atol=1e-10)
        self.assertAlmostEqual(float(np.sum(result["eigenvalues"])), 4.0, places=10)
        np.testing.assert_allclose(result["scores"].var(axis=0, ddof=1), result["eigenvalues"], atol=1e-10)
        between = [[np.corrcoef(self.x[:, j], result["scores"][:, a])[0, 1] for a in range(4)] for j in range(4)]
        np.testing.assert_allclose(result["correlation_loadings"], between, atol=1e-10)

    def test_covariance_matrix_pca(self):
        result = pca_decomposition(self.x, scale=False)
        expected = np.sort(np.linalg.eigvalsh(np.cov(self.x, rowvar=False)))[::-1]
        np.testing.assert_allclose(result["eigenvalues"], expected, atol=1e-9)
        between = [[np.corrcoef(self.x[:, j], result["scores"][:, a])[0, 1] for a in range(4)] for j in range(4)]
        np.testing.assert_allclose(result["correlation_loadings"], between, atol=1e-10)

    def test_matches_sklearn_up_to_sign(self):
        z = (self.x - self.x.mean(axis=0)) / self.x.std(axis=0, ddof=1)
        ref = PCA().fit(z)
        result = pca_decomposition(self.x, scale=True)
        np.testing.assert_allclose(result["eigenvalues"], ref.explained_variance_, atol=1e-10)
        np.testing.assert_allclose(np.abs(result["eigenvectors"].T), np.abs(ref.components_), atol=1e-8)


class PlsTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(14)
        self.x = rng.normal(size=(50, 4)) * [1, 3, 10, 0.5] + [5, 0, 100, 2]
        self.y = np.column_stack([self.x @ [0.5, 0.1, 0.02, 1.0], self.x @ [0.0, -0.2, 0.01, 2.0]]) + rng.normal(size=(50, 2))

    def test_x_variance_sums_to_one_with_all_components(self):
        pls = PLSRegression(n_components=4).fit(self.x, self.y)
        result = pls_summary(pls, self.x, self.y)
        self.assertAlmostEqual(sum(result["explained_variance_x"]), 1.0, places=8)
        self.assertAlmostEqual(float(np.mean(np.square(result["vip"]))), 1.0, places=10)

    def test_y_variance_equals_standardized_r2_of_predictions(self):
        pls = PLSRegression(n_components=2).fit(self.x, self.y)
        result = pls_summary(pls, self.x, self.y)
        mean, sd = self.y.mean(axis=0), self.y.std(axis=0, ddof=1)
        ys, ps = (self.y - mean) / sd, (pls.predict(self.x) - mean) / sd
        self.assertAlmostEqual(sum(result["explained_variance_y"]), 1 - np.sum((ys - ps) ** 2) / np.sum(ys ** 2), places=8)


class DistributionTests(unittest.TestCase):
    def test_standard_errors_match_spss(self):
        for n, se_skew, se_kurt in ((10, 0.687, 1.334), (30, 0.427, 0.833)):
            s, k = skewness_kurtosis_se(n)
            self.assertAlmostEqual(s, se_skew, places=3)
            self.assertAlmostEqual(k, se_kurt, places=3)

    def test_normality_tests_percentiles_and_outliers(self):
        rng = np.random.default_rng(15)
        x = np.append(rng.normal(50, 5, 60), 200.0)
        extras = distribution_extras(x)
        stat, p_value = lilliefors(x, dist="norm")
        self.assertAlmostEqual(extras["lilliefors"]["p_value"], p_value, places=12)
        self.assertAlmostEqual(extras["dagostino_pearson"]["p_value"], stats.normaltest(x).pvalue, places=12)
        self.assertAlmostEqual(extras["percentiles"]["p90"], np.percentile(x, 90), places=12)
        self.assertEqual([o["position"] for o in extras["z_outliers"]], [60])

    def test_shape_summary_uses_bias_corrected_moments(self):
        x = np.random.default_rng(16).gamma(2.0, 1.0, 45)
        summary = shape_summary(x)
        self.assertAlmostEqual(summary["skewness"], stats.skew(x, bias=False), places=12)
        self.assertAlmostEqual(summary["kurtosis"], stats.kurtosis(x, bias=False), places=12)
        self.assertAlmostEqual(summary["shapiro_p"], stats.shapiro(x).pvalue, places=12)


class ClusterQualityTests(unittest.TestCase):
    def test_cophenetic_correlation_matches_scipy(self):
        rng = np.random.default_rng(17)
        x = np.vstack([rng.normal(0, 1, (15, 3)), rng.normal(5, 1, (15, 3))])
        linkage = hierarchy.linkage(x, method="ward")
        labels = list(hierarchy.fcluster(linkage, t=2, criterion="maxclust") - 1)
        quality = cluster_quality(x, linkage, labels, labels, 12.5)
        from scipy.spatial.distance import pdist
        self.assertAlmostEqual(quality["cophenetic_correlation"], hierarchy.cophenet(linkage, pdist(x))[0], places=12)
        self.assertGreater(quality["kmeans_silhouette"], 0.5)
        self.assertIsNone(cluster_quality(x, linkage, [0] * 30, labels, 1.0)["kmeans_silhouette"])


if __name__ == "__main__":
    unittest.main()
