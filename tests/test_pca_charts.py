"""PCA charts and the statistics behind them: parallel analysis, contributions and group ellipses.

Run from backend/:  python -m unittest discover -s tests -t .
"""
import base64
import unittest

from tests.support import post_csv  # must come before other app imports
from tests.test_chart_gallery import field_frame  # noqa: E402
from tests.test_endpoints_smoke import PNG_SIGNATURE  # noqa: E402

import numpy as np  # noqa: E402
import scipy.stats as stats  # noqa: E402

from app.modules import pca_charts  # noqa: E402


class PcaStatisticsTests(unittest.TestCase):
    def test_parallel_analysis_is_reproducible_and_decreasing(self):
        first, iterations = pca_charts.parallel_analysis(40, 5)
        second, _ = pca_charts.parallel_analysis(40, 5)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(iterations, 200)
        self.assertTrue(np.all(np.diff(first) < 0))
        self.assertGreater(first[0], 1.0)  # sampling error inflates the largest random eigenvalue
        self.assertAlmostEqual(first.sum(), 5, delta=1.0)

    def test_retained_components_stops_at_first_failure(self):
        self.assertEqual(pca_charts.retained_components([3.0, 1.2, 1.5, 0.2], [1.6, 1.3, 1.1, 0.9]), 1)
        self.assertEqual(pca_charts.retained_components([0.5, 0.4], [1.2, 1.0]), 0)

    def test_contributions_sum_to_100_per_component(self):
        components = np.linalg.qr(np.random.default_rng(1).normal(size=(4, 4)))[0].T
        contributions = pca_charts.variable_contributions(components)
        np.testing.assert_allclose(contributions.sum(axis=1), 100.0)

    def test_data_ellipse_holds_95_percent_of_a_bivariate_normal_sample(self):
        rng = np.random.default_rng(2)
        sample = rng.multivariate_normal([2, -1], [[4, 1.5], [1.5, 1]], size=20000)
        (cx, cy), width, height, angle = pca_charts.data_ellipse(sample[:, 0], sample[:, 1])
        theta = np.radians(angle)
        dx, dy = sample[:, 0] - cx, sample[:, 1] - cy
        u = dx * np.cos(theta) + dy * np.sin(theta)
        v = -dx * np.sin(theta) + dy * np.cos(theta)
        inside = (u / (width / 2)) ** 2 + (v / (height / 2)) ** 2 <= 1
        self.assertAlmostEqual(inside.mean(), 0.95, delta=0.01)
        self.assertAlmostEqual(stats.chi2.ppf(0.95, 2), 5.991, places=3)

    def test_mean_ellipse_is_the_data_ellipse_shrunk_by_root_n(self):
        rng = np.random.default_rng(3)
        x, y = rng.normal(0, 2, 36), rng.normal(0, 1, 36)
        _, data_width, data_height, data_angle = pca_charts.data_ellipse(x, y)
        _, mean_width, mean_height, mean_angle = pca_charts.data_ellipse(x, y, of_mean=True)
        self.assertAlmostEqual(data_width / mean_width, 6.0, places=10)
        self.assertAlmostEqual(data_height / mean_height, 6.0, places=10)
        self.assertAlmostEqual(abs(np.cos(np.radians(data_angle - mean_angle))), 1.0, places=10)


class PcaEndpointChartTests(unittest.TestCase):
    def test_pca_returns_extra_charts_and_parallel_analysis(self):
        status, res = post_csv("/analyze/pca", field_frame(), columns_str="Yield,Height,Protein,Before,After",
                               hue_var="Treatment", palette="okabe_ito")
        self.assertEqual(status, 200, res)
        for key in ("scree", "correlation_circle", "contributions"):
            with self.subTest(chart=key):
                self.assertTrue(base64.b64decode(res["charts"][key]).startswith(PNG_SIGNATURE))
        parallel = res["parallel_analysis"]
        self.assertEqual(len(parallel["thresholds"]), len(res["eigenvalues"]))
        self.assertEqual(parallel["retained"],
                         pca_charts.retained_components(res["eigenvalues"], parallel["thresholds"]))
        for pc, values in res["contributions"].items():
            self.assertAlmostEqual(sum(values.values()), 100.0, places=6, msg=pc)

    def test_covariance_pca_has_no_parallel_analysis(self):
        status, res = post_csv("/analyze/pca", field_frame(), columns_str="Yield,Height,Protein", scale="false")
        self.assertEqual(status, 200, res)
        self.assertIsNone(res["parallel_analysis"])
        self.assertTrue(base64.b64decode(res["charts"]["scree"]).startswith(PNG_SIGNATURE))


def varimax_criterion(loadings):
    squared = np.asarray(loadings) ** 2
    return float(((squared ** 2).mean(axis=0) - squared.mean(axis=0) ** 2).sum())


class PcaUpgradeTests(unittest.TestCase):
    def test_varimax_matches_statsmodels(self):
        from statsmodels.multivariate.factor_rotation import rotate_factors
        loadings = np.random.default_rng(5).normal(size=(8, 3)) * [0.8, 0.5, 0.3]
        rotated, rotation = pca_charts.varimax(loadings)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
        np.testing.assert_allclose((rotated ** 2).sum(axis=1), (loadings ** 2).sum(axis=1), rtol=1e-10)  # communalities
        h = np.sqrt((loadings ** 2).sum(axis=1))[:, None]
        reference, _ = rotate_factors(loadings / h, "varimax")
        self.assertAlmostEqual(varimax_criterion(rotated / h), varimax_criterion(reference), delta=1e-4)
        self.assertGreaterEqual(varimax_criterion(rotated / h), varimax_criterion(loadings / h))
        ordered, variance = pca_charts.ordered_rotation(rotated)
        self.assertTrue(np.all(np.diff(variance) <= 0))
        self.assertTrue(np.all(ordered.sum(axis=0) >= 0))

    def test_observation_quality(self):
        x = np.random.default_rng(6).normal(size=(30, 4))
        z = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
        eigenvalues, vectors = np.linalg.eigh(np.corrcoef(x, rowvar=False))
        order = np.argsort(eigenvalues)[::-1]
        scores = z @ vectors[:, order]
        cos2, contributions = pca_charts.observation_quality(scores, eigenvalues[order])
        np.testing.assert_allclose(cos2.sum(axis=1), 1.0)
        np.testing.assert_allclose(contributions.sum(axis=0), 100.0)

    def test_biplot_options(self):
        status, res = post_csv("/analyze/pca", field_frame(), columns_str="Yield,Height,Protein,Before,After",
                               hue_var="Treatment", x_component="1", y_component="3", biplot_type="correlation",
                               ellipse="hull", label_col="Treatment")
        self.assertEqual(status, 200, res)
        self.assertEqual(res["biplot"]["y_component"], 3)
        self.assertEqual(res["observation_quality"]["components"], [1, 3])
        self.assertEqual(len(res["observation_quality"]["cos2_plane"]), len(res["sample_scores"]))
        varimax = res["varimax"]
        self.assertGreaterEqual(varimax["components"], 2)
        for col, h2 in varimax["communalities"].items():
            original = sum(v ** 2 for v in res["correlation_loadings"][col][:varimax["components"]])
            self.assertAlmostEqual(h2, original, places=8, msg=col)
        for key in ("loadings_heatmap", "varimax_heatmap"):
            self.assertTrue(base64.b64decode(res["charts"][key]).startswith(PNG_SIGNATURE))
        status, res = post_csv("/analyze/pca", field_frame(), columns_str="Yield,Height,Protein", hue_var="Treatment",
                               ellipse="data", rotate_components="3")
        self.assertEqual(status, 200, res)
        self.assertEqual(res["varimax"]["components"], 3)

    def test_biplot_validation(self):
        base = dict(columns_str="Yield,Height,Protein")
        for form in (dict(y_component="9"), dict(x_component="2", y_component="2"), dict(biplot_type="radial"),
                     dict(ellipse="star"), dict(label_col="Nope"), dict(rotate_components="1")):
            with self.subTest(form=form):
                status, res = post_csv("/analyze/pca", field_frame(), **base, **form)
                self.assertEqual(status, 400, res)


if __name__ == "__main__":
    unittest.main()
