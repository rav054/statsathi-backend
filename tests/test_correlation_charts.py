"""Correlation charts: heatmaps, scatter matrix (pairs plot) and mixed correlogram, plus column selection.

Run from backend/:  python -m unittest discover -s tests -t .
"""
import base64
import unittest

from tests.support import post_csv  # must come before other app imports
from tests.test_endpoints_smoke import PNG_SIGNATURE, sample_frame  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.stats as stats  # noqa: E402


class CorrelationChartTests(unittest.TestCase):
    def test_every_chart_type_renders_for_both_methods(self):
        df = sample_frame()
        df.loc[2, "Height"] = np.nan  # pairwise n now differs between pairs
        for plot_type in ("full", "lower", "upper", "pairs", "mixed"):
            for method in ("pearson", "spearman"):
                with self.subTest(plot_type=plot_type, method=method):
                    status, res = post_csv("/analyze/correlation", df, plot_type=plot_type, method=method, palette="RdBu_r")
                    self.assertEqual(status, 200, res)
                    self.assertEqual(res["plot_type"], plot_type)
                    self.assertTrue(base64.b64decode(res["plot"]).startswith(PNG_SIGNATURE))

    def test_selected_columns_define_the_matrix(self):
        df = sample_frame()
        status, res = post_csv("/analyze/correlation", df, columns_str="Protein,Yield", plot_type="pairs")
        self.assertEqual(status, 200, res)
        self.assertEqual(res["matrix"]["columns"], ["Protein", "Yield"])
        expected = stats.pearsonr(df["Protein"], df["Yield"])
        self.assertAlmostEqual(res["matrix"]["values"][0][1], expected.statistic, places=10)
        self.assertAlmostEqual(res["p_values"][0][1], expected.pvalue, places=10)

    def test_scatter_matrix_is_limited_to_twelve_variables(self):
        rng = np.random.default_rng(5)
        wide = pd.DataFrame(rng.normal(size=(30, 13)), columns=[f"V{i}" for i in range(13)])
        status, res = post_csv("/analyze/correlation", wide, plot_type="pairs")
        self.assertEqual(status, 400)
        self.assertIn("12", res["detail"])
        status, res = post_csv("/analyze/correlation", wide, plot_type="mixed")
        self.assertEqual(status, 200, res)

    def test_invalid_requests(self):
        df = sample_frame()
        cases = [
            dict(columns_str="Yield,NoSuchColumn"),
            dict(columns_str="Yield,Treatment"),  # not numeric
            dict(columns_str="Yield"),  # fewer than two
            dict(plot_type="violin"),  # not a correlation chart type
        ]
        for form in cases:
            with self.subTest(form=form):
                status, res = post_csv("/analyze/correlation", df, **form)
                self.assertEqual(status, 400, res)


class CorrelationOrderTargetNetworkTests(unittest.TestCase):
    def interleaved_frame(self):
        rng = np.random.default_rng(9)
        n = 60
        base_a, base_b = rng.normal(size=n), rng.normal(size=n)
        return pd.DataFrame({
            "A1": base_a + rng.normal(0, 0.2, n),
            "B1": base_b + rng.normal(0, 0.2, n),
            "A2": base_a + rng.normal(0, 0.2, n),
            "B2": base_b + rng.normal(0, 0.2, n),
            "A3": base_a + rng.normal(0, 0.3, n),
        })

    def test_clustered_order_puts_related_variables_together(self):
        df = self.interleaved_frame()
        status, res = post_csv("/analyze/correlation", df, order="clustered", plot_type="full")
        self.assertEqual(status, 200, res)
        order = res["matrix"]["columns"]
        self.assertEqual(sorted(order), sorted(df.columns))
        a_positions = sorted(order.index(c) for c in ("A1", "A2", "A3"))
        self.assertEqual(a_positions[-1] - a_positions[0], 2)  # the three A variables are contiguous
        i, j = order.index("B1"), order.index("A2")
        self.assertAlmostEqual(res["matrix"]["values"][i][j], stats.pearsonr(df["B1"], df["A2"]).statistic, places=10)

    def test_target_and_network_charts(self):
        df = self.interleaved_frame()
        cases = [
            dict(plot_type="target", target="A1"),
            dict(plot_type="target", target="B2", method="spearman", order="clustered"),
            dict(plot_type="network", threshold="0.5"),
            dict(plot_type="network", threshold="0.99"),  # nothing passes: still a valid chart
        ]
        for form in cases:
            with self.subTest(form=form):
                status, res = post_csv("/analyze/correlation", df, **form)
                self.assertEqual(status, 200, res)
                self.assertTrue(base64.b64decode(res["plot"]).startswith(PNG_SIGNATURE))

    def test_target_network_and_order_validation(self):
        df = self.interleaved_frame()
        for form in (dict(plot_type="target"), dict(plot_type="target", target="Nope"),
                     dict(plot_type="network", threshold="1.5"), dict(order="random")):
            with self.subTest(form=form):
                status, res = post_csv("/analyze/correlation", df, **form)
                self.assertEqual(status, 400, res)

    def test_fisher_interval(self):
        from app.modules.correlation import fisher_interval
        low, high = fisher_interval(0.5, 30)
        half = stats.norm.ppf(0.975) / np.sqrt(27)
        self.assertAlmostEqual(low, np.tanh(np.arctanh(0.5) - half), places=12)
        self.assertAlmostEqual(high, np.tanh(np.arctanh(0.5) + half), places=12)
        self.assertTrue(np.isnan(fisher_interval(0.5, 3)[0]))


if __name__ == "__main__":
    unittest.main()
