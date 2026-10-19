"""Modern chart types in the Plots module, and the statistics they draw.

Run from backend/:  python -m unittest discover -s tests -t .
"""
import base64
import io
import unittest

from tests.support import post_csv, post_file  # must come before other app imports
from tests.test_endpoints_smoke import PNG_SIGNATURE  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.stats as stats  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from PIL import Image  # noqa: E402

from app.modules import chart_gallery as cg  # noqa: E402


def field_frame():
    rng = np.random.default_rng(11)
    n = 48
    base = rng.normal(40, 4, n)
    before = rng.normal(20, 3, n)
    return pd.DataFrame({
        "Treatment": np.tile(["T1", "T2", "T3", "T10"], 12),
        "Location": np.repeat(["North", "South"], 24),
        "Rep": np.tile([1, 2, 3], 16),
        "Yield": base.round(2),
        "Height": (90 + 2 * (base - 40) + rng.normal(0, 5, n)).round(1),
        "Protein": rng.normal(12, 1.5, n).round(2),
        "Before": before.round(2),
        "After": (before + rng.normal(1.5, 2, n)).round(2),
    })


CASES = [
    ("raincloud", dict(x_var="Treatment", y_var="Yield")),
    ("boxswarm", dict(x_var="Treatment", y_var="Yield")),
    ("boxswarm", dict(x_var="Treatment", y_var="Yield", hue_var="Location")),
    ("ridgeline", dict(x_var="Yield", hue_var="Treatment")),
    ("ecdf", dict(x_var="Yield", hue_var="Location")),
    ("ecdf", dict(x_var="Yield")),
    ("meanci", dict(x_var="Treatment", y_var="Yield")),
    ("meanci", dict(x_var="Treatment", y_var="Yield", hue_var="Location")),
    ("lollipop", dict(x_var="Treatment", y_var="Yield")),
    ("dumbbell", dict(x_var="Treatment", y_var="Yield", hue_var="Location")),
    ("paired", dict(x_var="Before", y_var="After")),
    ("paired", dict(x_var="Before", y_var="After", hue_var="Location")),
    ("meanheatmap", dict(x_var="Treatment", y_var="Yield", hue_var="Rep")),
    ("joint", dict(x_var="Height", y_var="Yield", fit_reg="true")),
    ("joint", dict(x_var="Height", y_var="Yield", hue_var="Location", fit_reg="true")),
    ("density2d", dict(x_var="Height", y_var="Yield")),
    ("hexbin", dict(x_var="Height", y_var="Yield", fit_reg="true")),
    ("mosaic", dict(x_var="Treatment", hue_var="Location")),
    ("radar", dict(x_var="Yield", y_vars_str="Yield,Height,Protein", hue_var="Treatment")),
]


class ChartGalleryRenderTests(unittest.TestCase):
    def test_every_modern_chart_renders(self):
        df = field_frame()
        df.loc[5, "Yield"] = np.nan
        for plot_type, form in CASES:
            with self.subTest(plot_type=plot_type, form=form):
                status, res = post_csv("/analyze/plot", df, plot_type=plot_type, palette="okabe_ito", **form)
                self.assertEqual(status, 200, res)
                self.assertTrue(base64.b64decode(res["plot"]).startswith(PNG_SIGNATURE))

    def test_legacy_palettes_and_custom_labels(self):
        status, res = post_csv("/analyze/plot", field_frame(), plot_type="meanci", x_var="Treatment", y_var="Yield",
                               palette="sunset", title="Yield by treatment", xlabel="Treatment", ylabel="Yield (q/ha)",
                               legend_loc="none", show_grid="false")
        self.assertEqual(status, 200, res)

    def test_input_errors_are_explained(self):
        df = field_frame()
        cases = [
            dict(plot_type="dumbbell", x_var="Location", y_var="Yield", hue_var="Treatment"),  # 4 levels, needs 2
            dict(plot_type="ridgeline", x_var="Yield"),  # no grouping variable
            dict(plot_type="radar", x_var="Yield", y_vars_str="Yield,Height", hue_var="Treatment"),  # 2 variables
            dict(plot_type="mosaic", x_var="Treatment"),  # no second variable
            dict(plot_type="raincloud", x_var="Location", y_var="Treatment"),  # non-numeric value
            dict(plot_type="paired", x_var="Before", y_var="Before"),  # same column twice
            dict(plot_type="meanheatmap", x_var="Treatment", y_var="Yield", hue_var="Treatment"),
        ]
        for form in cases:
            with self.subTest(form=form):
                status, res = post_csv("/analyze/plot", df, **form)
                self.assertEqual(status, 400, res)
                self.assertTrue(res["detail"])

    def test_journal_export_keeps_the_preset_width(self):
        content = field_frame().to_csv(index=False).encode()
        response = post_file("/analyze/plot/download", "field.csv", content, "text/csv", plot_type="joint",
                             x_var="Height", y_var="Yield", download_format="tiff", dpi="300", preset="single_column")
        self.assertEqual(response.status_code, 200, response.text[:200])
        width, _ = Image.open(io.BytesIO(response.content)).size
        self.assertAlmostEqual(width, round(85 / 25.4 * 300), delta=1)


class ChartStatisticsTests(unittest.TestCase):
    def test_mean_confidence_interval_matches_t_interval(self):
        values = np.array([38.2, 41.5, 39.9, 44.1, 40.3, 42.8, 37.6])
        mean, low, high = cg.mean_confidence_interval(values)
        expected = stats.t.interval(0.95, len(values) - 1, loc=values.mean(), scale=stats.sem(values))
        self.assertAlmostEqual(mean, values.mean(), places=12)
        self.assertAlmostEqual(low, expected[0], places=10)
        self.assertAlmostEqual(high, expected[1], places=10)
        self.assertTrue(np.isnan(cg.mean_confidence_interval([5.0])[1]))

    def test_regression_band_matches_statsmodels(self):
        rng = np.random.default_rng(3)
        x = rng.normal(90, 8, 40)
        y = 10 + 0.4 * x + rng.normal(0, 3, 40)
        grid = np.linspace(x.min(), x.max(), 25)
        fitted, low, high = cg.regression_band(x, y, grid)
        model = sm.OLS(y, sm.add_constant(x)).fit()
        prediction = model.get_prediction(sm.add_constant(grid)).summary_frame(alpha=0.05)
        np.testing.assert_allclose(fitted, prediction["mean"], rtol=1e-10)
        np.testing.assert_allclose(low, prediction["mean_ci_lower"], rtol=1e-10)
        np.testing.assert_allclose(high, prediction["mean_ci_upper"], rtol=1e-10)

    def test_standardized_residuals(self):
        observed = np.array([[20, 15, 5], [10, 25, 25]])
        expected = stats.contingency.expected_freq(observed)
        np.testing.assert_allclose(cg.standardized_residuals(observed), (observed - expected) / np.sqrt(expected))

    def test_natural_level_order(self):
        self.assertEqual(sorted(["T10", "T2", "T1"], key=cg.natural_key), ["T1", "T2", "T10"])


if __name__ == "__main__":
    unittest.main()
