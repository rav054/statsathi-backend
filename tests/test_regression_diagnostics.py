"""Regression diagnostics and fit bands checked against statsmodels.

Run from backend/:  python -m unittest discover -s tests -t .
"""
import unittest

from tests.support import post_csv  # must come before other app imports

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.stats as stats  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from statsmodels.stats.outliers_influence import OLSInfluence  # noqa: E402


def regression_frame():
    rng = np.random.default_rng(12)
    n = 30
    height = rng.normal(90, 8, n)
    tillers = rng.normal(12, 2, n)
    return pd.DataFrame({
        "Height": height.round(2),
        "Tillers": tillers.round(2),
        "Yield": (5 + 0.35 * height + 0.8 * tillers + rng.normal(0, 3, n)).round(2),
    })


class RegressionDiagnosticsTests(unittest.TestCase):
    def fit(self, predictors):
        df = regression_frame()
        status, res = post_csv("/analyze/regression", df, regression_type="simple" if len(predictors) == 1 else "multiple",
                               dep_vars_str="Yield", ind_vars_str=",".join(predictors))
        self.assertEqual(status, 200, res)
        model = sm.OLS(df["Yield"], sm.add_constant(df[predictors])).fit()
        return df, res, model

    def test_diagnostics_match_statsmodels(self):
        _, res, model = self.fit(["Height", "Tillers"])
        influence = OLSInfluence(model)
        diagnostics = res["diagnostics"]
        np.testing.assert_allclose(diagnostics["leverage"], influence.hat_matrix_diag, rtol=1e-9)
        np.testing.assert_allclose(diagnostics["standardized_residuals"], influence.resid_studentized_internal, rtol=1e-9)
        np.testing.assert_allclose(diagnostics["cooks_distance"], influence.cooks_distance[0], rtol=1e-9)
        self.assertEqual(diagnostics["rows"][:3], [2, 3, 4])
        self.assertAlmostEqual(diagnostics["cooks_threshold"], 4 / 30)
        self.assertIsNone(res["fit_band"])  # bands only for simple regression

    def test_qq_points_are_sorted_standardized_residuals(self):
        _, res, model = self.fit(["Height", "Tillers"])
        diagnostics = res["diagnostics"]
        expected = np.sort(OLSInfluence(model).resid_studentized_internal)
        np.testing.assert_allclose(diagnostics["qq_sample"], expected, rtol=1e-9)
        n = len(expected)
        np.testing.assert_allclose(diagnostics["qq_theoretical"],
                                   stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25)), rtol=1e-12)

    def test_simple_regression_bands_match_statsmodels(self):
        df, res, model = self.fit(["Height"])
        band = res["fit_band"]
        grid = np.asarray(band["x"])
        frame = model.get_prediction(sm.add_constant(pd.DataFrame({"Height": grid}), has_constant="add")) \
            .summary_frame(alpha=0.05)
        np.testing.assert_allclose(band["fitted"], frame["mean"], rtol=1e-9)
        np.testing.assert_allclose(band["ci_lower"], frame["mean_ci_lower"], rtol=1e-9)
        np.testing.assert_allclose(band["ci_upper"], frame["mean_ci_upper"], rtol=1e-9)
        np.testing.assert_allclose(band["pi_lower"], frame["obs_ci_lower"], rtol=1e-9)
        np.testing.assert_allclose(band["pi_upper"], frame["obs_ci_upper"], rtol=1e-9)
        self.assertEqual(band["x_values"], df["Height"].tolist())


if __name__ == "__main__":
    unittest.main()
