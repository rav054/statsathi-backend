"""Time series: diagnostics, decomposition, baselines, ETS and SARIMA, accuracy measures and the /timeseries endpoint.

Checked against statsmodels (ACF/PACF, ADF/KPSS, Ljung-Box, ETS, SARIMAX) and the textbook prediction-interval formulas
in Hyndman & Athanasopoulos, Forecasting: Principles and Practice.
Run from backend/:  python -m unittest discover -s tests -t .
"""
import base64
import math
import unittest
import warnings

from tests.support import post_csv  # must come before other app imports
from tests.test_endpoints_smoke import PNG_SIGNATURE  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.stats as stats  # noqa: E402
from statsmodels.stats.diagnostic import acorr_ljungbox  # noqa: E402
from statsmodels.tsa.exponential_smoothing.ets import ETSModel  # noqa: E402
from statsmodels.tsa.seasonal import STL  # noqa: E402
from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: E402
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf  # noqa: E402

from app import timeseries as ts  # noqa: E402

warnings.simplefilter("ignore")


def monthly_series(n=60, seed=4, slope=0.4, amplitude=10.0):
    rng = np.random.default_rng(seed)
    season = amplitude * np.sin(2 * np.pi * np.arange(n) / 12)
    return 100 + slope * np.arange(n) + season + rng.normal(0, 2.5, n)


def monthly_frame(n=60, seed=4):
    dates = pd.date_range("2016-01-01", periods=n, freq="MS")
    return pd.DataFrame({"Month": dates.strftime("%Y-%m-%d"), "Rainfall": monthly_series(n, seed).round(3)})


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.y = monthly_series(48, seed=7)

    def test_naive_matches_the_textbook_formulas(self):
        result = ts.baseline_forecast(self.y, 6, 12, "naive")
        np.testing.assert_allclose(result["mean"], self.y[-1])
        sigma = np.std(np.diff(self.y), ddof=1)
        self.assertAlmostEqual(result["sigma"], sigma, places=10)
        np.testing.assert_allclose(result["spread"], sigma * np.sqrt(np.arange(1, 7)), rtol=1e-12)

    def test_seasonal_naive_repeats_the_last_season(self):
        result = ts.baseline_forecast(self.y, 14, 12, "snaive")
        np.testing.assert_allclose(result["mean"][:12], self.y[-12:])
        np.testing.assert_allclose(result["mean"][12:], self.y[-12:-10])  # the season repeats
        sigma = np.std(self.y[12:] - self.y[:-12], ddof=1)
        self.assertAlmostEqual(result["sigma"], sigma, places=10)
        steps = np.arange(1, 15)
        np.testing.assert_allclose(result["spread"], sigma * np.sqrt((steps - 1) // 12 + 1), rtol=1e-12)

    def test_drift_extends_the_line_through_the_first_and_last_points(self):
        result = ts.baseline_forecast(self.y, 5, 12, "drift")
        slope = (self.y[-1] - self.y[0]) / (self.y.size - 1)
        np.testing.assert_allclose(result["mean"], self.y[-1] + slope * np.arange(1, 6), rtol=1e-12)
        steps = np.arange(1, 6)
        sigma = result["sigma"]
        np.testing.assert_allclose(result["spread"], sigma * np.sqrt(steps * (1 + steps / (self.y.size - 1))), rtol=1e-12)

    def test_intervals_use_the_normal_quantiles(self):
        mean = np.array([10.0, 12.0])
        spread = np.array([1.0, 2.0])
        intervals = ts.intervals_from_spread(mean, spread)
        np.testing.assert_allclose(intervals["lower80"], mean - stats.norm.ppf(0.9) * spread)
        np.testing.assert_allclose(intervals["upper95"], mean + stats.norm.ppf(0.975) * spread)


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.y = monthly_series(60, seed=2)

    def test_stationarity_tests_match_statsmodels(self):
        result = ts.stationarity_tests(self.y)
        self.assertAlmostEqual(result["adf"]["p_value"], adfuller(self.y, autolag="AIC")[1], places=10)
        self.assertAlmostEqual(result["kpss"]["p_value"], kpss(self.y, regression="c", nlags="auto")[1], places=10)
        self.assertFalse(result["kpss"]["stationary"])  # a trending series is not stationary

    def test_correlogram_matches_statsmodels(self):
        result = ts.correlogram(self.y, season_length=12)
        nlags = len(result["lags"])
        np.testing.assert_allclose(result["acf"], acf(self.y, nlags=nlags, fft=False)[1:], rtol=1e-10)
        np.testing.assert_allclose(result["pacf"], pacf(self.y, nlags=nlags)[1:], rtol=1e-10)
        self.assertAlmostEqual(result["bound"], stats.norm.ppf(0.975) / math.sqrt(self.y.size), places=12)
        expected = acorr_ljungbox(self.y, lags=[result["ljung_box"]["lag"]], return_df=True).iloc[0]
        self.assertAlmostEqual(result["ljung_box"]["p_value"], float(expected["lb_pvalue"]), places=12)

    def test_decomposition_adds_up_and_measures_strength(self):
        parts = ts.decompose(self.y, 12)
        self.assertEqual(parts["method"], "stl")
        np.testing.assert_allclose(parts["trend"] + parts["seasonal"] + parts["remainder"], self.y, atol=1e-8)
        reference = STL(self.y, period=12, robust=True).fit()
        np.testing.assert_allclose(parts["trend"], reference.trend, rtol=1e-10)
        self.assertGreater(parts["strengths"]["seasonal"], 0.8)
        self.assertGreater(parts["strengths"]["trend"], 0.8)
        flat = np.random.default_rng(5).normal(size=60)
        self.assertLess(ts.decompose(flat, 12)["strengths"]["seasonal"], 0.5)

    def test_strength_formula(self):
        rng = np.random.default_rng(6)
        trend, seasonal, remainder = rng.normal(size=40), rng.normal(size=40), rng.normal(size=40)
        result = ts.strengths(trend, seasonal, remainder)
        self.assertAlmostEqual(result["trend"], max(0.0, 1 - np.var(remainder, ddof=1) / np.var(trend + remainder, ddof=1)))
        self.assertAlmostEqual(result["seasonal"],
                               max(0.0, 1 - np.var(remainder, ddof=1) / np.var(seasonal + remainder, ddof=1)))

    def test_suggested_differences(self):
        regular, seasonal = ts.differences_needed(self.y, 12)
        self.assertEqual(seasonal, 1)  # a strong seasonal pattern gets a seasonal difference
        self.assertGreaterEqual(regular, 0)
        noise = np.random.default_rng(8).normal(size=60)
        self.assertEqual(ts.differences_needed(noise, 1), (0, 0))


class AccuracyTests(unittest.TestCase):
    def test_metrics_match_their_definitions(self):
        train = np.array([10.0, 12, 11, 13, 12, 14, 13, 15])
        actual = np.array([16.0, 15, 17])
        predicted = np.array([15.0, 15, 15])
        result = ts.accuracy(actual, predicted, train, season_length=1)
        errors = actual - predicted
        self.assertAlmostEqual(result["rmse"], math.sqrt(np.mean(errors ** 2)))
        self.assertAlmostEqual(result["mae"], np.mean(np.abs(errors)))
        self.assertAlmostEqual(result["mape"], np.mean(np.abs(errors / actual)) * 100)
        scale = np.mean(np.abs(np.diff(train)))
        self.assertAlmostEqual(result["mase"], np.mean(np.abs(errors)) / scale)
        self.assertAlmostEqual(ts.scaling_error(train, 4), np.mean(np.abs(train[4:] - train[:-4])))

    def test_mape_is_skipped_when_a_value_is_zero(self):
        result = ts.accuracy(np.array([0.0, 5.0]), np.array([1.0, 4.0]), np.arange(10.0), 1)
        self.assertIsNone(result["mape"])


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.y = monthly_series(60, seed=3)

    def test_ets_matches_statsmodels(self):
        result = ts.fit_ets(self.y, "hw_add", 12, 6)
        reference = ETSModel(pd.Series(self.y), error="add", trend="add", seasonal="add", seasonal_periods=12,
                             damped_trend=False).fit(disp=False)
        np.testing.assert_allclose(result["mean"], reference.forecast(6), rtol=1e-8)
        self.assertAlmostEqual(result["aicc"], reference.aicc, places=8)
        frame = reference.get_prediction(start=60, end=65).summary_frame(alpha=0.05)
        np.testing.assert_allclose(result["lower95"], frame["pi_lower"], rtol=1e-8)
        self.assertTrue(np.all(result["lower80"] > result["lower95"]))

    def test_sarima_search_returns_the_best_corrected_aic(self):
        best = ts.search_sarima(self.y, 6, 12)
        self.assertIsNotNone(best)
        self.assertEqual(len(best["mean"]), 6)
        for order, seasonal_order in [((1, 1, 0), (0, 1, 0, 12)), ((0, 1, 1), (0, 1, 1, 12))]:
            with self.subTest(order=order):
                alternative = SARIMAX(self.y, order=order, seasonal_order=seasonal_order, enforce_stationarity=False,
                                      enforce_invertibility=False).fit(disp=False)
                self.assertLessEqual(best["aicc"], alternative.aicc + 1e-6)
        self.assertIn(best["seasonal_order"][3], (0, 12))

    def test_available_models_depend_on_the_series(self):
        seasonal = ts.available_models(60, 12, positive=True)
        self.assertEqual(seasonal[:3], ["naive", "drift", "snaive"])
        self.assertIn("hw_mul", seasonal)
        self.assertNotIn("hw_mul", ts.available_models(60, 12, positive=False))
        plain = ts.available_models(30, 1, positive=True)
        self.assertNotIn("snaive", plain)
        self.assertNotIn("hw_add", plain)
        self.assertIn("sarima", plain)

    def test_fit_model_shapes(self):
        for key in ("naive", "snaive", "drift", "ses", "holt", "damped", "hw_add"):
            with self.subTest(model=key):
                fitted = ts.fit_model(key, self.y, 4, 12)
                self.assertEqual(len(fitted["mean"]), 4)
                self.assertEqual(len(fitted["fitted"]), self.y.size)
                self.assertTrue(np.all(np.asarray(fitted["lower95"]) <= np.asarray(fitted["mean"]) + 1e-9))


class TimeSeriesEndpointTests(unittest.TestCase):
    def test_monthly_series_with_forecast(self):
        status, res = post_csv("/analyze/timeseries", monthly_frame(), value_col="Rainfall", time_col="Month",
                               horizon="6", holdout="12")
        self.assertEqual(status, 200, res)
        self.assertEqual(res["series"]["season_length"], 12)
        self.assertEqual(res["series"]["frequency"], "monthly")
        self.assertEqual(len(res["forecast"]["mean"]), 6)
        self.assertEqual(len(res["forecast"]["times"]), 6)
        self.assertEqual(res["forecast"]["times"][0][:7], "2021-01")  # continues the monthly grid
        for i in range(6):
            self.assertLessEqual(res["forecast"]["lower95"][i], res["forecast"]["lower80"][i])
            self.assertLessEqual(res["forecast"]["lower80"][i], res["forecast"]["mean"][i])
            self.assertLessEqual(res["forecast"]["mean"][i], res["forecast"]["upper80"][i])
            self.assertLessEqual(res["forecast"]["upper80"][i], res["forecast"]["upper95"][i])
        selected = [m for m in res["models"] if m["selected"]]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["key"], res["best"])
        errors = {m["key"]: m["holdout"]["rmse"] for m in res["models"]}
        self.assertLessEqual(errors[res["best"]], min(errors.values()) + 1e-9)
        self.assertLess(errors[res["best"]], errors["naive"])  # the seasonal pattern is used
        self.assertEqual(len(res["decomposition"]["trend"]), res["series"]["n"])
        self.assertGreater(res["decomposition"]["strengths"]["seasonal"], 0.5)
        for key, image in res["charts"].items():
            with self.subTest(chart=key):
                self.assertTrue(base64.b64decode(image).startswith(PNG_SIGNATURE))

    def test_yearly_numeric_time_without_season(self):
        rng = np.random.default_rng(9)
        years = np.arange(1990, 2024)
        df = pd.DataFrame({"Year": years, "Production": 50 + 1.2 * (years - 1990) + rng.normal(0, 4, years.size)})
        status, res = post_csv("/analyze/timeseries", df, value_col="Production", time_col="Year", horizon="4")
        self.assertEqual(status, 200, res)
        self.assertEqual(res["series"]["kind"], "numeric")
        self.assertEqual(res["series"]["season_length"], 1)
        self.assertEqual(res["forecast"]["times"], [2024.0, 2025.0, 2026.0, 2027.0])
        self.assertIsNone(res["charts"]["seasonal"])
        self.assertIn(res["best"], [m["key"] for m in res["models"]])

    def test_log_transform_and_row_order(self):
        frame = monthly_frame(48, seed=11).drop(columns=["Month"])
        status, res = post_csv("/analyze/timeseries", frame, value_col="Rainfall", season_length="12", horizon="3",
                               transform="log")
        self.assertEqual(status, 200, res)
        self.assertEqual(res["series"]["kind"], "order")
        self.assertEqual(res["series"]["transform"], "log")
        self.assertTrue(all(v > 0 for v in res["forecast"]["mean"]))
        self.assertTrue(any("log-transformed" in note for note in res["notes"]))

    def test_validation(self):
        df = monthly_frame(36)
        gaps = df.drop(index=range(5, 17))  # a year-long hole in the middle
        cases = [
            (df.head(6), dict(value_col="Rainfall", time_col="Month")),
            (gaps, dict(value_col="Rainfall", time_col="Month")),
            (df, dict(value_col="Rainfall", time_col="Month", horizon="300")),
            (df, dict(value_col="Rainfall", time_col="Month", transform="sqrt")),
            (df, dict(value_col="Rainfall", time_col="Nope")),
            (df, dict(value_col="Rainfall", time_col="Rainfall")),
            (df.assign(Rainfall=df["Rainfall"].astype(str).where(df.index != 3, "lost")),
             dict(value_col="Rainfall", time_col="Month")),
            (df.assign(Rainfall=df["Rainfall"] - df["Rainfall"].max()),
             dict(value_col="Rainfall", time_col="Month", transform="log")),
        ]
        for frame, form in cases:
            with self.subTest(form=form, rows=len(frame)):
                status, res = post_csv("/analyze/timeseries", frame, **form)
                self.assertEqual(status, 400, res)


if __name__ == "__main__":
    unittest.main()
