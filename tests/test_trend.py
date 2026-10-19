"""Trend analysis: Mann-Kendall family, Sen's slope, change-point and homogeneity tests, ITA, and the /trend endpoint.

The statistics are checked against brute-force definitions, scipy and statsmodels.
Run from backend/:  python -m unittest discover -s tests -t .
"""
import base64
import math
import unittest

from tests.support import post_csv  # must come before other app imports
from tests.test_endpoints_smoke import PNG_SIGNATURE  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.stats as stats  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from statsmodels.tsa.stattools import acf  # noqa: E402

from app import trend as tr  # noqa: E402


def brute_s(x):
    return sum(np.sign(x[j] - x[i]) for i in range(len(x)) for j in range(i + 1, len(x)))


def ar1(n, phi, seed):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


class MannKendallTests(unittest.TestCase):
    def test_score_and_tie_corrected_variance(self):
        x = np.array([3, 5, 5, 2, 8, 8, 8, 1, 9, 4, 4, 7], dtype=float)
        self.assertEqual(tr.mk_score(x), brute_s(x))
        n = x.size
        ties = [2, 3, 2]  # 5 twice, 8 three times, 4 twice
        expected = (n * (n - 1) * (2 * n + 1) - sum(t * (t - 1) * (2 * t + 1) for t in ties)) / 18
        self.assertAlmostEqual(tr.mk_variance(x), expected)

    def test_tau_and_p_value(self):
        rng = np.random.default_rng(1)
        x = np.arange(30) * 0.05 + rng.normal(size=30)
        result = tr.mann_kendall(x)
        self.assertAlmostEqual(result["tau"], stats.kendalltau(np.arange(30), x).statistic, places=12)
        s, var = result["s"], result["var_s"]
        z = (s - np.sign(s)) / math.sqrt(var)
        self.assertAlmostEqual(result["z"], z, places=12)
        self.assertAlmostEqual(result["p_value"], 2 * stats.norm.sf(abs(z)), places=12)

    def test_trend_labels(self):
        rising = np.arange(20) + np.random.default_rng(2).normal(0, 0.5, 20)
        self.assertEqual(tr.mann_kendall(rising)["trend"], "increasing")
        self.assertEqual(tr.mann_kendall(-rising)["trend"], "decreasing")
        self.assertEqual(tr.mann_kendall(np.random.default_rng(3).normal(size=25))["trend"], "no significant trend")

    def test_modified_mann_kendall_variance_correction(self):
        x = ar1(60, 0.7, seed=4) + 0.02 * np.arange(60)
        result = tr.modified_mann_kendall(x)
        n = x.size
        index = np.arange(1, n + 1)
        ranks = stats.rankdata(x - stats.theilslopes(x, index).slope * index)
        r = acf(ranks, nlags=n - 1, fft=False)
        bound = stats.norm.ppf(0.975) / math.sqrt(n)
        total = sum((n - k) * (n - k - 1) * (n - k - 2) * r[k] for k in range(1, n - 1) if abs(r[k]) > bound)
        expected = 1 + 2 / (n * (n - 1) * (n - 2)) * total
        self.assertAlmostEqual(result["variance_correction"], expected, places=10)
        self.assertGreater(result["variance_correction"], 1.0)  # positive autocorrelation inflates the variance
        self.assertGreater(result["p_value"], tr.mann_kendall(x)["p_value"])

    def test_seasonal_mann_kendall_sums_over_seasons(self):
        rng = np.random.default_rng(5)
        years = np.repeat(np.arange(2000, 2012), 4)
        seasons = np.tile(["Q1", "Q2", "Q3", "Q4"], 12)
        cycle = np.tile([0.0, 5.0, 10.0, 3.0], 12)
        x = cycle + 0.3 * (years - 2000) + rng.normal(0, 1, years.size)
        result = tr.seasonal_mann_kendall(years, x, seasons)
        expected_s = sum(brute_s(x[seasons == s]) for s in ["Q1", "Q2", "Q3", "Q4"])
        expected_var = sum(tr.mk_variance(x[seasons == s]) for s in ["Q1", "Q2", "Q3", "Q4"])
        self.assertEqual(result["s"], expected_s)
        self.assertAlmostEqual(result["var_s"], expected_var)
        slopes = []
        for s in ["Q1", "Q2", "Q3", "Q4"]:
            xs, ts = x[seasons == s], years[seasons == s]
            slopes += [(xs[j] - xs[i]) / (ts[j] - ts[i]) for i in range(xs.size) for j in range(i + 1, xs.size)]
        self.assertAlmostEqual(result["slope"], float(np.median(slopes)), places=12)
        self.assertEqual(result["trend"], "increasing")


class MagnitudeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(6)
        self.t = np.arange(1981, 2021, dtype=float)
        self.x = 500 + 3 * (self.t - 1981) + rng.normal(0, 25, self.t.size)

    def test_sen_slope_is_median_of_pairwise_slopes(self):
        result = tr.sens_slope(self.t, self.x)
        pairs = [(self.x[j] - self.x[i]) / (self.t[j] - self.t[i]) for i in range(self.t.size) for j in range(i + 1, self.t.size)]
        self.assertAlmostEqual(result["slope"], float(np.median(pairs)), places=10)
        reference = stats.theilslopes(self.x, self.t, alpha=0.95)
        self.assertAlmostEqual(result["ci_low"], reference.low_slope)
        self.assertAlmostEqual(result["ci_high"], reference.high_slope)
        self.assertLess(result["ci_low"], result["slope"])
        self.assertGreater(result["ci_high"], result["slope"])

    def test_linear_trend_matches_statsmodels(self):
        result = tr.linear_trend(self.t, self.x)
        model = sm.OLS(self.x, sm.add_constant(self.t)).fit()
        low, high = model.conf_int(alpha=0.05)[1]
        self.assertAlmostEqual(result["slope"], model.params[1], places=8)
        self.assertAlmostEqual(result["ci_low"], low, places=8)
        self.assertAlmostEqual(result["ci_high"], high, places=8)
        self.assertAlmostEqual(result["r2"], model.rsquared, places=10)

    def test_lag1_autocorrelation(self):
        result = tr.lag1_autocorrelation(ar1(80, 0.8, seed=7))
        self.assertTrue(result["significant"])
        self.assertAlmostEqual(result["bound"], stats.norm.ppf(0.975) / math.sqrt(80))


class ChangePointTests(unittest.TestCase):
    def test_pettitt_matches_brute_force(self):
        x = np.random.default_rng(8).normal(size=25)
        result = tr.pettitt(x)
        n = x.size
        u = [sum(np.sign(x[i] - x[j]) for i in range(t + 1) for j in range(t + 1, n)) for t in range(n - 1)]
        np.testing.assert_allclose(result["u"], u)
        k = max(abs(v) for v in u)
        self.assertEqual(result["k"], k)
        self.assertAlmostEqual(result["p_value"], min(1.0, 2 * math.exp(-6 * k ** 2 / (n ** 3 + n ** 2))))

    def test_pettitt_finds_a_step_change(self):
        rng = np.random.default_rng(9)
        x = np.concatenate([rng.normal(10, 0.3, 20), rng.normal(13, 0.3, 20)])
        result = tr.pettitt(x)
        self.assertEqual(result["change_index"], 19)
        self.assertLess(result["p_value"], 0.01)
        self.assertAlmostEqual(result["mean_before"], x[:20].mean())

    def test_homogeneity_statistics_match_their_definitions(self):
        x = np.random.default_rng(10).normal(size=30)
        n = x.size
        z = (x - x.mean()) / x.std(ddof=1)
        t_k = [(k * z[:k].mean() ** 2 + (n - k) * z[k:].mean() ** 2) for k in range(1, n)]
        np.testing.assert_allclose(tr.snht_series(x)[0], t_k, rtol=1e-10)
        dev = x - x.mean()
        np.testing.assert_allclose(tr.buishand_series(x)[0], np.cumsum(dev) / np.sqrt(np.mean(dev ** 2)), rtol=1e-10)
        self.assertAlmostEqual(tr.von_neumann_ratio(x)[0], np.sum(np.diff(x) ** 2) / np.sum(dev ** 2))

    def test_homogeneity_p_values(self):
        rng = np.random.default_rng(11)
        step = np.concatenate([rng.normal(0, 1, 25), rng.normal(2.5, 1, 25)])
        result = tr.homogeneity_tests(step, simulations=2000)
        self.assertLess(result["snht"]["p_value"], 0.01)
        self.assertLess(result["buishand"]["p_value_r"], 0.01)
        self.assertLess(result["von_neumann"]["p_value"], 0.01)
        self.assertIn(result["snht"]["change_index"], (23, 24, 25))
        again = tr.homogeneity_tests(step, simulations=2000)
        self.assertEqual(result["snht"]["p_value"], again["snht"]["p_value"])  # seeded, reproducible
        calm = tr.homogeneity_tests(rng.normal(size=50), simulations=2000)
        for p in (calm["snht"]["p_value"], calm["buishand"]["p_value_q"], calm["von_neumann"]["p_value"]):
            self.assertGreater(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_wijngaard_classification(self):
        self.assertEqual(tr.homogeneity_class([0.01, 0.2, 0.3, 0.4])["class"], "useful")
        self.assertEqual(tr.homogeneity_class([0.01, 0.02, 0.3, 0.4])["class"], "doubtful")
        self.assertEqual(tr.homogeneity_class([0.01, 0.02, 0.03, 0.4])["class"], "suspect")


class InnovativeTrendAnalysisTests(unittest.TestCase):
    def test_halves_slope_and_indicator(self):
        result = tr.innovative_trend_analysis(np.array([5, 1, 4, 2, 3, 10, 6, 9, 7, 8], dtype=float))
        self.assertEqual(result["first_half"], [1, 2, 3, 4, 5])
        self.assertEqual(result["second_half"], [6, 7, 8, 9, 10])
        self.assertAlmostEqual(result["slope"], 2 * (8 - 3) / 10)
        self.assertAlmostEqual(result["indicator"], 10 * 5 / 3)
        self.assertAlmostEqual(result["ranges"][0]["change_percent"], 100 * (6.5 - 1.5) / 1.5)

    def test_odd_length_drops_the_middle_value(self):
        result = tr.innovative_trend_analysis(np.arange(1, 12, dtype=float))
        self.assertEqual(result["first_half"], [1, 2, 3, 4, 5])
        self.assertEqual(result["second_half"], [7, 8, 9, 10, 11])


class TrendEndpointTests(unittest.TestCase):
    def annual_frame(self):
        rng = np.random.default_rng(12)
        years = np.arange(1981, 2021)
        return pd.DataFrame({
            "Year": years,
            "Rainfall": 800 + 6 * (years - 1981) + rng.normal(0, 40, years.size),
            "Temperature": np.where(years < 2001, 25.0, 26.2) + rng.normal(0, 0.3, years.size),
        })

    def test_annual_series(self):
        status, res = post_csv("/analyze/trend", self.annual_frame(), value_cols_str="Rainfall,Temperature", time_col="Year")
        self.assertEqual(status, 200, res)
        self.assertEqual(res["time"]["kind"], "numeric")
        rain, temp = res["results"]["Rainfall"], res["results"]["Temperature"]
        self.assertEqual(rain["mann_kendall"]["trend"], "increasing")
        self.assertAlmostEqual(rain["sen"]["slope"], 6, delta=2)
        self.assertLess(temp["pettitt"]["p_value"], 0.01)
        self.assertAlmostEqual(temp["time_values"][temp["pettitt"]["change_index"] + 1], 2001, delta=2)
        for column in ("Rainfall", "Temperature"):
            for chart in ("series", "ita", "homogeneity"):
                self.assertTrue(base64.b64decode(res["charts"][column][chart]).startswith(PNG_SIGNATURE))
        self.assertTrue(base64.b64decode(res["summary_chart"]).startswith(PNG_SIGNATURE))

    def test_monthly_dates_with_seasons(self):
        rng = np.random.default_rng(13)
        dates = pd.date_range("2000-01-01", periods=144, freq="MS")
        cycle = 10 * np.sin(2 * np.pi * (dates.month - 1) / 12)
        df = pd.DataFrame({
            "Date": dates.strftime("%Y-%m-%d"),
            "Month": dates.month,
            "Flow": 50 + cycle + 0.05 * np.arange(144) + rng.normal(0, 1.5, 144),
        })
        status, res = post_csv("/analyze/trend", df, value_cols_str="Flow", time_col="Date", season_col="Month")
        self.assertEqual(status, 200, res)
        self.assertEqual(res["time"]["kind"], "date")
        seasonal = res["results"]["Flow"]["seasonal_mann_kendall"]
        self.assertEqual(seasonal["seasons"], 12)
        self.assertLess(seasonal["p_value"], 0.001)
        self.assertAlmostEqual(seasonal["slope"], 0.6, delta=0.15)  # 0.05 per month = 0.6 per year
        self.assertIsNone(res["summary_chart"])

    def test_year_and_month_columns_and_repeated_times(self):
        rng = np.random.default_rng(14)
        df = pd.DataFrame({"Year": np.repeat(np.arange(2001, 2011), 12), "Month": np.tile(np.arange(1, 13), 10)})
        df["Value"] = rng.normal(size=len(df)) + 0.1 * (df["Year"] - 2001)
        status, res = post_csv("/analyze/trend", df, value_cols_str="Value", time_col="Year", season_col="Month")
        self.assertEqual(status, 200, res)
        self.assertEqual(res["results"]["Value"]["n"], 120)  # year + month places every row separately
        repeated = pd.DataFrame({"Year": [2001, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009],
                                 "Value": [1, 3, 2, 4, 3, 5, 4, 6, 5, 7]})
        status, res = post_csv("/analyze/trend", repeated, value_cols_str="Value", time_col="Year")
        self.assertEqual(status, 200, res)
        self.assertEqual(res["results"]["Value"]["n"], 9)
        self.assertTrue(res["notes"])

    def test_row_order_without_time_column(self):
        status, res = post_csv("/analyze/trend", self.annual_frame(), value_cols_str="Rainfall")
        self.assertEqual(status, 200, res)
        self.assertEqual(res["time"]["kind"], "order")
        self.assertEqual(res["results"]["Rainfall"]["time_values"][:3], [1.0, 2.0, 3.0])

    def test_validation(self):
        df = self.annual_frame()
        short = df.head(5)
        text = df.assign(Rainfall=df["Rainfall"].astype(str).where(df.index != 3, "dry"))
        cases = [
            (df, dict(value_cols_str="Rainfall", alpha="0.2")),
            (short, dict(value_cols_str="Rainfall", time_col="Year")),
            (text, dict(value_cols_str="Rainfall", time_col="Year")),
            (df, dict(value_cols_str="Year", time_col="Year")),
            (df, dict(value_cols_str="Nope")),
            (df.assign(Year=["x"] * len(df)), dict(value_cols_str="Rainfall", time_col="Year")),
        ]
        for frame, form in cases:
            with self.subTest(form=form):
                status, res = post_csv("/analyze/trend", frame, **form)
                self.assertEqual(status, 400, res)


if __name__ == "__main__":
    unittest.main()
