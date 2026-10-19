"""Time-series analysis and forecasting.

- Frequency detection, stationarity tests (ADF, KPSS) and the correlogram (ACF/PACF with Ljung-Box)
- Decomposition into trend, seasonal and remainder (STL or classical), with Hyndman's strength measures
- Forecasting models: naive, seasonal naive and drift baselines with their textbook prediction intervals
  (Hyndman & Athanasopoulos, Forecasting: Principles and Practice), exponential smoothing (ETS state-space models)
  and SARIMA with an automatic order search by corrected AIC
- Accuracy on a holdout period: RMSE, MAE, MAPE and MASE
"""

import itertools
import math
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

MIN_OBSERVATIONS = 12
MAX_OBSERVATIONS = 5000
MAX_SARIMA_FITS = 40
SEASONAL_STRENGTH_FOR_DIFFERENCING = 0.64  # Hyndman's rule of thumb for a seasonal difference

BASELINES = {"naive": "Naive (last value)", "snaive": "Seasonal naive", "drift": "Drift"}
ETS_SPECS = {
    "ses": ("Simple exponential smoothing", dict(trend=None, seasonal=None, damped_trend=False)),
    "holt": ("Holt linear trend", dict(trend="add", seasonal=None, damped_trend=False)),
    "damped": ("Damped trend", dict(trend="add", seasonal=None, damped_trend=True)),
    "hw_add": ("Holt-Winters additive", dict(trend="add", seasonal="add", damped_trend=False)),
    "hw_mul": ("Holt-Winters multiplicative", dict(trend="add", seasonal="mul", damped_trend=False)),
}
MODEL_NAMES = {**BASELINES, **{k: v[0] for k, v in ETS_SPECS.items()}, "sarima": "SARIMA (automatic order)"}


def quiet():
    """statsmodels prints convergence and frequency warnings that are not useful to the user."""
    return warnings.catch_warnings()


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def stationarity_tests(y, regression="c"):
    """Augmented Dickey-Fuller (H0: a unit root) and KPSS (H0: stationary)."""
    y = np.asarray(y, dtype=float)
    with quiet():
        warnings.simplefilter("ignore")
        adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, _ = adfuller(y, autolag="AIC")
        kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(y, regression=regression, nlags="auto")
    return {
        "adf": {"statistic": float(adf_stat), "p_value": float(adf_p), "lags": int(adf_lags),
                "critical": {k: float(v) for k, v in adf_crit.items()},
                "stationary": bool(adf_p < 0.05)},
        "kpss": {"statistic": float(kpss_stat), "p_value": float(kpss_p), "lags": int(kpss_lags),
                 "critical": {k: float(v) for k, v in kpss_crit.items()},
                 "stationary": bool(kpss_p > 0.05)},
    }


def differences_needed(y, season_length, max_d=2):
    """Number of differences suggested by KPSS, and a seasonal difference when the season is strong."""
    y = np.asarray(y, dtype=float)
    seasonal = 0
    if season_length > 1 and y.size > 2 * season_length + 4:
        parts = decompose(y, season_length)
        if parts["strengths"]["seasonal"] >= SEASONAL_STRENGTH_FOR_DIFFERENCING:
            seasonal = 1
    working = np.diff(y, n=season_length) if seasonal else y.copy()
    regular = 0
    while regular < max_d and working.size > 12:
        with quiet():
            warnings.simplefilter("ignore")
            p_value = kpss(working, regression="c", nlags="auto")[1]
        if p_value > 0.05:
            break
        working = np.diff(working)
        regular += 1
    return regular, seasonal


def correlogram(y, nlags=None, season_length=1):
    """ACF and PACF with their 95% bounds, and the Ljung-Box test of overall autocorrelation."""
    y = np.asarray(y, dtype=float)
    n = y.size
    if nlags is None:
        nlags = int(min(max(10, 2 * season_length), n // 2 - 1, 36))
    nlags = max(1, min(nlags, n // 2 - 1))
    with quiet():
        warnings.simplefilter("ignore")
        acf_values = acf(y, nlags=nlags, fft=False)
        pacf_values = pacf(y, nlags=nlags)
        lag = min(2 * season_length if season_length > 1 else 10, n // 5, nlags)
        test = acorr_ljungbox(y, lags=[max(1, lag)], return_df=True).iloc[0]
    return {
        "lags": list(range(1, nlags + 1)),
        "acf": [float(v) for v in acf_values[1:]],
        "pacf": [float(v) for v in pacf_values[1:]],
        "bound": float(stats.norm.ppf(0.975) / math.sqrt(n)),
        "ljung_box": {"lag": int(max(1, lag)), "statistic": float(test["lb_stat"]), "p_value": float(test["lb_pvalue"])},
    }


def strengths(trend, seasonal, remainder):
    """Strength of trend and seasonality (Wang, Smith & Hyndman 2006), between 0 and 1."""
    trend = np.asarray(trend, dtype=float)
    seasonal = np.asarray(seasonal, dtype=float)
    remainder = np.asarray(remainder, dtype=float)
    valid = np.isfinite(trend) & np.isfinite(seasonal) & np.isfinite(remainder)
    trend, seasonal, remainder = trend[valid], seasonal[valid], remainder[valid]
    if remainder.size < 3:
        return {"trend": None, "seasonal": None}

    def strength(component):
        variance = np.var(component + remainder, ddof=1)
        return float(max(0.0, 1 - np.var(remainder, ddof=1) / variance)) if variance > 0 else 0.0

    return {"trend": strength(trend), "seasonal": strength(seasonal) if np.any(seasonal != 0) else 0.0}


def decompose(y, season_length, method="stl", multiplicative=False):
    """Trend, seasonal and remainder components. STL when the series is seasonal, classical otherwise."""
    y = np.asarray(y, dtype=float)
    if season_length > 1 and y.size > 2 * season_length:
        if method == "stl":
            with quiet():
                warnings.simplefilter("ignore")
                fitted = STL(y, period=season_length, robust=True).fit()
            trend, seasonal, remainder = fitted.trend, fitted.seasonal, fitted.resid
        else:
            with quiet():
                warnings.simplefilter("ignore")
                fitted = seasonal_decompose(y, model="multiplicative" if multiplicative else "additive",
                                            period=season_length)
            trend, seasonal, remainder = fitted.trend, fitted.seasonal, fitted.resid
    else:  # no season: a centred moving average is the trend
        method = "moving average"
        window = max(3, min(int(round(y.size / 10)) | 1, y.size // 2 * 2 - 1))
        kernel = np.ones(window) / window
        trend = np.full(y.size, np.nan)
        half = window // 2
        trend[half:y.size - half] = np.convolve(y, kernel, mode="valid")
        seasonal = np.zeros(y.size)
        remainder = y - trend
    return {"method": method, "trend": np.asarray(trend, dtype=float), "seasonal": np.asarray(seasonal, dtype=float),
            "remainder": np.asarray(remainder, dtype=float),
            "strengths": strengths(trend, seasonal, remainder)}


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------
def scaling_error(train, season_length):
    """Mean absolute error of the seasonal naive forecast on the training data (the denominator of MASE)."""
    train = np.asarray(train, dtype=float)
    lag = season_length if season_length > 1 and train.size > season_length else 1
    differences = np.abs(train[lag:] - train[:-lag])
    return float(differences.mean()) if differences.size else None


def accuracy(actual, predicted, train=None, season_length=1):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = actual - predicted
    scale = scaling_error(train, season_length) if train is not None else None
    return {
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mae": float(np.mean(np.abs(errors))),
        "mape": float(np.mean(np.abs(errors / actual)) * 100) if np.all(actual != 0) else None,
        "mase": float(np.mean(np.abs(errors)) / scale) if scale else None,
    }


# ---------------------------------------------------------------------------
# Forecasting models
# ---------------------------------------------------------------------------
def baseline_forecast(y, horizon, season_length=1, kind="naive"):
    """Naive, seasonal naive and drift forecasts with their analytic prediction intervals."""
    y = np.asarray(y, dtype=float)
    n = y.size
    steps = np.arange(1, horizon + 1)
    if kind == "naive":
        fitted = np.concatenate([[np.nan], y[:-1]])
        mean = np.full(horizon, y[-1])
        sigma = np.nanstd(y - fitted, ddof=1)
        spread = sigma * np.sqrt(steps)
    elif kind == "snaive":
        lag = season_length
        fitted = np.concatenate([np.full(lag, np.nan), y[:-lag]])
        mean = y[n - lag + (steps - 1) % lag]
        sigma = np.nanstd(y - fitted, ddof=1)
        spread = sigma * np.sqrt((steps - 1) // lag + 1)
    elif kind == "drift":
        slope = (y[-1] - y[0]) / (n - 1)
        fitted = np.concatenate([[np.nan], y[:-1] + slope])
        mean = y[-1] + slope * steps
        sigma = np.nanstd(y - fitted, ddof=1)
        spread = sigma * np.sqrt(steps * (1 + steps / (n - 1)))
    else:
        raise ValueError(f"Unknown baseline: {kind}")
    return {"mean": mean, "spread": spread, "fitted": fitted, "sigma": float(sigma)}


def intervals_from_spread(mean, spread):
    z80, z95 = stats.norm.ppf(0.9), stats.norm.ppf(0.975)
    return {"lower80": mean - z80 * spread, "upper80": mean + z80 * spread,
            "lower95": mean - z95 * spread, "upper95": mean + z95 * spread}


def fit_ets(y, key, season_length, horizon):
    name, options = ETS_SPECS[key]
    seasonal = options["seasonal"]
    series = pd.Series(np.asarray(y, dtype=float))  # ETS prediction intervals need a pandas index
    with quiet():
        warnings.simplefilter("ignore")
        model = ETSModel(series, error="add" if seasonal != "mul" else "mul",
                         trend=options["trend"], seasonal=seasonal, damped_trend=options["damped_trend"],
                         seasonal_periods=season_length if seasonal else None)
        fitted = model.fit(disp=False)
        prediction = fitted.get_prediction(start=series.size, end=series.size + horizon - 1)
        eighty = prediction.summary_frame(alpha=0.2)
        ninety_five = prediction.summary_frame(alpha=0.05)
    return {
        "mean": np.asarray(eighty["mean"], dtype=float),
        "lower80": np.asarray(eighty["pi_lower"], dtype=float), "upper80": np.asarray(eighty["pi_upper"], dtype=float),
        "lower95": np.asarray(ninety_five["pi_lower"], dtype=float),
        "upper95": np.asarray(ninety_five["pi_upper"], dtype=float),
        "fitted": np.asarray(fitted.fittedvalues, dtype=float),
        "aic": float(fitted.aic), "aicc": float(fitted.aicc), "bic": float(fitted.bic),
        "detail": name,
    }


def sarima_orders(d, seasonal_d, season_length, limit=MAX_SARIMA_FITS):
    """Candidate (p, q) and (P, Q) combinations, smallest models first."""
    seasonal = season_length > 1
    combinations = []
    for p, q in itertools.product(range(3), repeat=2):
        for P, Q in itertools.product(range(2), repeat=2) if seasonal else [(0, 0)]:
            combinations.append(((p, d, q), (P, seasonal_d, Q, season_length if seasonal else 0)))
    combinations.sort(key=lambda spec: sum(spec[0]) + sum(spec[1][:3]))
    return combinations[:limit]


def fit_sarima(y, horizon, order, seasonal_order):
    with quiet():
        warnings.simplefilter("ignore")
        fitted = SARIMAX(np.asarray(y, dtype=float), order=order, seasonal_order=seasonal_order,
                         enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        forecast = fitted.get_forecast(horizon)
        eighty = forecast.conf_int(alpha=0.2)
        ninety_five = forecast.conf_int(alpha=0.05)
    return {
        "mean": np.asarray(forecast.predicted_mean, dtype=float),
        "lower80": eighty[:, 0], "upper80": eighty[:, 1],
        "lower95": ninety_five[:, 0], "upper95": ninety_five[:, 1],
        "fitted": np.asarray(fitted.fittedvalues, dtype=float),
        "aic": float(fitted.aic), "aicc": float(fitted.aicc), "bic": float(fitted.bic),
        "order": tuple(int(v) for v in order), "seasonal_order": tuple(int(v) for v in seasonal_order),
        "detail": f"SARIMA{tuple(order)}" + (f"{tuple(seasonal_order)}" if seasonal_order[3] else ""),
    }


def search_sarima(y, horizon, season_length):
    """The SARIMA order with the smallest corrected AIC among the candidates that fit."""
    d, seasonal_d = differences_needed(y, season_length)
    best = None
    for order, seasonal_order in sarima_orders(d, seasonal_d, season_length):
        if sum(order) + sum(seasonal_order[:3]) == 0:
            continue
        try:
            candidate = fit_sarima(y, horizon, order, seasonal_order)
        except Exception:
            continue
        if np.isfinite(candidate["aicc"]) and (best is None or candidate["aicc"] < best["aicc"]):
            best = candidate
    return best


def available_models(n, season_length, positive):
    keys = ["naive", "drift"]
    if season_length > 1 and n > 2 * season_length:
        keys.append("snaive")
    keys += ["ses", "holt", "damped"]
    if season_length > 1 and n >= 2 * season_length + 4:
        keys.append("hw_add")
        if positive:
            keys.append("hw_mul")
    keys.append("sarima")
    return keys


def fit_model(key, y, horizon, season_length, sarima_order=None):
    """One model fitted to `y`, forecasting `horizon` steps with 80% and 95% prediction intervals."""
    if key in BASELINES:
        result = baseline_forecast(y, horizon, season_length, key)
        return {**intervals_from_spread(result["mean"], result["spread"]), "mean": result["mean"],
                "fitted": result["fitted"], "aic": None, "aicc": None, "bic": None, "detail": BASELINES[key]}
    if key in ETS_SPECS:
        return fit_ets(y, key, season_length, horizon)
    if key == "sarima":
        if sarima_order is not None:
            return fit_sarima(y, horizon, *sarima_order)
        return search_sarima(y, horizon, season_length)
    raise ValueError(f"Unknown model: {key}")
