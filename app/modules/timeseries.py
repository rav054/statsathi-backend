"""Time series: diagnostics, decomposition, forecasting models and their accuracy."""

import math
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from .. import timeseries as ts
from ..auth import get_current_user
from ..models import User
from . import chart_gallery, timeseries_charts
from .common import read_uploaded_table

router = APIRouter()

MAX_HORIZON = 120
MAX_GAP_SHARE = 0.10
# Median spacing in days -> pandas frequency, season length and a label
DATE_FREQUENCIES = [
    (1.0, "D", 7, "daily"),
    (7.0, "W", 52, "weekly"),
    (14.0, "2W", 26, "fortnightly"),
    (30.44, "MS", 12, "monthly"),
    (91.3, "QS", 4, "quarterly"),
    (182.6, "2QS", 2, "half-yearly"),
    (365.25, "YS", 1, "yearly"),
]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def bad_request(message):
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)


def number(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def numbers(values):
    return [number(v) for v in np.asarray(values, dtype=float)]


def match_frequency(days):
    best = min(DATE_FREQUENCIES, key=lambda row: abs(math.log(days / row[0])))
    return best if abs(math.log(days / best[0])) < math.log(1.35) else None


def date_series(frame, time_col, value_col, notes):
    """Values on a regular date grid, with the pandas frequency and the natural season length."""
    times = pd.to_datetime(frame[time_col], errors="coerce")
    if times.isna().any():
        raise bad_request(f"'{time_col}' contains values that are not dates.")
    series = pd.Series(frame[value_col].to_numpy(dtype=float), index=times).sort_index()
    if series.index.has_duplicates:
        repeated = int(series.index.duplicated().sum())
        series = series.groupby(level=0).mean()
        notes.append(f"{repeated} repeated date{'s were' if repeated != 1 else ' was'} averaged.")
    if series.size < 3:
        raise bad_request("At least 3 dated observations are needed.")
    spacing = float(np.median(np.diff(series.index.values).astype("timedelta64[s]").astype(float)) / 86400)
    match = match_frequency(spacing) if spacing > 0 else None
    if match is None:
        raise bad_request("The dates are not evenly spaced (daily, weekly, monthly, quarterly or yearly). Aggregate the "
                          "data to a regular interval first.")
    _, freq, season, label = match
    regular = pd.date_range(series.index.min(), series.index.max(), freq=freq)
    if len(regular) < series.size:  # the inferred grid must be able to hold every observation
        regular = pd.date_range(series.index.min(), series.index.max(), periods=series.size)
    grid = series.reindex(regular.union(series.index)).reindex(regular)
    missing = int(grid.isna().sum())
    if missing > max(1, MAX_GAP_SHARE * len(grid)):
        raise bad_request(f"{missing} of {len(grid)} {label} periods have no value. Fill the gaps or aggregate the data "
                          "to a coarser interval.")
    if missing:
        grid = grid.interpolate(limit_direction="both")
        notes.append(f"{missing} missing {label} value{'s were' if missing != 1 else ' was'} filled by linear interpolation.")
    return grid, freq, season, label


def numeric_series(frame, time_col, value_col, notes):
    times = pd.to_numeric(frame[time_col], errors="coerce")
    if times.isna().any():
        raise bad_request(f"'{time_col}' must contain numbers (for example years) or dates.")
    series = pd.Series(frame[value_col].to_numpy(dtype=float), index=times).sort_index()
    if series.index.has_duplicates:
        repeated = int(series.index.duplicated().sum())
        series = series.groupby(level=0).mean()
        notes.append(f"{repeated} repeated time value{'s were' if repeated != 1 else ' was'} averaged.")
    values = series.index.to_numpy(dtype=float)
    steps = np.diff(values)
    step = float(np.median(steps)) if steps.size else 1.0
    if step <= 0:
        raise bad_request("The time column must increase.")
    if np.any(np.abs(steps - step) > 0.01 * step):
        raise bad_request("The time values are not evenly spaced. Aggregate the data to a regular interval first.")
    return series, step


def season_labels(index, season_length, kind, freq):
    """Labels for the positions inside a season and for each complete cycle."""
    if kind == "date" and season_length == 12:
        start = pd.Timestamp(index[0]).month - 1
        positions = [MONTHS[(start + i) % 12] for i in range(12)]
    elif kind == "date" and season_length == 4:
        start = (pd.Timestamp(index[0]).quarter - 1)
        positions = [f"Q{(start + i) % 4 + 1}" for i in range(4)]
    else:
        positions = [str(i + 1) for i in range(season_length)]
    cycles = []
    for start in range(0, len(index), season_length):
        point = index[start]
        cycles.append(str(pd.Timestamp(point).year) if kind == "date" else f"from {chart_gallery.format_value(float(point))}")
    return positions, cycles


@router.post("/timeseries")
def analyze_timeseries(
    file: UploadFile = File(...),
    value_col: str = Form(...),
    time_col: Optional[str] = Form(None),  # dates or numbers; default: row order
    season_length: Optional[int] = Form(None),  # 12 for monthly data, 4 for quarterly; 1 = no season
    horizon: int = Form(12),
    holdout: Optional[int] = Form(None),  # observations kept back to test the models
    transform: str = Form("none"),  # none or log
    palette: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user)
):
    if transform not in ("none", "log"):
        raise bad_request("Transformation must be 'none' or 'log'.")
    if not 1 <= horizon <= MAX_HORIZON:
        raise bad_request(f"The forecast horizon must be between 1 and {MAX_HORIZON} periods.")
    df = read_uploaded_table(file)
    time_col = time_col or None
    for column in [value_col] + ([time_col] if time_col else []):
        if column not in df.columns:
            raise bad_request(f"Column '{column}' not found in dataset.")
    if time_col == value_col:
        raise bad_request("The time column and the value column must be different.")

    notes = []
    frame = df[[value_col] + ([time_col] if time_col else [])].copy()
    values = pd.to_numeric(frame[value_col], errors="coerce")
    if (values.isna() & frame[value_col].notna()).any():
        raise bad_request(f"'{value_col}' contains non-numeric values.")
    frame[value_col] = values
    before = len(frame)
    frame = frame.dropna().reset_index(drop=True)
    if before - len(frame):
        dropped = before - len(frame)
        notes.append(f"{dropped} row{'s' if dropped != 1 else ''} with missing values {'were' if dropped != 1 else 'was'} left out.")

    freq, step, natural_season, frequency_label = None, 1.0, 1, "observation"
    if time_col:
        numeric = pd.to_numeric(frame[time_col], errors="coerce")
        if numeric.notna().all():
            series, step = numeric_series(frame, time_col, value_col, notes)
            kind, time_label = "numeric", time_col
            natural_season = 1
        else:
            series, freq, natural_season, frequency_label = date_series(frame, time_col, value_col, notes)
            kind, time_label = "date", time_col
    else:
        series = pd.Series(frame[value_col].to_numpy(dtype=float), index=np.arange(1, len(frame) + 1))
        kind, time_label = "order", "Observation"

    y = series.to_numpy(dtype=float)
    n = y.size
    if n < ts.MIN_OBSERVATIONS:
        raise bad_request(f"Forecasting needs at least {ts.MIN_OBSERVATIONS} observations (found {n}).")
    if n > ts.MAX_OBSERVATIONS:
        raise bad_request(f"At most {ts.MAX_OBSERVATIONS} observations can be analysed; aggregate the series first.")

    season = int(season_length) if season_length else natural_season
    if season < 1 or season > max(1, n // 2):
        raise bad_request(f"The season length must be between 1 and {n // 2} for {n} observations.")
    if season > 1 and n < 2 * season + 4:
        season = 1
        notes.append("The series is too short for its seasonal pattern, so it was analysed without a season.")

    if transform == "log":
        if np.any(y <= 0):
            raise bad_request("A log transformation needs values above zero.")
        y = np.log(y)
        notes.append("Values were log-transformed before modelling; forecasts and intervals are back-transformed, so "
                     "they describe the median rather than the mean.")

    minimum_train = max(2 * season + 4, 10)
    default_holdout = min(horizon, max(4, n // 5))
    requested = int(holdout) if holdout is not None else default_holdout
    test_size = max(0, min(requested, n - minimum_train))
    if requested and not test_size:
        notes.append("The series is too short to hold observations back for testing, so models were compared by corrected AIC.")
    train = y[: n - test_size] if test_size else y
    actual = y[n - test_size:] if test_size else None

    # Fit every candidate on the training part and score it on the test period
    keys = ts.available_models(train.size, season, bool(np.all(y > 0)))
    models, sarima_order, fits = [], None, {}
    for key in keys:
        try:
            fitted = ts.fit_model(key, train, max(test_size, 1), season)
        except Exception:
            continue
        if fitted is None:
            continue
        if key == "sarima":
            sarima_order = (fitted["order"], fitted["seasonal_order"])
        fits[key] = fitted
        models.append({
            "key": key,
            "name": ts.MODEL_NAMES[key],
            "detail": fitted.get("detail"),
            "aic": number(fitted.get("aic")),
            "aicc": number(fitted.get("aicc")),
            "bic": number(fitted.get("bic")),
            "holdout": ts.accuracy(actual, fitted["mean"][:test_size], train, season) if test_size else None,
        })
    if not models:
        raise bad_request("No forecasting model could be fitted to this series.")

    if test_size:
        best = min(models, key=lambda m: m["holdout"]["rmse"])
    else:
        scored = [m for m in models if m["aicc"] is not None]
        best = min(scored, key=lambda m: m["aicc"]) if scored else models[0]
    for model in models:
        model["selected"] = model["key"] == best["key"]

    # Refit the chosen model on the whole series for the forecast the user sees
    final = ts.fit_model(best["key"], y, horizon, season, sarima_order if best["key"] == "sarima" else None)
    fitted_values = np.asarray(final["fitted"], dtype=float)
    residuals = y - fitted_values
    residual_correlogram = ts.correlogram(residuals[np.isfinite(residuals)], season_length=season)
    clean_residuals = residuals[np.isfinite(residuals)]
    shapiro = {"p_value": None}
    if 3 <= clean_residuals.size <= 5000:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shapiro_result = stats.shapiro(clean_residuals)
        shapiro = {"statistic": float(shapiro_result.statistic), "p_value": float(shapiro_result.pvalue)}

    diagnostics = ts.stationarity_tests(y)
    suggested_d, suggested_seasonal_d = ts.differences_needed(y, season)
    series_correlogram = ts.correlogram(y, season_length=season)
    parts = ts.decompose(y, season)

    # Times for the forecast period
    if kind == "date":
        future_index = pd.date_range(series.index[-1], periods=horizon + 1, freq=freq)[1:]
        times, future_times = series.index.to_numpy(), future_index.to_numpy()
        time_values = [t.isoformat() for t in series.index]
        future_values = [t.isoformat() for t in future_index]
    else:
        last = float(series.index[-1])
        future_index = last + step * np.arange(1, horizon + 1)
        times, future_times = series.index.to_numpy(dtype=float), future_index
        time_values = numbers(times)
        future_values = numbers(future_index)

    back = np.exp if transform == "log" else (lambda v: v)
    forecast = {key: numbers(back(np.asarray(final[key], dtype=float)))
                for key in ("mean", "lower80", "upper80", "lower95", "upper95")}
    display_values = back(y)
    display_fitted = back(fitted_values)
    value_label = value_col if transform == "none" else f"{value_col} (log scale)"

    charts = {
        "forecast": timeseries_charts.forecast_chart(
            times, back(y), future_times, {k: back(np.asarray(final[k], dtype=float)) for k in
                                           ("mean", "lower80", "upper80", "lower95", "upper95")},
            display_fitted, best["detail"] or best["name"], best["holdout"], value_col, time_label, test_size, palette),
        "decomposition": timeseries_charts.decomposition_chart(times, y, parts, value_label, time_label, palette),
        "correlogram": timeseries_charts.correlogram_chart(series_correlogram, "Autocorrelation of the series", palette),
        "residuals": timeseries_charts.residual_chart(times, residuals, residual_correlogram, shapiro, time_label, palette),
        "models": timeseries_charts.model_comparison_chart(models, best["key"], palette=palette) if test_size else None,
        "seasonal": None,
    }
    if season > 1:
        positions, cycles = season_labels(series.index, season, kind, freq)
        charts["seasonal"] = timeseries_charts.seasonal_chart(display_values, season, positions, cycles, value_col, palette)

    return {
        "value_column": value_col,
        "series": {"times": time_values, "values": numbers(display_values), "kind": kind, "label": time_label,
                   "n": int(n), "frequency": frequency_label, "season_length": int(season), "transform": transform},
        "diagnostics": {
            **diagnostics,
            "correlogram": series_correlogram,
            "suggested_differences": {"regular": int(suggested_d), "seasonal": int(suggested_seasonal_d)},
        },
        "decomposition": {
            "method": parts["method"],
            "trend": numbers(parts["trend"]),
            "seasonal": numbers(parts["seasonal"]),
            "remainder": numbers(parts["remainder"]),
            "strengths": parts["strengths"],
        },
        "models": models,
        "best": best["key"],
        "holdout": int(test_size),
        "forecast": {"horizon": int(horizon), "times": future_values, **forecast,
                     "fitted": numbers(display_fitted), "model": best["detail"] or best["name"]},
        "residuals": {"values": numbers(residuals), "ljung_box": residual_correlogram["ljung_box"], "shapiro": shapiro},
        "charts": charts,
        "notes": notes,
    }
