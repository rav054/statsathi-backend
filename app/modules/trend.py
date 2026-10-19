"""Trend analysis: Mann-Kendall family, Sen's slope, change-point and homogeneity tests, Innovative Trend Analysis."""

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from .. import trend as trend_stats
from ..auth import get_current_user
from ..models import User
from . import chart_gallery, trend_charts
from .common import read_uploaded_table

router = APIRouter()

MAX_VARIABLES = 20
ALPHAS = (0.01, 0.05, 0.10)


def bad_request(message):
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)


def decimal_years(dates):
    """Dates as decimal years (for example 1 July 2020 ≈ 2020.50), so slopes are per year."""
    years = dates.dt.year
    start = pd.to_datetime(years.astype("Int64").astype(str) + "-01-01", errors="coerce")
    days_in_year = np.where(dates.dt.is_leap_year, 366, 365)
    return years + (dates - start).dt.days / days_in_year


def time_axis(df, time_col):
    """Numeric time values, their kind, the slope unit and an axis label."""
    if not time_col:
        return pd.Series(np.arange(1, len(df) + 1), index=df.index, dtype=float), "order", "per observation", "Observation"
    raw = df[time_col]
    present = raw.notna().sum()
    numeric = pd.to_numeric(raw, errors="coerce")
    if present and numeric.notna().sum() >= 0.9 * present:
        return numeric.astype(float), "numeric", f"per {time_col}", time_col
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # "could not infer format": each value is parsed on its own
        dates = pd.to_datetime(raw, errors="coerce")
    if not present or dates.notna().sum() < 0.9 * present:
        raise bad_request("The time column must contain numbers (for example years) or dates.")
    return decimal_years(dates).astype(float), "date", "per year", f"{time_col} (years)"


@router.post("/trend")
def analyze_trend(
    file: UploadFile = File(...),
    value_cols_str: str = Form(...),  # comma-separated numeric columns to test
    time_col: Optional[str] = Form(None),  # years, dates or any numeric time; default: row order
    season_col: Optional[str] = Form(None),  # e.g. month, for the seasonal Mann-Kendall test
    alpha: float = Form(0.05),
    palette: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user)
):
    if not any(abs(alpha - a) < 1e-9 for a in ALPHAS):
        raise bad_request("The significance level must be 0.01, 0.05 or 0.10.")
    df = read_uploaded_table(file)
    values = list(dict.fromkeys(c.strip() for c in value_cols_str.split(",") if c.strip()))
    if not values:
        raise bad_request("Choose at least one variable to test for a trend.")
    if len(values) > MAX_VARIABLES:
        raise bad_request(f"Choose at most {MAX_VARIABLES} variables at a time.")
    missing = [c for c in values + [time_col, season_col] if c and c not in df.columns]
    if missing:
        raise bad_request(f"Column(s) not found in dataset: {', '.join(missing)}.")
    if time_col and time_col in values:
        raise bad_request("The time column cannot also be one of the variables tested.")
    if season_col and (season_col in values or season_col == time_col):
        raise bad_request("The season column must differ from the time column and the tested variables.")

    time, kind, unit, time_label = time_axis(df, time_col)
    notes, results, charts = [], {}, {}

    for column in values:
        x = pd.to_numeric(df[column], errors="coerce")
        if x.notna().sum() < df[column].notna().sum():
            raise bad_request(f"'{column}' contains non-numeric values.")
        frame = pd.DataFrame({"t": time, "x": x})
        if season_col:
            frame["season"] = df[season_col]
        frame = frame.dropna()
        frame["order_t"] = frame["t"]
        if season_col:
            frame["season"] = chart_gallery.as_category(frame["season"])
            if frame["t"].duplicated().any():
                # e.g. a year column plus a month column: place each season within its year
                seasons = sorted(frame["season"].unique(), key=chart_gallery.natural_key)
                position = {s: i for i, s in enumerate(seasons)}
                frame["order_t"] = frame["t"] + frame["season"].map(position) / len(seasons)
        repeated = int(frame["order_t"].duplicated().sum())
        if repeated:
            aggregations = {"x": ("x", "mean"), "t": ("t", "first")}
            if season_col:
                aggregations["season"] = ("season", "first")
            frame = frame.groupby("order_t", as_index=False).agg(**aggregations)
            notes.append(f"{column}: {repeated} repeated time value{'s were' if repeated != 1 else ' was'} averaged.")
        frame = frame.sort_values("order_t")
        n = len(frame)
        if n < trend_stats.MIN_OBSERVATIONS:
            raise bad_request(f"'{column}' has {n} usable observations; trend tests need at least {trend_stats.MIN_OBSERVATIONS}.")
        if n > trend_stats.MAX_OBSERVATIONS:
            raise bad_request(f"'{column}' has {n} observations; aggregate the data (for example to monthly or annual "
                              f"values) to at most {trend_stats.MAX_OBSERVATIONS}.")
        if frame["x"].nunique() < 2:
            raise bad_request(f"'{column}' is constant, so no trend can be tested.")

        t_values = frame["order_t"].to_numpy(dtype=float)
        x_values = frame["x"].to_numpy(dtype=float)
        seasons = frame["season"].to_numpy() if season_col else None
        result = trend_stats.analyse_series(t_values, x_values, alpha, seasons)
        result["time_values"] = t_values.tolist()
        result["values"] = x_values.tolist()
        results[column] = result
        charts[column] = {
            "series": trend_charts.series_chart(t_values, x_values, column, result, time_label, unit, alpha, palette),
            "ita": trend_charts.ita_chart(result["ita"], column, palette),
            "homogeneity": trend_charts.homogeneity_chart(t_values, column, result, time_label, alpha, palette),
        }

    return {
        "variables": values,
        "time": {"column": time_col, "kind": kind, "unit": unit, "label": time_label},
        "season_column": season_col,
        "alpha": alpha,
        "results": results,
        "charts": charts,
        "summary_chart": trend_charts.summary_chart(values, results, alpha, palette) if len(values) > 1 else None,
        "notes": notes,
    }
