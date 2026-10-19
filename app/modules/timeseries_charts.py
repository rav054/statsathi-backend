"""Time-series charts: forecast with prediction intervals, decomposition, correlogram, model comparison,
residual diagnostics and the seasonal plot."""

import base64

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless rendering for the web server
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.lines import Line2D

from ..figure_style import COLORS, export_figure, figure_size
from . import chart_gallery

format_value = chart_gallery.format_value
format_p = chart_gallery.format_p


def _png(fig):
    try:
        content, _ = export_figure(fig, "png", 150)
    finally:
        plt.close(fig)
    return base64.b64encode(content).decode("utf-8")


def _tidy(ax, spec, title, xlabel, ylabel):
    chart_gallery.value_grid(ax, spec, axis="both")
    chart_gallery.decorate(ax, spec, title, xlabel, ylabel)


def forecast_chart(times, values, future_times, forecast, fitted, model_name, accuracy, value_label, time_label,
                   holdout=0, palette=None):
    spec = chart_gallery.analysis_spec(palette=palette)
    fig, ax = plt.subplots(figsize=figure_size())
    ax.plot(times, values, color=spec.color(0), linewidth=1.3, marker="o", markersize=2.5, zorder=3, label="Observed")
    if fitted is not None:
        shown = np.asarray(fitted, dtype=float)
        ax.plot(times, shown, color=COLORS["muted"], linewidth=1.0, linestyle="--", zorder=2, label="Fitted (one step ahead)")
    mean = np.asarray(forecast["mean"], dtype=float)
    ax.fill_between(future_times, forecast["lower95"], forecast["upper95"], color=spec.color(1), alpha=0.15, linewidth=0,
                    zorder=1, label="95% interval")
    ax.fill_between(future_times, forecast["lower80"], forecast["upper80"], color=spec.color(1), alpha=0.28, linewidth=0,
                    zorder=1, label="80% interval")
    ax.plot(future_times, mean, color=spec.color(1), linewidth=1.8, marker="o", markersize=3, zorder=4, label="Forecast")
    ax.axvline(times[-1], color=COLORS["grid"], linewidth=1.0, zorder=0)
    _tidy(ax, spec, f"Forecast: {model_name}", time_label, value_label)
    chart_gallery.add_legend(ax, spec, loc="best")
    note = f"Forecast from {model_name}."
    if accuracy:
        parts = [f"RMSE {format_value(accuracy['rmse'])}", f"MAE {format_value(accuracy['mae'])}"]
        if accuracy.get("mape") is not None:
            parts.append(f"MAPE {accuracy['mape']:.1f}%")
        if accuracy.get("mase") is not None:
            parts.append(f"MASE {accuracy['mase']:.2f}")
        note += f" Accuracy on the last {holdout} observations held out for testing: " + ", ".join(parts) + "."
    note += (" Shaded bands are 80% and 95% prediction intervals: the wider the band, the less certain the forecast. "
             "They assume the pattern in the data continues.")
    chart_gallery.footnote(fig, spec, note)
    fig.tight_layout(rect=(0, chart_gallery.footnote_space(fig), 1, 1))
    return _png(fig)


def decomposition_chart(times, values, parts, value_label, time_label, palette=None):
    spec = chart_gallery.analysis_spec(palette=palette)
    width, height = figure_size()
    fig, axes = plt.subplots(4, 1, figsize=(width, height * 1.5), sharex=True)
    panels = [("Observed", values, spec.color(0)), ("Trend", parts["trend"], spec.color(1)),
              ("Seasonal", parts["seasonal"], spec.color(2)), ("Remainder", parts["remainder"], COLORS["muted"])]
    for ax, (title, series, color) in zip(axes, panels):
        series = np.asarray(series, dtype=float)
        if title == "Remainder":
            ax.axhline(0, color=COLORS["grid"], linewidth=0.9, zorder=1)
            ax.vlines(times, 0, np.nan_to_num(series), color=color, linewidth=0.9, zorder=2)
        else:
            ax.plot(times, series, color=color, linewidth=1.3, zorder=2)
        ax.set_title(title, loc="left", fontsize=spec.label_size, color=COLORS["text"])
        chart_gallery.value_grid(ax, spec, axis="both")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(spec.tick_size)
    axes[-1].set_xlabel(time_label, fontsize=spec.label_size)
    fig.suptitle(f"Decomposition of {value_label}", x=0.01, ha="left", fontweight="bold", fontsize=spec.title_size,
                 color=COLORS["text"])
    trend_strength, seasonal_strength = parts["strengths"]["trend"], parts["strengths"]["seasonal"]
    note = f"{parts['method'].upper() if parts['method'] == 'stl' else parts['method'].capitalize()} decomposition. "
    if trend_strength is not None:
        note += f"Strength of trend {trend_strength:.2f}"
        if seasonal_strength is not None:
            note += f", strength of seasonality {seasonal_strength:.2f}"
        note += " (0 = none, 1 = very strong). "
    note += "The remainder is what neither the trend nor the season explains."
    chart_gallery.footnote(fig, spec, note)
    fig.tight_layout(rect=(0, chart_gallery.footnote_space(fig), 1, 0.96))
    return _png(fig)


def correlogram_chart(correlogram, title="Autocorrelation", palette=None):
    spec = chart_gallery.analysis_spec(palette=palette)
    lags = np.asarray(correlogram["lags"])
    bound = correlogram["bound"]
    width, height = figure_size()
    fig, axes = plt.subplots(1, 2, figsize=(width * 1.15, height * 0.8))
    for ax, key, label in zip(axes, ("acf", "pacf"), ("ACF (autocorrelation)", "PACF (partial autocorrelation)")):
        values = np.asarray(correlogram[key], dtype=float)
        ax.axhspan(-bound, bound, color=spec.color(1), alpha=0.15, linewidth=0, zorder=0)
        ax.vlines(lags, 0, values, color=spec.color(0), linewidth=1.6, zorder=2)
        ax.scatter(lags, values, s=14, color=spec.color(0), zorder=3)
        ax.axhline(0, color=COLORS["axis"], linewidth=0.9, zorder=1)
        ax.set_ylim(min(-0.25, values.min() * 1.15 - 0.05), max(0.35, values.max() * 1.15 + 0.05))
        _tidy(ax, spec, label, "Lag", "Correlation")
        ax.title.set_fontsize(spec.label_size)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=spec.title_size, color=COLORS["text"])
    test = correlogram["ljung_box"]
    chart_gallery.footnote(fig, spec, f"Shaded band: values inside it are within what random noise produces (95%). "
                                      f"Ljung-Box test up to lag {test['lag']}: {format_p(test['p_value'])} "
                                      f"{'(autocorrelation remains)' if test['p_value'] < 0.05 else '(no autocorrelation left)'}.")
    fig.tight_layout(rect=(0, chart_gallery.footnote_space(fig), 1, 0.93))
    return _png(fig)


def model_comparison_chart(models, best_key, metric="rmse", palette=None):
    spec = chart_gallery.analysis_spec(palette=palette)
    rows = [m for m in models if m.get("holdout") and m["holdout"].get(metric) is not None]
    rows.sort(key=lambda m: m["holdout"][metric], reverse=True)
    values = [m["holdout"][metric] for m in rows]
    width, height = figure_size()
    fig, ax = plt.subplots(figsize=(width, max(height * 0.7, 0.42 * len(rows) + 1.7)))
    colors = [spec.color(1) if m["key"] == best_key else "#CBD5E1" for m in rows]
    ax.barh([m["name"] for m in rows], values, color=colors, height=0.62, zorder=2)
    for position, (row, value) in enumerate(zip(rows, values)):
        mase = row["holdout"].get("mase")
        label = f"  {format_value(value)}" + (f" (MASE {mase:.2f})" if mase is not None else "")
        ax.text(value, position, label, va="center", fontsize=spec.tick_size * 0.85, color=COLORS["text"])
    ax.set_xlim(0, max(values) * 1.28 if values else 1)
    chart_gallery.value_grid(ax, spec, axis="x")
    chart_gallery.decorate(ax, spec, "Forecast accuracy on the test period", metric.upper(), "")
    chart_gallery.footnote(fig, spec, "Lower is better. Each model was fitted without the test period and then asked to "
                                      "forecast it. The coloured bar is the model chosen for the forecast. MASE below 1 "
                                      "beats a seasonal naive forecast.")
    fig.tight_layout(rect=(0, chart_gallery.footnote_space(fig), 1, 1))
    return _png(fig)


def residual_chart(times, residuals, correlogram, shapiro, time_label, palette=None):
    spec = chart_gallery.analysis_spec(palette=palette)
    residuals = np.asarray(residuals, dtype=float)
    valid = np.isfinite(residuals)
    width, height = figure_size()
    fig = plt.figure(figsize=(width, height * 1.25))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    top = fig.add_subplot(grid[0, :])
    top.axhline(0, color=COLORS["grid"], linewidth=1.0, zorder=1)
    top.plot(np.asarray(times)[valid], residuals[valid], color=spec.color(0), linewidth=1.1, marker="o", markersize=2.5, zorder=2)
    _tidy(top, spec, "Residuals over time", time_label, "Residual")
    top.title.set_fontsize(spec.label_size)

    left = fig.add_subplot(grid[1, 0])
    lags = np.asarray(correlogram["lags"])
    values = np.asarray(correlogram["acf"], dtype=float)
    left.axhspan(-correlogram["bound"], correlogram["bound"], color=spec.color(1), alpha=0.15, linewidth=0, zorder=0)
    left.vlines(lags, 0, values, color=spec.color(0), linewidth=1.5, zorder=2)
    left.axhline(0, color=COLORS["axis"], linewidth=0.9, zorder=1)
    _tidy(left, spec, "ACF of residuals", "Lag", "Correlation")
    left.title.set_fontsize(spec.label_size)

    right = fig.add_subplot(grid[1, 1])
    clean = residuals[valid]
    right.hist(clean, bins=min(20, max(6, clean.size // 4)), color=spec.color(0), alpha=0.75, density=True, zorder=2)
    span = np.linspace(clean.min(), clean.max(), 200)
    right.plot(span, stats.norm.pdf(span, clean.mean(), clean.std(ddof=1)), color=spec.color(1), linewidth=1.5, zorder=3)
    _tidy(right, spec, "Distribution of residuals", "Residual", "Density")
    right.title.set_fontsize(spec.label_size)
    chart_gallery.add_legend(right, spec, handles=[Line2D([], [], color=spec.color(1), label="Normal curve")], loc="best")

    fig.suptitle("Residual diagnostics", x=0.01, ha="left", fontweight="bold", fontsize=spec.title_size, color=COLORS["text"])
    test = correlogram["ljung_box"]
    note = (f"Good residuals look like noise: centred on zero, no pattern over time and no bars outside the shaded band. "
            f"Ljung-Box up to lag {test['lag']}: {format_p(test['p_value'])}.")
    if shapiro and shapiro.get("p_value") is not None:
        note += f" Shapiro-Wilk normality: {format_p(shapiro['p_value'])} (matters for the prediction intervals)."
    chart_gallery.footnote(fig, spec, note)
    fig.tight_layout(rect=(0, chart_gallery.footnote_space(fig), 1, 0.95))
    return _png(fig)


def seasonal_chart(values, season_length, position_labels, cycle_labels, value_label, palette=None):
    spec = chart_gallery.analysis_spec(palette=palette)
    values = np.asarray(values, dtype=float)
    positions = np.arange(values.size) % season_length
    cycles = np.arange(values.size) // season_length
    unique_cycles = np.unique(cycles)
    fig, ax = plt.subplots(figsize=figure_size())
    for index, cycle in enumerate(unique_cycles):
        mask = cycles == cycle
        color = spec.color(index) if len(unique_cycles) <= 8 else "#CBD5E1"
        ax.plot(positions[mask], values[mask], color=color, linewidth=1.0, marker="o", markersize=2.5, alpha=0.85,
                zorder=2, label=str(cycle_labels[index]) if len(unique_cycles) <= 8 else None)
    means = [np.nanmean(values[positions == p]) for p in range(season_length)]
    ax.plot(range(season_length), means, color=COLORS["text"], linewidth=2.2, zorder=4, label="Average")
    ax.set_xticks(range(season_length), labels=position_labels)
    _tidy(ax, spec, "Seasonal pattern", "Position in the season", value_label)
    chart_gallery.rotate_long_labels(ax)
    chart_gallery.add_legend(ax, spec, loc="best")
    note = "Each line is one complete season (for monthly data, one year); the thick line is the average of all of them."
    if len(unique_cycles) > 8:
        note += " Individual seasons are drawn in grey because there are many of them."
    chart_gallery.footnote(fig, spec, note)
    fig.tight_layout(rect=(0, chart_gallery.footnote_space(fig), 1, 1))
    return _png(fig)
