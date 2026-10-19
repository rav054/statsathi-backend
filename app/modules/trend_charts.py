"""Charts for trend analysis: time series with Sen's slope and change point, Innovative Trend Analysis plot,
homogeneity test panel, and a summary of trend magnitude across variables."""

import base64

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless rendering for the web server
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

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


def series_chart(t, x, name, result, time_label, slope_unit, alpha, palette=None):
    spec = chart_gallery.analysis_spec(palette=palette)
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    fig, ax = plt.subplots(figsize=figure_size())
    ax.plot(t, x, color=spec.color(0), linewidth=1.0, alpha=0.55, zorder=2)
    ax.scatter(t, x, s=14, color=spec.color(0), zorder=3, label="Observations")
    if x.size >= 10:
        smooth = lowess(x, t, frac=0.5, return_sorted=True)
        ax.plot(smooth[:, 0], smooth[:, 1], color=spec.color(1), linewidth=1.8, zorder=4, label="LOESS smooth")
    sen = result["sen"]
    ax.plot(t, sen["intercept"] + sen["slope"] * t, color=COLORS["text"], linestyle="--", linewidth=1.6, zorder=5,
            label=f"Sen's slope: {format_value(sen['slope'])} {slope_unit}")
    pettitt = result["pettitt"]
    if pettitt["p_value"] < alpha:
        k = pettitt["change_index"]
        split = (t[k] + t[k + 1]) / 2
        ax.axvline(split, color=spec.color(2), linestyle=":", linewidth=1.5, zorder=1,
                   label=f"Change point (Pettitt, {format_p(pettitt['p_value'])})")
        ax.hlines(pettitt["mean_before"], t[0], split, color=spec.color(2), linewidth=2.2, zorder=4)
        ax.hlines(pettitt["mean_after"], split, t[-1], color=spec.color(2), linewidth=2.2, zorder=4)
    mk, modified = result["mann_kendall"], result["modified_mann_kendall"]
    ax.text(0.01, 0.98, f"Mann-Kendall: τ = {mk['tau']:.3f}, Z = {mk['z']:.2f}, {format_p(mk['p_value'])}\n"
                        f"Modified MK (autocorrelation-corrected): {format_p(modified['p_value'])}",
            transform=ax.transAxes, va="top", ha="left", fontsize=spec.tick_size * 0.9, color=COLORS["text"], zorder=6,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=COLORS["grid"], alpha=0.92))
    chart_gallery.value_grid(ax, spec, axis="both")
    chart_gallery.decorate(ax, spec, f"{name}: trend over time", time_label, name)
    chart_gallery.add_legend(ax, spec, loc="lower right")
    homogeneity = result["homogeneity"]
    chart_gallery.footnote(fig, spec, f"Trend: {mk['trend']} (Mann-Kendall, α = {alpha:g}). Homogeneity: {homogeneity['class']} "
                                      f"({homogeneity['tests_rejecting']} of {homogeneity['tests']} tests detect a break).")
    fig.tight_layout(rect=(0, chart_gallery.footnote_space(fig), 1, 1))
    return _png(fig)


def ita_chart(ita, name, palette=None):
    spec = chart_gallery.analysis_spec(palette=palette)
    first = np.asarray(ita["first_half"], dtype=float)
    second = np.asarray(ita["second_half"], dtype=float)
    low, high = min(first.min(), second.min()), max(first.max(), second.max())
    pad = 0.06 * ((high - low) or 1.0)
    line = np.array([low - pad, high + pad])
    side = figure_size()[0] * 0.85

    fig, ax = plt.subplots(figsize=(side, side * 0.92))
    parts = np.array_split(np.arange(first.size), 3)
    for idx, label, color in zip(parts, ("Low values", "Medium values", "High values"),
                                 (spec.color(0), spec.color(2), spec.color(1))):
        ax.scatter(first[idx], second[idx], s=30, color=color, edgecolors="white", linewidths=0.5, zorder=3, label=label)
    ax.plot(line, line, color=COLORS["text"], linewidth=1.2, zorder=2, label="1:1 line (no trend)")
    labelled = False
    for factor in (1.1, 0.9):
        band = line * factor
        if min(band) < line[1] and max(band) > line[0]:  # draw (and list) the ±10% lines only where they are visible
            ax.plot(line, band, color=COLORS["muted"], linestyle="--", linewidth=0.9, zorder=1,
                    label=None if labelled else "±10% of the 1:1 line")
            labelled = True
    ax.set_xlim(line)
    ax.set_ylim(line)
    ax.set_aspect("equal", adjustable="box")
    chart_gallery.value_grid(ax, spec, axis="both")
    chart_gallery.decorate(ax, spec, f"Innovative trend analysis: {name}", "First half of the series (sorted)",
                           "Second half of the series (sorted)")
    # the legend goes in the empty triangle: below the 1:1 line when values increased, above it otherwise
    chart_gallery.add_legend(ax, spec, loc="lower right" if np.mean(second - first) >= 0 else "upper left")
    changes = " / ".join("n/a" if r["change_percent"] is None else f"{r['change_percent']:+.1f}%" for r in ita["ranges"])
    indicator = "n/a" if ita["indicator"] is None else f"{ita['indicator']:.2f}"
    chart_gallery.footnote(fig, spec, f"Points above the 1:1 line increased in the second half. Trend indicator D = {indicator}; "
                                      f"ITA slope = {format_value(ita['slope'])} per time step. Low / medium / high values "
                                      f"changed by {changes}.")
    fig.tight_layout(rect=(0, chart_gallery.footnote_space(fig), 1, 1))
    return _png(fig)


def homogeneity_chart(t, name, result, time_label, alpha, palette=None):
    spec = chart_gallery.analysis_spec(palette=palette)
    t = np.asarray(t, dtype=float)
    width = figure_size()[0]
    fig, axes = plt.subplots(3, 1, figsize=(width, width * 0.95), sharex=True)

    pettitt = result["pettitt"]
    u = np.abs(np.asarray(pettitt["u"], dtype=float))
    k = pettitt["change_index"]
    axes[0].plot(t[:-1], u, color=spec.color(0), linewidth=1.5)
    axes[0].scatter([t[k]], [u[k]], color=spec.color(1), s=36, zorder=3)
    axes[0].set_title(f"Pettitt |U_t|: K = {pettitt['k']:.0f}, {format_p(pettitt['p_value'])}", loc="left",
                      fontsize=spec.label_size)

    snht = result["snht"]
    series = np.asarray(snht["series"], dtype=float)
    j = snht["change_index"]
    axes[1].plot(t[:-1], series, color=spec.color(0), linewidth=1.5)
    axes[1].scatter([t[j]], [series[j]], color=spec.color(1), s=36, zorder=3)
    axes[1].set_title(f"SNHT T_k: T0 = {snht['t0']:.2f}, {format_p(snht['p_value'])}", loc="left", fontsize=spec.label_size)

    buishand = result["buishand"]
    cumulative = np.asarray(buishand["series"], dtype=float)
    b = buishand["change_index"]
    axes[2].plot(t, cumulative, color=spec.color(0), linewidth=1.5)
    axes[2].axhline(0, color=COLORS["axis"], linewidth=0.9)
    axes[2].scatter([t[b]], [cumulative[b]], color=spec.color(1), s=36, zorder=3)
    axes[2].set_title(f"Buishand cumulative deviations S_k/D: Q/√n = {buishand['q']:.2f} ({format_p(buishand['p_value_q'])}), "
                      f"R/√n = {buishand['r']:.2f} ({format_p(buishand['p_value_r'])})", loc="left", fontsize=spec.label_size)
    for ax in axes:
        chart_gallery.value_grid(ax, spec, axis="both")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(spec.tick_size)
    axes[2].set_xlabel(time_label, fontsize=spec.label_size)

    fig.suptitle(f"Homogeneity tests: {name}", x=0.01, ha="left", fontweight="bold", fontsize=spec.title_size)
    vn = result["von_neumann"]
    homogeneity = result["homogeneity"]
    chart_gallery.footnote(fig, spec, f"Orange markers: most likely break. Von Neumann ratio = {vn['ratio']:.2f} "
                                      f"({format_p(vn['p_value'])}). Classification (Wijngaard et al. 2003): "
                                      f"{homogeneity['class']}, {homogeneity['tests_rejecting']} of {homogeneity['tests']} tests "
                                      f"detect a break at α = {alpha:g}. SNHT, Buishand and von Neumann p-values from "
                                      f"{result['simulations']} simulated homogeneous series.")
    fig.tight_layout(rect=(0, chart_gallery.footnote_space(fig), 1, 0.96))
    return _png(fig)


def summary_chart(names, results, alpha, palette=None):
    """Change over the period as a percentage of the mean, with the confidence interval of Sen's slope."""
    spec = chart_gallery.analysis_spec(palette=palette)
    rows = []
    for name in names:
        result = results[name]
        span = result["time_end"] - result["time_start"]
        mean = result["mean"]
        if mean == 0:
            continue
        scale = 100 * span / abs(mean)
        rows.append((name, result["sen"]["slope"] * scale, result["sen"]["ci_low"] * scale,
                     result["sen"]["ci_high"] * scale, result["mann_kendall"]["p_value"] < alpha))
    rows.sort(key=lambda r: r[1])
    width, height = figure_size()
    fig, ax = plt.subplots(figsize=(width, max(height * 0.7, 0.45 * len(rows) + 1.8)))
    for position, (name, value, low, high, significant) in enumerate(rows):
        color = (spec.color(1) if value > 0 else spec.color(0)) if significant else "#9CA3AF"
        ax.hlines(position, low, high, color=color, linewidth=2.2, zorder=2)
        ax.scatter(value, position, s=60, color=color, edgecolors="white", linewidths=0.8, zorder=3)
        label = f"{value:+.1f}%{' *' if significant else ''}"
        if value >= 0:
            ax.text(max(high, value), position, f"  {label}", va="center", ha="left", fontsize=spec.tick_size * 0.9,
                    color=COLORS["text"])
        else:  # negative changes are labelled on the left, away from the zero line
            ax.text(min(low, value), position, f"{label}  ", va="center", ha="right", fontsize=spec.tick_size * 0.9,
                    color=COLORS["text"])
    if rows:  # leave room for the value labels beyond the intervals
        low = min(0.0, min(r[2] for r in rows))
        high = max(0.0, max(r[3] for r in rows))
        span = (high - low) or 1.0
        ax.set_xlim(low - span * (0.22 if any(r[1] < 0 for r in rows) else 0.05),
                    high + span * (0.22 if any(r[1] >= 0 for r in rows) else 0.05))
    ax.axvline(0, color=COLORS["axis"], linewidth=1.0, zorder=1)
    ax.set_yticks(range(len(rows)), labels=[r[0] for r in rows])
    ax.set_ylim(-0.7, len(rows) - 0.3)
    chart_gallery.value_grid(ax, spec, axis="x")
    chart_gallery.decorate(ax, spec, "Change over the period (% of the mean)", "Sen's slope × period ÷ mean (%)", "")
    chart_gallery.footnote(fig, spec, f"Bars: 95% confidence interval of Sen's slope. Coloured (*): significant Mann-Kendall "
                                      f"trend at α = {alpha:g}; grey: not significant.")
    fig.tight_layout(rect=(0, chart_gallery.footnote_space(fig), 1, 1))
    return _png(fig)
