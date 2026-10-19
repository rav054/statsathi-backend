"""Modern chart types for the Plots module.

Distributions: raincloud, box + swarm, ridgeline, ECDF.
Comparisons:   mean ± 95% CI, lollipop, dumbbell, paired lines, two-factor mean heatmap.
Relationships: scatter with marginal distributions, 2-D density, hexbin.
Composition:   mosaic plot, radar chart.

Every chart builds its own figure (several need more than one set of axes) in the shared figure style.
"""

import re
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib

matplotlib.use("Agg")  # headless rendering for the web server
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import HTTPException, status
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from ..figure_style import COLORS, FONT_STACK, colormap_name, figure_size, palette_colors

FONT_FAMILIES = {
    "sans-serif": FONT_STACK + ["DejaVu Sans"],
    "serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "monospace": ["Courier New", "DejaVu Sans Mono", "monospace"],
}


@dataclass
class ChartSpec:
    x: Optional[str]
    y: Optional[str]
    hue: Optional[str]
    y_vars: List[str]
    colors: List[str]
    cmap: str
    figsize: tuple
    font_scale: float = 1.0
    fit_reg: bool = False
    show_grid: bool = True
    legend_loc: str = "best"
    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    text_color: str = COLORS["text"]
    title_size: float = 12
    label_size: float = 10
    tick_size: float = 9
    title_family: str = "sans-serif"
    label_family: str = "sans-serif"
    tick_family: str = "sans-serif"
    extra: dict = field(default_factory=dict)

    def color(self, index):
        return self.colors[index % len(self.colors)]


# ---------------------------------------------------------------------------
# Statistics drawn on the charts (tested against scipy / statsmodels)
# ---------------------------------------------------------------------------
def mean_confidence_interval(values, level=0.95):
    """Mean and t-based confidence interval. The interval is NaN for fewer than two values."""
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if values.size < 2:
        return mean, float("nan"), float("nan")
    half = stats.t.ppf((1 + level) / 2, values.size - 1) * values.std(ddof=1) / np.sqrt(values.size)
    return mean, float(mean - half), float(mean + half)


def regression_band(x, y, grid, level=0.95):
    """Least-squares line and the confidence band of the mean response at `grid`."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    grid = np.asarray(grid, dtype=float)
    n = x.size
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * grid
    residual = y - (intercept + slope * x)
    s = np.sqrt(np.sum(residual ** 2) / (n - 2))
    sxx = np.sum((x - x.mean()) ** 2)
    half = stats.t.ppf((1 + level) / 2, n - 2) * s * np.sqrt(1 / n + (grid - x.mean()) ** 2 / sxx)
    return fitted, fitted - half, fitted + half


def difference_interval(a, b, equal_var=False, level=0.95):
    """Difference in means (a − b) with its t confidence interval.

    Returns (difference, low, high, standard error, df) using the pooled (Student) or Welch standard error.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = a.size, b.size
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    difference = a.mean() - b.mean()
    if equal_var:
        pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        se = np.sqrt(pooled * (1 / n1 + 1 / n2))
        dof = n1 + n2 - 2
    else:
        se = np.sqrt(v1 / n1 + v2 / n2)
        dof = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)) if se > 0 else n1 + n2 - 2
    half = stats.t.ppf((1 + level) / 2, dof) * se
    return float(difference), float(difference - half), float(difference + half), float(se), float(dof)


def paired_difference_interval(a, b, level=0.95):
    """Mean of the paired differences (a − b) with its t confidence interval: (mean, low, high, se, df)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    se = d.std(ddof=1) / np.sqrt(d.size)
    dof = d.size - 1
    half = stats.t.ppf((1 + level) / 2, dof) * se
    return float(d.mean()), float(d.mean() - half), float(d.mean() + half), float(se), float(dof)


def standardized_residuals(observed):
    """Pearson residuals (observed - expected) / sqrt(expected) of a contingency table."""
    observed = np.asarray(observed, dtype=float)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        residuals = (observed - expected) / np.sqrt(expected)
    return np.nan_to_num(residuals)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def bad_request(message):
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)


def require(value, message):
    if not value:
        raise bad_request(message)
    return value


def natural_key(value):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]


def ordered_levels(series, name, max_levels, spec=None):
    """Levels in natural order, or in the order given by spec.extra["orders"][name] (e.g. the column order chosen)."""
    present = list(pd.unique(series.dropna()))
    preferred = (spec.extra.get("orders") or {}).get(name) if spec is not None else None
    if preferred:
        present_text = {str(v) for v in present}
        preferred_text = [str(v) for v in preferred]
        levels = [v for v in preferred_text if v in present_text]
        levels += sorted((v for v in present if str(v) not in set(preferred_text)), key=natural_key)
    else:
        levels = sorted(present, key=natural_key)
    if len(levels) > max_levels:
        raise bad_request(f"'{name}' has {len(levels)} levels; this chart shows at most {max_levels}.")
    return levels


def as_category(series):
    if pd.api.types.is_float_dtype(series) and np.all(np.mod(series.dropna(), 1) == 0):
        series = series.astype("Int64")
    return series.astype(str)


def clean_frame(df, numeric=(), categorical=()):
    """Selected columns with numeric ones validated, categories as text, incomplete rows removed."""
    columns = list(dict.fromkeys(c for c in (*numeric, *categorical) if c))
    frame = df[columns].copy()
    for column in dict.fromkeys(c for c in numeric if c):
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().sum() < frame[column].notna().sum() or converted.notna().sum() == 0:
            raise bad_request(f"'{column}' must be numeric for this chart.")
        frame[column] = converted
    frame = frame.dropna()
    for column in dict.fromkeys(c for c in categorical if c):
        if column not in numeric:
            frame[column] = as_category(frame[column])
    if len(frame) < 2:
        raise bad_request("Not enough complete observations for this chart after removing missing values.")
    return frame


def family(name):
    return FONT_FAMILIES.get(name or "sans-serif", FONT_FAMILIES["sans-serif"])


def format_value(value):
    if value == 0 or 0.01 <= abs(value) < 10000:
        return f"{value:.2f}"
    return f"{value:.3g}"


def format_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def readable_text(rgba):
    red, green, blue = to_rgb(rgba)
    return COLORS["text"] if 0.299 * red + 0.587 * green + 0.114 * blue > 0.55 else "#FFFFFF"


def rotate_long_labels(ax):
    labels = ax.get_xticklabels()
    longest = max((len(t.get_text()) for t in labels), default=0)
    if longest * len(labels) > 48:
        plt.setp(labels, rotation=30, ha="right", rotation_mode="anchor")


def value_grid(ax, spec, axis="y"):
    ax.grid(False)
    if spec.show_grid:
        ax.grid(True, axis=axis)


def decorate(ax, spec, default_title, default_xlabel=None, default_ylabel=None, figure_title=False):
    title_style = dict(fontsize=spec.title_size, fontweight="bold", color=spec.text_color, family=family(spec.title_family))
    title = spec.title or default_title
    if figure_title:
        ax.figure.suptitle(title, x=0.01, ha="left", **title_style)
    else:
        ax.set_title(title, **title_style)
    label_style = dict(fontsize=spec.label_size, color=spec.text_color, family=family(spec.label_family))
    if default_xlabel is not None:
        ax.set_xlabel(spec.xlabel or default_xlabel, **label_style)
    if default_ylabel is not None:
        ax.set_ylabel(spec.ylabel or default_ylabel, **label_style)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(spec.tick_size)
        label.set_family(family(spec.tick_family))
        label.set_color(spec.text_color)


def add_legend(ax, spec, title=None, handles=None, **kwargs):
    if spec.legend_loc == "none":
        if ax.get_legend():
            ax.get_legend().remove()
        return
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    else:
        labels = [h.get_label() for h in handles]
    if not handles:
        return
    ax.legend(handles, labels, loc=kwargs.pop("loc", spec.legend_loc), title=title,
              prop={"size": spec.tick_size, "family": family(spec.tick_family)},
              title_fontsize=spec.label_size, **kwargs)


def analysis_spec(x=None, y=None, hue=None, y_vars=(), palette=None, extra=None, title=None, xlabel=None, ylabel=None):
    """Chart settings for figures drawn from analysis results (screen size, default fonts)."""
    return ChartSpec(x=x, y=y, hue=hue, y_vars=list(y_vars), colors=palette_colors(palette), cmap=colormap_name(palette),
                     figsize=figure_size(), extra=extra or {}, title=title, xlabel=xlabel, ylabel=ylabel)


def annotate_groups(ax, spec, positions, groups, tops):
    """Optional extras supplied by the analysis: a reference line, compact letters, or a two-group bracket."""
    positions = list(positions)
    reference = spec.extra.get("reference")
    low, high = ax.get_ylim()
    if reference is not None:
        value = reference["value"]
        ax.axhline(value, color=COLORS["muted"], linestyle="--", linewidth=1.1, zorder=1, label=reference.get("label"))
        margin = 0.05 * (high - low)
        low, high = min(low, value - margin), max(high, value + margin)
        ax.set_ylim(low, high)
    letters = spec.extra.get("letters") or {}
    bracket = spec.extra.get("bracket")
    pad = 0.04 * (high - low)
    top = high
    if letters:
        for position, group, value in zip(positions, groups, tops):
            letter = letters.get(str(group))
            if letter:
                ax.text(position, value + pad, str(letter), ha="center", va="bottom", fontsize=spec.label_size,
                        fontweight="bold", color=spec.text_color, zorder=5)
                top = max(top, value + pad * 3)
    elif bracket and len(groups) == 2:
        y = max(tops) + pad
        rise = pad * 0.6
        ax.plot([positions[0], positions[0], positions[1], positions[1]], [y, y + rise, y + rise, y],
                color=spec.text_color, linewidth=1.0, zorder=5)
        ax.text((positions[0] + positions[1]) / 2, y + rise * 1.3, bracket, ha="center", va="bottom",
                fontsize=spec.tick_size, color=spec.text_color, zorder=5)
        top = max(top, y + pad * 3)
    ax.set_ylim(low, top)
    if reference is not None and reference.get("label"):
        add_legend(ax, spec)


def footnote(fig, spec, text):
    """Small explanatory note under the chart, wrapped to the figure width."""
    size = spec.tick_size * 0.85
    width_in, height_in = fig.get_size_inches()
    wrapped = textwrap.fill(text, max(40, int(width_in * 72 / (size * 0.52))))
    lines = wrapped.count("\n") + 1
    fig.text(0.01, 0.005, wrapped, ha="left", va="bottom", fontsize=size, linespacing=1.3,
             color=COLORS["muted"], family=family(spec.tick_family))
    fig.statsathi_footnote_height = min(0.3, (lines * size * 1.35 + 6) / 72 / height_in)


def footnote_space(fig):
    """Fraction of the figure height reserved at the bottom for the footnote."""
    return getattr(fig, "statsathi_footnote_height", 0.0)


def nudge_labels(xy, ax, min_dx=0.09, min_dy=0.045, passes=4):
    """Label anchors (data coordinates) moved apart vertically where labels would overlap, using the current axis limits."""
    xy = np.array(xy, dtype=float, copy=True)
    if len(xy) < 2:
        return xy
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx, dy = min_dx * abs(x1 - x0), min_dy * abs(y1 - y0)
    order = np.argsort(xy[:, 1], kind="stable")
    for _ in range(passes):
        moved = False
        for position, i in enumerate(order):
            for j in order[:position]:
                if abs(xy[i, 0] - xy[j, 0]) < dx and abs(xy[i, 1] - xy[j, 1]) < dy:
                    xy[i, 1] = xy[j, 1] + dy if xy[i, 1] >= xy[j, 1] else xy[j, 1] - dy
                    moved = True
        if not moved:
            break
    return xy


def density_curve(values, grid):
    values = np.asarray(values, dtype=float)
    if values.size > 2 and values.std() > 0:
        return stats.gaussian_kde(values)(grid)
    return None


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------
def raincloud(df, spec):
    x = require(spec.x, "Choose the grouping variable (X axis).")
    y = require(spec.y, "Choose the numeric variable (Y axis).")
    frame = clean_frame(df, numeric=[y], categorical=[x])
    groups = ordered_levels(frame[x], x, 20, spec)
    rng = np.random.default_rng(0)
    scale = spec.font_scale

    fig, ax = plt.subplots(figsize=spec.figsize)
    tops = []
    for i, group in enumerate(groups):
        values = frame.loc[frame[x] == group, y].to_numpy(dtype=float)
        color = spec.color(i)
        top = values.max()
        if values.size > 2 and values.std() > 0:
            kde = stats.gaussian_kde(values)
            reach = 3 * kde.factor * values.std(ddof=1)  # three bandwidths past the data so the cloud tapers off
            grid = np.linspace(values.min() - reach, values.max() + reach, 200)
            density = kde(grid)
            ax.fill_betweenx(grid, i + 0.08, i + 0.08 + density / density.max() * 0.38,
                             color=color, alpha=0.55, linewidth=0)
        ax.boxplot(values, positions=[i], widths=0.1, orientation="vertical", patch_artist=True, showfliers=False,
                   manage_ticks=False,
                   boxprops=dict(facecolor="white", edgecolor=COLORS["text"], linewidth=0.8),
                   medianprops=dict(color=COLORS["text"], linewidth=1.4),
                   whiskerprops=dict(color=COLORS["text"], linewidth=0.8), capprops=dict(linewidth=0))
        ax.scatter(i - 0.12 - rng.uniform(0, 0.22, values.size), values, s=14 * scale ** 2, color=color,
                   alpha=0.7, edgecolors="none", zorder=3)
        tops.append(top)

    ax.set_xticks(range(len(groups)), labels=groups)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    value_grid(ax, spec)
    annotate_groups(ax, spec, range(len(groups)), groups, tops)
    decorate(ax, spec, f"Raincloud plot of {y} by {x}", x, y)
    rotate_long_labels(ax)
    footnote(fig, spec, "Points: observations. Box: median and quartiles. Cloud: estimated density.")
    fig.tight_layout(rect=(0, footnote_space(fig), 1, 1))
    return fig


def box_swarm(df, spec):
    x = require(spec.x, "Choose the grouping variable (X axis).")
    y = require(spec.y, "Choose the numeric variable (Y axis).")
    hue = spec.hue if spec.hue and spec.hue != x else None
    frame = clean_frame(df, numeric=[y], categorical=[x, hue])
    groups = ordered_levels(frame[x], x, 25, spec)
    hue_levels = ordered_levels(frame[hue], hue, 8, spec) if hue else None
    color_key = hue or x
    color_levels = hue_levels or groups
    palette = {level: spec.color(i) for i, level in enumerate(color_levels)}

    fig, ax = plt.subplots(figsize=spec.figsize)
    sns.boxplot(data=frame, x=x, y=y, hue=color_key, order=groups, hue_order=color_levels, palette=palette,
                width=0.6, fliersize=0, linewidth=0.8, boxprops=dict(alpha=0.35), dodge=bool(hue), legend=False, ax=ax)
    largest_group = frame.groupby([x] + ([hue] if hue else [])).size().max()
    point_options = dict(data=frame, x=x, y=y, hue=color_key, order=groups, hue_order=color_levels, palette=palette,
                         dodge=bool(hue), size=3.2 * spec.font_scale, legend=bool(hue), ax=ax)
    if largest_group <= 120:
        sns.swarmplot(**point_options)
    else:  # swarms of hundreds of points overflow; jittered points stay readable
        sns.stripplot(jitter=0.25, alpha=0.5, **point_options)

    value_grid(ax, spec)
    if not hue:
        annotate_groups(ax, spec, range(len(groups)), groups, [frame.loc[frame[x] == g, y].max() for g in groups])
    decorate(ax, spec, f"{y} by {x}: box plots with every observation", x, y)
    rotate_long_labels(ax)
    if hue:
        add_legend(ax, spec, title=hue)
    fig.tight_layout()
    return fig


def ridgeline(df, spec):
    x = require(spec.x, "Choose the numeric variable (X axis).")
    hue = require(spec.hue, "Choose the grouping variable (one ridge per group).")
    frame = clean_frame(df, numeric=[x], categorical=[hue])
    groups = ordered_levels(frame[hue], hue, 25, spec)
    values_all = frame[x].to_numpy(dtype=float)
    span = (values_all.max() - values_all.min()) or 1.0
    grid = np.linspace(values_all.min() - 0.1 * span, values_all.max() + 0.1 * span, 300)
    densities = [density_curve(frame.loc[frame[hue] == g, x], grid) for g in groups]
    peak = max((d.max() for d in densities if d is not None), default=1.0)

    k = len(groups)
    fig, axes = plt.subplots(k, 1, figsize=(spec.figsize[0], max(spec.figsize[1], 0.45 * k + 1.4)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    for i, (ax, group, density) in enumerate(zip(axes, groups, densities)):
        color = spec.color(i)
        ax.patch.set_alpha(0)
        if density is not None:
            ax.fill_between(grid, density, color=color, alpha=0.8, linewidth=0)
            ax.plot(grid, density, color="white", linewidth=1.0)
        else:
            ax.vlines(frame.loc[frame[hue] == group, x], 0, peak * 0.3, color=color, linewidth=1.2)
        ax.axhline(0, color=color, linewidth=1.0)
        ax.set_ylim(0, peak * 1.05)
        ax.set_yticks([])
        ax.grid(False)
        for side in ("left", "right", "top"):
            ax.spines[side].set_visible(False)
        if i < k - 1:
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(bottom=False)
        ax.text(-0.01, 0.04, str(group), transform=ax.transAxes, ha="right", va="bottom",
                fontsize=spec.tick_size, color=spec.text_color, family=family(spec.tick_family))

    decorate(axes[-1], spec, f"Distribution of {x} by {hue}", x, "", figure_title=True)
    longest = max(len(str(g)) for g in groups)
    fig.subplots_adjust(left=min(0.4, 0.06 + 0.011 * longest * spec.tick_size / 9), right=0.97,
                        top=0.9, bottom=0.14, hspace=-0.45)
    return fig


def ecdf(df, spec):
    x = require(spec.x, "Choose the numeric variable (X axis).")
    hue = spec.hue if spec.hue and spec.hue != x else None
    frame = clean_frame(df, numeric=[x], categorical=[hue])
    levels = ordered_levels(frame[hue], hue, 10, spec) if hue else [None]

    fig, ax = plt.subplots(figsize=spec.figsize)
    for i, level in enumerate(levels):
        values = np.sort((frame[x] if level is None else frame.loc[frame[hue] == level, x]).to_numpy(dtype=float))
        proportion = np.arange(1, values.size + 1) / values.size
        ax.step(np.r_[values[0], values], np.r_[0, proportion], where="post", color=spec.color(i),
                linewidth=1.8 * spec.font_scale, label=None if level is None else str(level))
    ax.axhline(0.5, color=COLORS["axis"], linestyle=":", linewidth=0.9)
    ax.set_ylim(0, 1.02)
    value_grid(ax, spec, axis="both")
    decorate(ax, spec, f"Cumulative distribution of {x}", x, "Cumulative proportion")
    if hue:
        add_legend(ax, spec, title=hue)
    footnote(fig, spec, "Each step is one observation. Dotted line: median (half the observations lie below).")
    fig.tight_layout(rect=(0, footnote_space(fig), 1, 1))
    return fig


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------
def mean_ci(df, spec):
    x = require(spec.x, "Choose the grouping variable (X axis).")
    y = require(spec.y, "Choose the numeric variable (Y axis).")
    hue = spec.hue if spec.hue and spec.hue != x else None
    frame = clean_frame(df, numeric=[y], categorical=[x, hue])
    groups = ordered_levels(frame[x], x, 30, spec)
    hue_levels = ordered_levels(frame[hue], hue, 8, spec) if hue else [None]
    rng = np.random.default_rng(0)
    scale = spec.font_scale
    step = 0.7 / len(hue_levels)
    offsets = [(k - (len(hue_levels) - 1) / 2) * step for k in range(len(hue_levels))]

    fig, ax = plt.subplots(figsize=spec.figsize)
    labelled = set()
    tops = [float("nan")] * len(groups)
    for hi, level in enumerate(hue_levels):
        for gi, group in enumerate(groups):
            mask = frame[x] == group
            if level is not None:
                mask &= frame[hue] == level
            values = frame.loc[mask, y].to_numpy(dtype=float)
            if values.size == 0:
                continue
            color = spec.color(hi) if hue else spec.color(gi)
            position = gi + offsets[hi]
            ax.scatter(position + rng.uniform(-0.35, 0.35, values.size) * step, values, s=10 * scale ** 2,
                       color=color, alpha=0.3, edgecolors="none", zorder=2)
            mean, low, high = mean_confidence_interval(values)
            if not hue:
                tops[gi] = max(values.max(), mean if np.isnan(high) else high)
            error = None if np.isnan(low) else np.array([[mean - low], [high - mean]])
            label = None
            if hue and level not in labelled:
                label = str(level)
                labelled.add(level)
            ax.errorbar([position], [mean], yerr=error, fmt="D", color=color, markersize=6 * scale,
                        markeredgecolor="white", markeredgewidth=0.8, elinewidth=1.6 * scale,
                        capsize=4 * scale, zorder=4, label=label)

    ax.set_xticks(range(len(groups)), labels=groups)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    value_grid(ax, spec)
    if not hue:
        shown =[(i, g, t) for i, (g, t) in enumerate(zip(groups, tops)) if np.isfinite(t)]
        annotate_groups(ax, spec, [s[0] for s in shown], [s[1] for s in shown], [s[2] for s in shown])
    decorate(ax, spec, f"Mean {y} by {x} with 95% confidence intervals", x, y)
    rotate_long_labels(ax)
    if hue:
        add_legend(ax, spec, title=hue)
    footnote(fig, spec, "Diamonds: means. Bars: 95% confidence intervals (t distribution). Faint points: observations.")
    fig.tight_layout(rect=(0, footnote_space(fig), 1, 1))
    return fig


def lollipop(df, spec):
    x = require(spec.x, "Choose the category variable.")
    y = require(spec.y, "Choose the numeric variable.")
    frame = clean_frame(df, numeric=[y], categorical=[x])
    ordered_levels(frame[x], x, 50, spec)
    means = frame.groupby(x)[y].mean().sort_values()
    positions = np.arange(means.size)
    scale = spec.font_scale

    values = means.to_numpy()
    # When every mean is far from zero, stems starting at zero hide the differences between groups,
    # so the axis then starts just below the smallest mean (and the footnote says so).
    truncated = values.min() > 0 and values.min() > 0.5 * values.max()
    if truncated:
        data_span = (values.max() - values.min()) or abs(values.max()) * 0.1 or 1.0
        low = values.min() - 0.25 * data_span
    else:
        low = min(0.0, values.min())
    high = max(0.0, values.max())
    span = (high - low) or 1.0
    baseline = low if truncated else 0.0

    fig, ax = plt.subplots(figsize=(spec.figsize[0], max(spec.figsize[1], 0.32 * means.size + 1.2)))
    ax.hlines(positions, baseline, values, color=COLORS["axis"], linewidth=1.5 * scale, zorder=1)
    ax.scatter(values, positions, s=70 * scale ** 2, color=spec.color(0), edgecolors="white", zorder=3)
    for position, value in zip(positions, values):
        ax.text(value, position, f"  {format_value(value)}  ", va="center", ha="left" if value >= 0 else "right",
                fontsize=spec.tick_size, color=spec.text_color)
    ax.set_xlim(low - (0.15 * span if low < 0 and not truncated else 0), high + 0.15 * span)
    ax.set_yticks(positions, labels=[str(v) for v in means.index])
    if not truncated:
        ax.axvline(0, color=COLORS["axis"], linewidth=0.8)
    value_grid(ax, spec, axis="x")
    decorate(ax, spec, f"Mean {y} by {x}", f"Mean {y}", x)
    if truncated:
        footnote(fig, spec, f"The axis starts at {format_value(low)}, not zero, so that differences between groups are visible.")
    fig.tight_layout(rect=(0, footnote_space(fig), 1, 1))
    return fig


def dumbbell(df, spec):
    x = require(spec.x, "Choose the category variable (one row per category).")
    y = require(spec.y, "Choose the numeric variable.")
    hue = require(spec.hue, "Choose a grouping variable with exactly two levels (for example Control / Treated).")
    if hue == x:
        raise bad_request("The two-level grouping variable must differ from the category variable.")
    frame = clean_frame(df, numeric=[y], categorical=[x, hue])
    levels = sorted(pd.unique(frame[hue]), key=natural_key)
    if len(levels) != 2:
        raise bad_request(f"'{hue}' must have exactly 2 levels for a dumbbell chart (it has {len(levels)}).")
    ordered_levels(frame[x], x, 50, spec)
    first, second = levels
    means = frame.groupby([x, hue])[y].mean().unstack(hue).reindex(columns=levels).dropna()
    if means.empty:
        raise bad_request(f"No category of '{x}' has observations at both levels of '{hue}'.")
    difference = means[second] - means[first]
    means = means.loc[difference.sort_values().index]
    difference = difference.loc[means.index]
    positions = np.arange(len(means))
    scale = spec.font_scale

    fig, ax = plt.subplots(figsize=(spec.figsize[0], max(spec.figsize[1], 0.34 * len(means) + 1.4)))
    ax.hlines(positions, means[first], means[second], color="#CBD5E1", linewidth=3.2 * scale, zorder=1)
    ax.scatter(means[first], positions, s=65 * scale ** 2, color=spec.color(0), edgecolors="white", zorder=3, label=str(first))
    ax.scatter(means[second], positions, s=65 * scale ** 2, color=spec.color(1), edgecolors="white", zorder=3, label=str(second))
    right_edge = means.max(axis=1)
    for position, edge, change in zip(positions, right_edge, difference):
        ax.text(edge, position, f"  {change:+.2f}", va="center", ha="left", fontsize=spec.tick_size * 0.9, color=COLORS["muted"])
    span = (means.to_numpy().max() - means.to_numpy().min()) or 1.0
    ax.set_xlim(means.to_numpy().min() - 0.05 * span, means.to_numpy().max() + 0.18 * span)
    ax.set_yticks(positions, labels=[str(v) for v in means.index])
    value_grid(ax, spec, axis="x")
    decorate(ax, spec, f"{y}: {first} vs {second} by {x}", f"Mean {y}", x)
    add_legend(ax, spec, title=hue)
    footnote(fig, spec, f"Labels: difference in means ({second} − {first}). Rows are sorted by the difference.")
    fig.tight_layout(rect=(0, footnote_space(fig), 1, 1))
    return fig


def paired(df, spec):
    before = require(spec.x, "Choose the first measurement (X variable, for example Before).")
    after = require(spec.y, "Choose the second measurement (Y variable, for example After).")
    if before == after:
        raise bad_request("Choose two different columns for the paired measurements.")
    hue = spec.hue if spec.hue and spec.hue not in (before, after) else None
    frame = clean_frame(df, numeric=[before, after], categorical=[hue])
    n = len(frame)
    first = frame[before].to_numpy(dtype=float)
    second = frame[after].to_numpy(dtype=float)
    change = second - first
    scale = spec.font_scale

    if hue:
        levels = ordered_levels(frame[hue], hue, 8, spec)
        level_color = {level: spec.color(i) for i, level in enumerate(levels)}
        row_colors = [level_color[v] for v in frame[hue]]
        handles = [Line2D([0], [0], color=level_color[level], marker="o", label=str(level)) for level in levels]
        legend_title = hue
    else:
        row_colors = [spec.color(0) if d > 0 else spec.color(1) if d < 0 else COLORS["axis"] for d in change]
        handles = [Line2D([0], [0], color=spec.color(0), marker="o", label="Increase"),
                   Line2D([0], [0], color=spec.color(1), marker="o", label="Decrease")]
        legend_title = None
    alpha = 0.6 if n <= 50 else max(0.08, 30 / n)

    fig, ax = plt.subplots(figsize=spec.figsize)
    ax.add_collection(LineCollection([[(0, a), (1, b)] for a, b in zip(first, second)], colors=row_colors,
                                     linewidths=0.9 * scale, alpha=alpha, zorder=2))
    ax.scatter(np.zeros(n), first, c=row_colors, s=12 * scale ** 2, alpha=min(1.0, alpha + 0.2), edgecolors="none", zorder=3)
    ax.scatter(np.ones(n), second, c=row_colors, s=12 * scale ** 2, alpha=min(1.0, alpha + 0.2), edgecolors="none", zorder=3)

    summary = [mean_confidence_interval(first), mean_confidence_interval(second)]
    for position, (mean, low, high) in zip((-0.2, 1.2), summary):
        error = None if np.isnan(low) else np.array([[mean - low], [high - mean]])
        ax.errorbar([position], [mean], yerr=error, fmt="D", color=COLORS["text"], markersize=7 * scale,
                    capsize=4 * scale, elinewidth=1.6 * scale, zorder=5)
    ax.plot([-0.2, 1.2], [summary[0][0], summary[1][0]], color=COLORS["text"], linewidth=2 * scale, zorder=4)

    ax.set_xticks([0, 1], labels=[before, after])
    ax.set_xlim(-0.5, 1.5)
    value_grid(ax, spec)
    decorate(ax, spec, f"Paired change from {before} to {after}", "", "Value")
    add_legend(ax, spec, title=legend_title, handles=handles)
    note = f"n = {n} pairs: {(change > 0).sum()} increased, {(change < 0).sum()} decreased; mean change {change.mean():+.3g}"
    if n >= 2 and np.std(change) > 0:
        note += f" (paired t-test {format_p(stats.ttest_rel(second, first).pvalue)})"
    footnote(fig, spec, note + ". Diamonds: means ± 95% CI.")
    fig.tight_layout(rect=(0, footnote_space(fig), 1, 1))
    return fig


def mean_heatmap(df, spec):
    columns_factor = require(spec.x, "Choose the column factor (X axis).")
    rows_factor = require(spec.hue, "Choose the row factor (grouping variable).")
    y = require(spec.y, "Choose the numeric variable whose means are shown.")
    if columns_factor == rows_factor:
        raise bad_request("The row and column factors must be different variables.")
    frame = clean_frame(df, numeric=[y], categorical=[columns_factor, rows_factor])
    column_levels = ordered_levels(frame[columns_factor], columns_factor, 30, spec)
    row_levels = ordered_levels(frame[rows_factor], rows_factor, 30, spec)
    table = frame.pivot_table(index=rows_factor, columns=columns_factor, values=y, aggfunc="mean") \
        .reindex(index=row_levels, columns=column_levels)
    labels = np.array([["" if np.isnan(v) else format_value(v) for v in row] for row in table.to_numpy(dtype=float)])

    width = max(spec.figsize[0], 0.75 * len(column_levels) + 2.8)
    height = max(spec.figsize[1], 0.45 * len(row_levels) + 1.8)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(table, annot=labels, fmt="", cmap=spec.cmap, linewidths=0.8, linecolor="white",
                cbar_kws={"label": f"Mean {y}", "shrink": 0.8}, annot_kws={"size": spec.tick_size}, ax=ax)
    ax.grid(False)
    plt.setp(ax.get_yticklabels(), rotation=0)
    decorate(ax, spec, f"Mean {y} by {rows_factor} × {columns_factor}", columns_factor, rows_factor)
    rotate_long_labels(ax)
    if table.isna().to_numpy().any():
        footnote(fig, spec, "Empty cells: no observations for that combination.")
        fig.tight_layout(rect=(0, footnote_space(fig), 1, 1))
    else:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------
def draw_marginal(axis, values, color, vertical):
    values = np.asarray(values, dtype=float)
    axis.hist(values, bins="auto", density=True, color=color, alpha=0.3, linewidth=0,
              orientation="vertical" if vertical else "horizontal")
    if values.size > 2 and values.std() > 0:
        grid = np.linspace(values.min(), values.max(), 200)
        density = stats.gaussian_kde(values)(grid)
        if vertical:
            axis.plot(grid, density, color=color, linewidth=1.2)
        else:
            axis.plot(density, grid, color=color, linewidth=1.2)


def joint(df, spec):
    x = require(spec.x, "Choose the numeric X variable.")
    y = require(spec.y, "Choose the numeric Y variable.")
    hue = spec.hue if spec.hue and spec.hue not in (x, y) else None
    frame = clean_frame(df, numeric=[x, y], categorical=[hue])
    levels = ordered_levels(frame[hue], hue, 8, spec) if hue else [None]
    scale = spec.font_scale

    side = spec.figsize[0]
    fig = plt.figure(figsize=(side, side * 0.92), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4))
    ax = fig.add_subplot(grid[1, 0])
    top = fig.add_subplot(grid[0, 0], sharex=ax)
    right = fig.add_subplot(grid[1, 1], sharey=ax)

    for i, level in enumerate(levels):
        subset = frame if level is None else frame[frame[hue] == level]
        xs = subset[x].to_numpy(dtype=float)
        ys = subset[y].to_numpy(dtype=float)
        color = spec.color(i)
        ax.scatter(xs, ys, s=18 * scale ** 2, color=color, alpha=0.7, edgecolors="white", linewidths=0.3,
                   label=None if level is None else str(level), zorder=3)
        draw_marginal(top, xs, color, vertical=True)
        draw_marginal(right, ys, color, vertical=False)
        if spec.fit_reg and xs.size >= 3 and xs.std() > 0:
            line_grid = np.linspace(xs.min(), xs.max(), 100)
            fitted, low, high = regression_band(xs, ys, line_grid)
            line_color = color if hue else COLORS["text"]
            ax.plot(line_grid, fitted, color=line_color, linewidth=1.5 * scale, zorder=4)
            ax.fill_between(line_grid, low, high, color=line_color, alpha=0.12, linewidth=0, zorder=2)

    for axis in (top, right):
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(False)
    top.tick_params(labelbottom=False, bottom=False, left=False, labelleft=False)
    right.tick_params(labelleft=False, left=False, bottom=False, labelbottom=False)
    top.spines["bottom"].set_visible(True)
    right.spines["left"].set_visible(True)

    value_grid(ax, spec, axis="both")
    xs_all, ys_all = frame[x].to_numpy(dtype=float), frame[y].to_numpy(dtype=float)
    if len(frame) >= 3 and xs_all.std() > 0 and ys_all.std() > 0:
        r, p = stats.pearsonr(xs_all, ys_all)
        ax.text(0.02, 0.98, f"r = {r:.2f}, {format_p(p)}, n = {len(frame)}", transform=ax.transAxes, va="top",
                ha="left", fontsize=spec.tick_size, color=spec.text_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.85), zorder=5)
    decorate(ax, spec, f"{y} vs {x} with marginal distributions", x, y, figure_title=True)
    if hue:
        add_legend(ax, spec, title=hue, loc="lower right" if spec.legend_loc == "best" else spec.legend_loc)
    return fig


def density_2d(df, spec):
    x = require(spec.x, "Choose the numeric X variable.")
    y = require(spec.y, "Choose the numeric Y variable.")
    frame = clean_frame(df, numeric=[x, y])
    if len(frame) < 10:
        raise bad_request("A 2-D density needs at least 10 complete observations.")

    fig, ax = plt.subplots(figsize=spec.figsize)
    try:
        sns.kdeplot(data=frame, x=x, y=y, fill=True, levels=10, thresh=0.05, cmap=spec.cmap, alpha=0.85, ax=ax)
        sns.kdeplot(data=frame, x=x, y=y, fill=False, levels=10, thresh=0.05, color="white", linewidths=0.4, ax=ax)
    except (np.linalg.LinAlgError, ValueError):
        plt.close(fig)
        raise bad_request("The 2-D density cannot be estimated; the variables may be constant or perfectly related.")
    ax.scatter(frame[x], frame[y], s=6 * spec.font_scale ** 2, color=COLORS["text"], alpha=0.35, edgecolors="none", zorder=3)
    ax.grid(False)
    decorate(ax, spec, f"Two-dimensional density of {y} and {x}", x, y)
    footnote(fig, spec, "Shaded contours mark where observations are most concentrated; dots are observations.")
    fig.tight_layout(rect=(0, footnote_space(fig), 1, 1))
    return fig


def hexbin(df, spec):
    x = require(spec.x, "Choose the numeric X variable.")
    y = require(spec.y, "Choose the numeric Y variable.")
    frame = clean_frame(df, numeric=[x, y])
    xs, ys = frame[x].to_numpy(dtype=float), frame[y].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=spec.figsize)
    gridsize = int(np.clip(np.sqrt(len(frame)) / 1.5, 8, 60))
    bins = ax.hexbin(xs, ys, gridsize=gridsize, cmap=spec.cmap, mincnt=1, linewidths=0.2, edgecolors="white")
    colorbar = fig.colorbar(bins, ax=ax)
    colorbar.set_label("Observations per hexagon")
    colorbar.outline.set_visible(False)
    if spec.fit_reg and xs.size >= 3 and xs.std() > 0:
        line_grid = np.linspace(xs.min(), xs.max(), 100)
        fitted, _, _ = regression_band(xs, ys, line_grid)
        ax.plot(line_grid, fitted, color=COLORS["text"], linewidth=1.5 * spec.font_scale)
    ax.grid(False)
    decorate(ax, spec, f"{y} vs {x} (hexagonal bins)", x, y)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------
def mosaic(df, spec):
    first = require(spec.x, "Choose the first categorical variable (X axis).")
    second = require(spec.hue, "Choose the second categorical variable (grouping variable).")
    if first == second:
        raise bad_request("Choose two different categorical variables.")
    frame = clean_frame(df, categorical=[first, second])
    column_levels = ordered_levels(frame[first], first, 12, spec)
    row_levels = ordered_levels(frame[second], second, 12, spec)
    if len(column_levels) < 2 or len(row_levels) < 2:
        raise bad_request("Both variables need at least 2 categories for a mosaic plot.")
    observed = pd.crosstab(frame[first], frame[second]).reindex(index=column_levels, columns=row_levels, fill_value=0) \
        .to_numpy(dtype=float)
    residuals = standardized_residuals(observed)
    total = observed.sum()
    norm = TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4)
    cmap = matplotlib.colormaps["RdBu_r"]
    gap = 0.012

    fig, ax = plt.subplots(figsize=spec.figsize)
    usable_width = 1 - gap * (len(column_levels) - 1)
    usable_height = 1 - gap * (len(row_levels) - 1)
    x0 = 0.0
    centres = []
    for i in range(len(column_levels)):
        width = observed[i].sum() / total * usable_width
        column_total = observed[i].sum()
        y0 = 0.0
        for j in range(len(row_levels)):
            height = observed[i, j] / column_total * usable_height if column_total else 0.0
            face = cmap(norm(np.clip(residuals[i, j], -4, 4)))
            ax.add_patch(Rectangle((x0, y0), width, height, facecolor=face, edgecolor="white", linewidth=0.6))
            if observed[i, j] > 0 and width * height > 0.01:
                ax.text(x0 + width / 2, y0 + height / 2, f"{int(observed[i, j])}", ha="center", va="center",
                        fontsize=spec.tick_size * 0.9, color=readable_text(face))
            y0 += height + gap
        centres.append(x0 + width / 2)
        x0 += width + gap

    row_share = observed.sum(axis=0) / total * usable_height
    row_centres = np.cumsum(row_share) - row_share / 2 + gap * np.arange(len(row_levels))
    ax.set_xlim(0, x0 - gap)
    ax.set_ylim(0, 1)
    ax.set_xticks(centres, labels=column_levels)
    ax.set_yticks(row_centres, labels=row_levels)
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("Standardized residual")
    colorbar.outline.set_visible(False)

    decorate(ax, spec, f"Mosaic plot: {second} within {first}", first, second)
    rotate_long_labels(ax)
    note = (f"Width: share of each {first}; height: share of each {second} within it. "
            "Colour: standardized residual (red = more than expected under independence, blue = fewer).")
    if (observed.sum(axis=0) > 0).all() and (observed.sum(axis=1) > 0).all():
        chi2, p, dof, _ = stats.chi2_contingency(observed, correction=False)
        note += f" χ² = {chi2:.2f}, df = {dof}, {format_p(p)}."
    footnote(fig, spec, note)
    fig.tight_layout(rect=(0, footnote_space(fig), 1, 1))
    return fig


def radar(df, spec):
    variables = list(dict.fromkeys(v for v in spec.y_vars if v))
    if len(variables) < 3:
        raise bad_request("Select at least 3 numeric variables (Y) for a radar chart.")
    if len(variables) > 12:
        raise bad_request("A radar chart shows at most 12 variables.")
    group = require(spec.hue, "Choose the grouping variable (one outline per group).")
    if group in variables:
        raise bad_request("The grouping variable cannot also be one of the plotted variables.")
    frame = clean_frame(df, numeric=variables, categorical=[group])
    groups = ordered_levels(frame[group], group, 8, spec)
    means = frame.groupby(group)[variables].mean().reindex(groups)
    low, high = means.min(), means.max()
    scaled = ((means - low) / (high - low).replace(0, np.nan)).fillna(0.5)
    angles = np.linspace(0, 2 * np.pi, len(variables), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]
    scale = spec.font_scale

    side = spec.figsize[0]
    fig, ax = plt.subplots(figsize=(side, side * 0.85), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    for i, level in enumerate(groups):
        values = scaled.loc[level].to_numpy(dtype=float)
        closed = np.r_[values, values[0]]
        color = spec.color(i)
        ax.plot(closed_angles, closed, color=color, linewidth=1.8 * scale, label=str(level))
        ax.fill(closed_angles, closed, color=color, alpha=0.12)
        ax.scatter(angles, values, s=16 * scale ** 2, color=color, zorder=3)
    ax.set_xticks(angles, labels=variables)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0], labels=[])
    ax.grid(True, color=COLORS["grid"])
    ax.spines["polar"].set_color(COLORS["grid"])
    ax.tick_params(axis="x", pad=6 * scale)

    decorate(ax, spec, f"Profile of {group} groups across {len(variables)} variables", figure_title=True)
    for label in ax.get_xticklabels():
        label.set_fontsize(spec.label_size)
    if spec.legend_loc == "best":
        add_legend(ax, spec, title=group, loc="upper left", bbox_to_anchor=(1.05, 1.08))
    else:
        add_legend(ax, spec, title=group)
    footnote(fig, spec, "Each axis runs from the lowest group mean (centre) to the highest group mean (edge).")
    fig.tight_layout(rect=(0, footnote_space(fig), 1, 0.95))
    return fig


# ---------------------------------------------------------------------------
# Estimation plots (difference and its confidence interval on a floating axis)
# ---------------------------------------------------------------------------
def draw_difference_axis(ax, dax, spec, anchor, difference, low, high, se, dof, label):
    """Gardner-Altman difference axis: zero is aligned with `anchor` on the data axis."""
    lo, hi = ax.get_ylim()
    pad = 0.05 * (hi - lo)
    lo, hi = min(lo, anchor + low - pad), max(hi, anchor + high + pad)
    ax.set_ylim(lo, hi)
    dax.set_ylim(lo - anchor, hi - anchor)
    if se > 0:
        grid = np.linspace(difference - 4 * se, difference + 4 * se, 200)
        density = stats.t.pdf((grid - difference) / se, dof)
        dax.fill_betweenx(grid, 0.05, 0.05 + density / density.max() * 0.4, color=spec.color(0), alpha=0.35, linewidth=0)
    dax.errorbar([0], [difference], yerr=np.array([[difference - low], [high - difference]]), fmt="o",
                 color=COLORS["text"], markersize=7 * spec.font_scale, elinewidth=2 * spec.font_scale, capsize=0, zorder=4)
    dax.axhline(0, color=COLORS["axis"], linestyle="--", linewidth=1.0)
    dax.set_xlim(-0.4, 0.6)
    dax.set_xticks([0], labels=[label])
    dax.grid(False)
    dax.spines["left"].set_visible(False)
    dax.spines["right"].set_visible(True)
    dax.yaxis.tick_right()
    dax.yaxis.set_label_position("right")
    dax.set_ylabel("Mean difference", fontsize=spec.label_size, color=spec.text_color, family=family(spec.label_family))
    for tick in dax.get_xticklabels() + dax.get_yticklabels():
        tick.set_fontsize(spec.tick_size)
        tick.set_family(family(spec.tick_family))
        tick.set_color(spec.text_color)


def estimation(df, spec):
    x = require(spec.x, "Choose the grouping variable (exactly 2 groups).")
    y = require(spec.y, "Choose the numeric variable.")
    frame = clean_frame(df, numeric=[y], categorical=[x])
    groups = sorted(pd.unique(frame[x]), key=natural_key) if not (spec.extra.get("orders") or {}).get(x) \
        else [g for g in spec.extra["orders"][x] if g in set(frame[x])]
    if len(groups) != 2:
        raise bad_request(f"An estimation plot compares exactly 2 groups ('{x}' has {len(groups)}).")
    first, second = groups
    a = frame.loc[frame[x] == first, y].to_numpy(dtype=float)
    b = frame.loc[frame[x] == second, y].to_numpy(dtype=float)
    if a.size < 2 or b.size < 2:
        raise bad_request("Each group needs at least 2 observations for an estimation plot.")
    equal_var = bool(spec.extra.get("equal_var", False))
    difference, low, high, se, dof = difference_interval(a, b, equal_var)
    rng = np.random.default_rng(0)
    scale = spec.font_scale

    fig, (ax, dax) = plt.subplots(1, 2, figsize=spec.figsize, gridspec_kw={"width_ratios": (2.4, 1)})
    for i, values in enumerate((a, b)):
        ax.scatter(i + rng.uniform(-0.15, 0.15, values.size), values, s=16 * scale ** 2, color=spec.color(i),
                   alpha=0.7, edgecolors="none", zorder=3)
        ax.errorbar([i + 0.3], [values.mean()], yerr=[values.std(ddof=1)], fmt="o", color=COLORS["text"],
                    markersize=5 * scale, elinewidth=1.8 * scale, capsize=0, zorder=4)
    ax.set_xticks([0, 1], labels=[str(first), str(second)])
    ax.set_xlim(-0.5, 1.6)
    value_grid(ax, spec)
    draw_difference_axis(ax, dax, spec, b.mean(), difference, low, high, se, dof, f"{first} − {second}")
    decorate(ax, spec, f"Estimation plot: {y} by {x}", "", y, figure_title=True)
    method = "Student" if equal_var else "Welch"
    footnote(fig, spec, f"Left: observations with mean ± SD. Right: difference in means ({first} − {second}) = "
                        f"{format_value(difference)}, 95% CI [{format_value(low)}, {format_value(high)}] "
                        f"({method} t, df = {dof:.1f}); the curve is its sampling distribution and 0 is aligned "
                        f"with the mean of {second}.")
    fig.tight_layout(rect=(0, footnote_space(fig), 1, 1))
    return fig


def paired_estimation(df, spec):
    before = require(spec.x, "Choose the first measurement (X variable).")
    after = require(spec.y, "Choose the second measurement (Y variable).")
    if before == after:
        raise bad_request("Choose two different columns for the paired measurements.")
    frame = clean_frame(df, numeric=[before, after])
    first = frame[before].to_numpy(dtype=float)
    second = frame[after].to_numpy(dtype=float)
    difference, low, high, se, dof = paired_difference_interval(first, second)
    n = len(frame)
    change = second - first
    scale = spec.font_scale
    alpha = 0.6 if n <= 50 else max(0.08, 30 / n)
    row_colors = [spec.color(0) if d > 0 else spec.color(1) if d < 0 else COLORS["axis"] for d in change]

    fig, (ax, dax) = plt.subplots(1, 2, figsize=spec.figsize, gridspec_kw={"width_ratios": (2.4, 1)})
    ax.add_collection(LineCollection([[(0, a), (1, b)] for a, b in zip(first, second)], colors=row_colors,
                                     linewidths=0.9 * scale, alpha=alpha, zorder=2))
    ax.scatter(np.zeros(n), first, c=row_colors, s=12 * scale ** 2, alpha=min(1.0, alpha + 0.2), edgecolors="none", zorder=3)
    ax.scatter(np.ones(n), second, c=row_colors, s=12 * scale ** 2, alpha=min(1.0, alpha + 0.2), edgecolors="none", zorder=3)
    ax.set_xticks([0, 1], labels=[before, after])
    ax.set_xlim(-0.3, 1.3)
    value_grid(ax, spec)
    draw_difference_axis(ax, dax, spec, second.mean(), difference, low, high, se, dof, f"{before} − {after}")
    decorate(ax, spec, f"Paired estimation plot: {before} and {after}", "", "Value", figure_title=True)
    add_legend(ax, spec, handles=[Line2D([0], [0], color=spec.color(0), marker="o", label="Increase"),
                                  Line2D([0], [0], color=spec.color(1), marker="o", label="Decrease")])
    footnote(fig, spec, f"Left: one line per pair (n = {n}). Right: mean paired difference ({before} − {after}) = "
                        f"{format_value(difference)}, 95% CI [{format_value(low)}, {format_value(high)}] "
                        f"(paired t, df = {dof:.0f}); 0 is aligned with the mean of {after}.")
    fig.tight_layout(rect=(0, footnote_space(fig), 1, 1))
    return fig


CHARTS = {
    "estimation": estimation,
    "paired_estimation": paired_estimation,
    "raincloud": raincloud,
    "boxswarm": box_swarm,
    "ridgeline": ridgeline,
    "ecdf": ecdf,
    "meanci": mean_ci,
    "lollipop": lollipop,
    "dumbbell": dumbbell,
    "paired": paired,
    "meanheatmap": mean_heatmap,
    "joint": joint,
    "density2d": density_2d,
    "hexbin": hexbin,
    "mosaic": mosaic,
    "radar": radar,
}


def render(plot_type, df, spec):
    return CHARTS[plot_type](df, spec)
