"""PCA charts and statistics: biplot (distance or correlation scaling, group ellipses or hulls, point labels), scree
plot with parallel analysis, correlation circle, variable contributions, loadings heatmap, varimax rotation and the
quality and contribution of observations."""

import base64

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib

matplotlib.use("Agg")  # headless rendering for the web server
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Ellipse, Polygon
from scipy.spatial import ConvexHull

from ..figure_style import COLORS, export_figure, figure_size, palette_colors
from . import chart_gallery

PARALLEL_SEED = 20260911
BIPLOT_TYPES = ("distance", "correlation")
ELLIPSES = ("none", "confidence", "data", "hull")
MAX_POINT_LABELS = 40


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def parallel_analysis(n_obs, n_vars, iterations=200, percentile=95, seed=PARALLEL_SEED):
    """Horn's parallel analysis: the given percentile of correlation-matrix eigenvalues of random normal data."""
    if n_obs * n_vars > 500_000:
        iterations = min(iterations, 50)  # keep large datasets responsive
    rng = np.random.default_rng(seed)
    eigenvalues = np.empty((iterations, n_vars))
    for i in range(iterations):
        data = rng.standard_normal((n_obs, n_vars))
        eigenvalues[i] = np.sort(np.linalg.eigvalsh(np.corrcoef(data, rowvar=False)))[::-1]
    return np.percentile(eigenvalues, percentile, axis=0), iterations


def retained_components(eigenvalues, thresholds):
    """Number of leading components whose eigenvalue exceeds the parallel-analysis threshold."""
    count = 0
    for value, threshold in zip(eigenvalues, thresholds):
        if value <= threshold:
            break
        count += 1
    return count


def variable_contributions(components):
    """Contribution (%) of each variable to each component: squared eigenvector elements × 100 (rows = components)."""
    return np.asarray(components, dtype=float) ** 2 * 100


def observation_quality(scores, eigenvalues):
    """cos² (share of an observation's squared distance from the centre on each component) and contribution (%) of
    each observation to each component: score² / ((n − 1) λ) × 100."""
    scores = np.asarray(scores, dtype=float)
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    squared = scores ** 2
    total = squared.sum(axis=1, keepdims=True)
    cos2 = np.divide(squared, total, out=np.zeros_like(squared), where=total > 0)
    denominator = (scores.shape[0] - 1) * eigenvalues[None, :]
    contributions = np.divide(100 * squared, denominator, out=np.zeros_like(squared), where=denominator > 0)
    return cos2, contributions


def varimax(loadings, normalize=True, tolerance=1e-5, max_iterations=1000):
    """Varimax rotation with Kaiser normalization, following R's stats::varimax. Returns (rotated, rotation matrix)."""
    x = np.asarray(loadings, dtype=float).copy()
    p, k = x.shape
    if k < 2:
        return x, np.eye(k)
    scale = np.ones(p)
    if normalize:
        scale = np.sqrt((x ** 2).sum(axis=1))
        scale[scale == 0] = 1.0
        x = x / scale[:, None]
    rotation = np.eye(k)
    criterion = 0.0
    for _ in range(max_iterations):
        z = x @ rotation
        u, s, vt = np.linalg.svd(x.T @ (z ** 3 - z * (z ** 2).sum(axis=0) / p))
        rotation = u @ vt
        previous, criterion = criterion, s.sum()
        if criterion < previous * (1 + tolerance):
            break
    return x @ rotation * scale[:, None], rotation


def ordered_rotation(rotated):
    """Rotated components sorted by the variance they explain, each oriented so its loadings sum to a positive value."""
    rotated = np.asarray(rotated, dtype=float)
    variance = (rotated ** 2).sum(axis=0)
    order = np.argsort(variance)[::-1]
    rotated = rotated[:, order]
    signs = np.where(rotated.sum(axis=0) < 0, -1.0, 1.0)
    return rotated * signs, variance[order]


def data_ellipse(x, y, level=0.95, of_mean=False):
    """Centre, width, height and angle (degrees) of a bivariate normal ellipse.

    of_mean=False: the region expected to hold `level` of the group's observations.
    of_mean=True:  the confidence region of the group mean (covariance divided by n).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    covariance = np.cov(x, y) / (x.size if of_mean else 1)
    values, vectors = np.linalg.eigh(covariance)
    order = values.argsort()[::-1]
    values, vectors = values[order], vectors[:, order]
    radius = np.sqrt(stats.chi2.ppf(level, 2))
    width, height = 2 * radius * np.sqrt(np.clip(values, 0, None))
    angle = float(np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0])))
    return (float(x.mean()), float(y.mean())), float(width), float(height), angle


def draw_group_ellipse(ax, x, y, color, of_mean=True):
    """95% ellipse for a group: the confidence region of its mean (default) or the region holding its observations."""
    centre, width, height, angle = data_ellipse(x, y, of_mean=of_mean)
    ax.add_patch(Ellipse(centre, width, height, angle=angle, facecolor=color, edgecolor=color, alpha=0.12, linewidth=0))
    ax.add_patch(Ellipse(centre, width, height, angle=angle, facecolor="none", edgecolor=color, linewidth=1.2, alpha=0.9))
    theta, a, b = np.radians(angle), width / 2, height / 2
    # centre and half-extents of the ellipse's bounding box, so the axes can be sized to show it whole
    return centre, float(np.hypot(a * np.cos(theta), b * np.sin(theta))), float(np.hypot(a * np.sin(theta), b * np.cos(theta)))


def draw_group_hull(ax, points, color):
    try:
        hull = ConvexHull(points)
    except Exception:  # fewer than three distinct, non-collinear points
        return
    polygon = points[hull.vertices]
    ax.add_patch(Polygon(polygon, closed=True, facecolor=color, edgecolor=color, alpha=0.12, linewidth=0))
    ax.add_patch(Polygon(polygon, closed=True, facecolor="none", edgecolor=color, linewidth=1.1, alpha=0.9))


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _png(fig):
    try:
        content, _ = export_figure(fig, "png", 150)
    finally:
        plt.close(fig)
    return base64.b64encode(content).decode("utf-8")


def biplot_chart(scores, eigenvectors, eigenvalues, correlation_loadings, names, explained_ratio, components=(0, 1),
                 biplot_type="distance", groups=None, group_name=None, labels=None, ellipse="confidence", palette=None,
                 colors=None):
    """Observations and variable arrows on two components. Returns (PNG, arrow stretch factor)."""
    spec = chart_gallery.analysis_spec(palette=palette)
    a, b = components
    scores = np.asarray(scores, dtype=float)
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    if biplot_type == "correlation":
        root = np.sqrt(np.where(eigenvalues[[a, b]] > 0, eigenvalues[[a, b]], 1.0))
        points = scores[:, [a, b]] / root  # standardized scores
        arrows = np.asarray(correlation_loadings, dtype=float)[:, [a, b]]
    else:
        points = scores[:, [a, b]]
        arrows = np.asarray(eigenvectors, dtype=float)[:, [a, b]]
    point_color, arrow_color = colors or (spec.color(0), spec.color(1))
    reach = float(np.abs(points).max()) if points.size else 1.0
    longest = float(np.linalg.norm(arrows, axis=1).max())
    stretch = 0.9 * reach / longest if longest > 0 and reach > 0 else 1.0
    tips = arrows * stretch

    width, height = figure_size()
    fig, ax = plt.subplots(figsize=(width, height * 1.08))
    extents = []
    if groups is not None:
        groups = pd.Series(list(groups))
        levels = sorted(pd.unique(groups), key=chart_gallery.natural_key)
        level_colors = dict(zip(levels, palette_colors(palette, len(levels))))
        for level in levels:
            mask = (groups == level).to_numpy()
            group = points[mask]
            color = level_colors[level]
            ax.scatter(group[:, 0], group[:, 1], s=26, color=color, alpha=0.85, edgecolors="white", linewidths=0.4,
                       zorder=3, label=str(level))
            if ellipse != "none" and len(group) >= 3 and group[:, 0].std() > 0 and group[:, 1].std() > 0:
                if ellipse == "hull":
                    draw_group_hull(ax, group, color)
                else:
                    (cx, cy), hx, hy = draw_group_ellipse(ax, group[:, 0], group[:, 1], color, of_mean=ellipse == "confidence")
                    extents.extend([(cx - hx, cy - hy), (cx + hx, cy + hy)])
    else:
        ax.scatter(points[:, 0], points[:, 1], s=26, color=point_color, alpha=0.8, edgecolors="white", linewidths=0.4, zorder=3)

    bounds = np.vstack([points, tips * 1.3, np.zeros((1, 2))] + ([np.array(extents)] if extents else []))
    xs, ys = bounds[:, 0], bounds[:, 1]
    x_pad = 0.06 * ((xs.max() - xs.min()) or 1.0)
    y_pad = 0.06 * ((ys.max() - ys.min()) or 1.0)
    ax.set_xlim(xs.min() - x_pad, xs.max() + x_pad)
    ax.set_ylim(ys.min() - y_pad, ys.max() + y_pad)
    ax.set_aspect("equal", adjustable="box")  # equal units on both axes, so distances and angles are not distorted

    for x, y in tips:
        ax.annotate("", xy=(x, y), xytext=(0, 0), zorder=4,
                    arrowprops=dict(arrowstyle="-|>", color=arrow_color, linewidth=1.3, shrinkA=0, shrinkB=0))
    for name, (x, y), (tx, ty) in zip(names, tips, chart_gallery.nudge_labels(tips * 1.08, ax)):
        ax.text(tx, ty, name, ha="left" if x >= 0 else "right", va="bottom" if y >= 0 else "top",
                fontsize=spec.tick_size, fontweight="bold", color=COLORS["text"], zorder=5)

    labelled = 0
    if labels is not None:
        labels = [str(v) for v in labels]
        shown = range(len(labels)) if len(labels) <= MAX_POINT_LABELS else np.argsort(-(points ** 2).sum(axis=1))[:12]
        for i in shown:
            if labels[i]:
                ax.annotate(labels[i], points[i], xytext=(3, 3), textcoords="offset points", fontsize=spec.tick_size * 0.78,
                            color=COLORS["muted"], zorder=5)
                labelled += 1

    ax.axhline(0, color=COLORS["grid"], linewidth=0.9, zorder=0)
    ax.axvline(0, color=COLORS["grid"], linewidth=0.9, zorder=0)
    chart_gallery.value_grid(ax, spec, axis="both")
    chart_gallery.decorate(ax, spec, f"PCA biplot (PC{a + 1} vs PC{b + 1})",
                           f"PC{a + 1} ({explained_ratio[a] * 100:.1f}% of variance)",
                           f"PC{b + 1} ({explained_ratio[b] * 100:.1f}% of variance)")
    if groups is not None:
        chart_gallery.add_legend(ax, spec, title=group_name)

    if biplot_type == "correlation":
        note = ("Correlation biplot: observation scores are standardized; angles between arrows approximate correlations "
                "between variables and arrow lengths their standard deviations.")
    else:
        note = ("Distance biplot: distances between observations approximate their multivariate distances; arrows are the "
                "eigenvectors.")
    note += f" Arrows stretched ×{stretch:.2g} to fit."
    if groups is not None and ellipse != "none":
        note += {"confidence": " Ellipses: 95% confidence regions of group means.",
                 "data": " Ellipses: regions expected to hold 95% of each group's observations.",
                 "hull": " Outlines: convex hull of each group."}[ellipse]
    if labels is not None and len(labels) > MAX_POINT_LABELS:
        note += f" The {labelled} observations furthest from the centre are labelled."
    chart_gallery.footnote(fig, spec, note)
    fig.tight_layout(rect=(0, chart_gallery.footnote_space(fig), 1, 1))
    return _png(fig), float(stretch)


def scree_chart(eigenvalues, explained_ratio, thresholds, scaled, palette=None):
    spec = chart_gallery.analysis_spec(palette=palette)
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    positions = np.arange(1, eigenvalues.size + 1)
    retained = retained_components(eigenvalues, thresholds) if thresholds is not None else None

    fig, ax = plt.subplots(figsize=figure_size())
    colors = [spec.color(0) if retained is None or i < retained else "#CBD5E1" for i in range(eigenvalues.size)]
    ax.bar(positions, eigenvalues, color=colors, width=0.6, zorder=2)
    ax.plot(positions, eigenvalues, color=COLORS["text"], marker="o", markersize=4, linewidth=1.2, zorder=3)
    if thresholds is not None:
        ax.plot(positions, thresholds, color=spec.color(1), linestyle="--", marker="s", markersize=3, linewidth=1.3,
                zorder=4, label="Parallel analysis (95th percentile, random data)")
    if scaled:
        ax.axhline(1, color=COLORS["muted"], linestyle=":", linewidth=1.1, zorder=1, label="Kaiser criterion (eigenvalue = 1)")
    ax.set_xticks(positions, labels=[f"PC{i}\n{share * 100:.1f}%" for i, share in zip(positions, explained_ratio)])
    chart_gallery.value_grid(ax, spec)
    chart_gallery.decorate(ax, spec, "Scree plot", "Component (share of variance explained)", "Eigenvalue")
    chart_gallery.add_legend(ax, spec, loc="upper right")
    note = "Bars and line: eigenvalues."
    if retained is not None:
        note += f" Parallel analysis retains {retained} component{'s' if retained != 1 else ''} (coloured bars)."
    chart_gallery.footnote(fig, spec, note)
    fig.tight_layout(rect=(0, chart_gallery.footnote_space(fig), 1, 1))
    return _png(fig)


def correlation_circle_chart(correlation_loadings, names, explained_ratio, palette=None, components=(0, 1)):
    spec = chart_gallery.analysis_spec(palette=palette)
    a, b = components
    loadings = np.asarray(correlation_loadings, dtype=float)
    r1, r2 = loadings[:, a], loadings[:, b]
    quality = np.clip(r1 ** 2 + r2 ** 2, 0, 1)
    cmap = matplotlib.colormaps["viridis"]
    norm = Normalize(vmin=0, vmax=1)

    side = figure_size()[0] * 0.9
    fig, ax = plt.subplots(figsize=(side, side * 0.85))
    ax.add_patch(Circle((0, 0), 1, facecolor="none", edgecolor=COLORS["axis"], linewidth=1.0))
    ax.axhline(0, color=COLORS["grid"], linewidth=0.9, linestyle="--", zorder=0)
    ax.axvline(0, color=COLORS["grid"], linewidth=0.9, linestyle="--", zorder=0)
    for name, x, y, q in zip(names, r1, r2, quality):
        color = cmap(norm(q))
        ax.annotate("", xy=(x, y), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.6))
        ax.text(x * 1.08, y * 1.08, name, ha="left" if x >= 0 else "right", va="bottom" if y >= 0 else "top",
                fontsize=spec.tick_size, color=COLORS["text"])
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.grid(False)
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("cos² (quality of representation)")
    colorbar.outline.set_visible(False)
    chart_gallery.decorate(ax, spec, f"Correlation circle (PC{a + 1} vs PC{b + 1})",
                           f"PC{a + 1} ({explained_ratio[a] * 100:.1f}% of variance)",
                           f"PC{b + 1} ({explained_ratio[b] * 100:.1f}% of variance)")
    chart_gallery.footnote(fig, spec, f"Arrows show each variable's correlation with PC{a + 1} and PC{b + 1}. Arrows close to "
                                      "the circle are well represented; small angles between arrows mean positively "
                                      "correlated variables.")
    fig.tight_layout(rect=(0, chart_gallery.footnote_space(fig), 1, 1))
    return _png(fig)


def contributions_chart(components, names, explained_ratio, palette=None, shown_components=(0, 1)):
    spec = chart_gallery.analysis_spec(palette=palette)
    contributions = variable_contributions(components)
    shown = [k for k in shown_components if k < contributions.shape[0]]
    p = len(names)
    width, height = figure_size()
    fig, axes = plt.subplots(1, len(shown), figsize=(width, max(height, 0.32 * p + 1.8)), squeeze=False)
    average = 100 / p
    for position, (k, ax) in enumerate(zip(shown, axes[0])):
        values = contributions[k]
        order = np.argsort(values)
        ax.barh([names[i] for i in order], values[order], zorder=2,
                color=[spec.color(position) if values[i] > average else "#CBD5E1" for i in order])
        ax.axvline(average, color=COLORS["text"], linestyle="--", linewidth=1.0, zorder=3)
        chart_gallery.value_grid(ax, spec, axis="x")
        chart_gallery.decorate(ax, spec, f"PC{k + 1} ({explained_ratio[k] * 100:.1f}% of variance)", "Contribution (%)", "")
        ax.title.set_fontsize(spec.label_size)
    fig.suptitle("Variable contributions to the components", x=0.01, ha="left", fontweight="bold",
                 fontsize=spec.title_size, color=COLORS["text"])
    chart_gallery.footnote(fig, spec, f"Dashed line: the average contribution (100 / {p} variables = {average:.1f}%). "
                                      "Coloured bars contribute more than average.")
    fig.tight_layout(rect=(0, chart_gallery.footnote_space(fig), 1, 0.95))
    return _png(fig)


def loadings_heatmap_chart(correlation_loadings, names, explained_ratio, count, palette=None, title=None, labels=None,
                           note=None):
    spec = chart_gallery.analysis_spec(palette=palette)
    values = np.asarray(correlation_loadings, dtype=float)[:, :count]
    p = len(names)
    width, height = figure_size()
    fig, ax = plt.subplots(figsize=(max(width * 0.75, 0.85 * count + 3.0), max(height * 0.75, 0.36 * p + 1.9)))
    image = ax.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(p):
        for k in range(count):
            value = values[i, k]
            text = f"{value:.2f}"
            ax.text(k, i, "0.00" if text == "-0.00" else text, ha="center", va="center", fontsize=spec.tick_size * 0.85,
                    fontweight="bold" if abs(value) >= 0.5 else "normal",
                    color=chart_gallery.readable_text(image.cmap(image.norm(value))))
    ax.set_xticks(range(count), labels=labels or [f"PC{k + 1}\n{explained_ratio[k] * 100:.1f}%" for k in range(count)])
    ax.set_yticks(range(p), labels=names)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Correlation")
    colorbar.outline.set_visible(False)
    chart_gallery.decorate(ax, spec, title or "Correlation loadings", "", "")
    chart_gallery.footnote(fig, spec, note or "Each cell: the correlation between a variable and a component. Bold: |r| ≥ 0.5.")
    fig.tight_layout(rect=(0, chart_gallery.footnote_space(fig), 1, 1))
    return _png(fig)
