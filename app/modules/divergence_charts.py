"""Charts for genetic divergence: Tocher cluster diagram, D² heatmap, UPGMA dendrogram, character contributions,
cluster means and canonical variates."""

import base64

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless rendering for the web server
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch
from scipy.cluster import hierarchy

from ..figure_style import COLORS, export_figure, figure_size, palette_colors
from . import chart_gallery
from .pca_charts import draw_group_hull

MAX_LEGEND_CLUSTERS = 12


def roman(number):
    numerals = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
                (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    text = ""
    for value, symbol in numerals:
        while number >= value:
            text += symbol
            number -= value
    return text


def _png(fig):
    try:
        content, _ = export_figure(fig, "png", 150)
    finally:
        plt.close(fig)
    return base64.b64encode(content).decode("utf-8")


def _cluster_legend(ax, spec, names, sizes, colors, **kwargs):
    if len(names) > MAX_LEGEND_CLUSTERS:
        return
    handles = [Patch(facecolor=colors[c % len(colors)], label=f"{names[c]} ({sizes[c]})") for c in range(len(names))]
    chart_gallery.add_legend(ax, spec, handles=handles, **kwargs)


def cluster_positions(inter_distances):
    """Two-dimensional positions whose distances approximate the given matrix (classical scaling)."""
    D = np.asarray(inter_distances, dtype=float)
    K = D.shape[0]
    if K == 1:
        return np.zeros((1, 2))
    if K == 2:
        return np.array([[-D[0, 1] / 2, 0.0], [D[0, 1] / 2, 0.0]])
    centring = np.eye(K) - 1.0 / K
    values, vectors = np.linalg.eigh(-0.5 * centring @ (D ** 2) @ centring)
    order = np.argsort(values)[::-1][:2]
    return vectors[:, order] * np.sqrt(np.clip(values[order], 0, None))


def tocher_diagram_chart(names, sizes, distance_matrix_d2, palette=None):
    spec = chart_gallery.analysis_spec(palette=palette)
    d = np.sqrt(np.clip(np.asarray(distance_matrix_d2, dtype=float), 0, None))
    intra = np.diag(d).copy()
    inter = d.copy()
    np.fill_diagonal(inter, 0.0)
    K = len(names)
    centres = cluster_positions(inter)
    colors = palette_colors(palette, K)
    gaps = [float(np.linalg.norm(centres[a] - centres[b])) for a in range(K) for b in range(a + 1, K)]
    unit = max(float(np.percentile(gaps, 25)), 1e-9) if gaps else 1.0
    largest = intra.max()
    radii = np.array([unit * (0.14 + 0.24 * (v / largest if largest > 0 else 0)) for v in intra])

    side = figure_size()[0]
    fig, ax = plt.subplots(figsize=(side, side * 0.9))
    pairs = [(a, b) for a in range(K) for b in range(a + 1, K)]
    if K > 4:  # keep the diagram readable: link each cluster to its two nearest clusters
        kept = set()
        for a in range(K):
            for b in np.argsort(inter[a])[1:3]:
                kept.add(tuple(sorted((a, int(b)))))
        pairs = sorted(kept)
    for a, b in pairs:
        (x1, y1), (x2, y2) = centres[a], centres[b]
        ax.plot([x1, x2], [y1, y2], color=COLORS["muted"], linewidth=0.9, zorder=1)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"{inter[a, b]:.2f}", ha="center", va="center", fontsize=spec.tick_size * 0.78,
                color=COLORS["text"], zorder=4, bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.9))
    for c in range(K):
        color = colors[c % len(colors)]
        ax.add_patch(Circle(centres[c], radii[c], facecolor=color, edgecolor="none", alpha=0.25, zorder=2))
        ax.add_patch(Circle(centres[c], radii[c], facecolor="none", edgecolor=color, linewidth=1.4, zorder=3))
        label = f"{names[c].replace('Cluster ', '')} ({sizes[c]})" + (f"\n{intra[c]:.2f}" if sizes[c] > 1 else "")
        ax.text(centres[c][0], centres[c][1], label, ha="center", va="center", fontsize=spec.tick_size * 0.85,
                fontweight="bold", color=COLORS["text"], zorder=5)
    low = centres.min(axis=0) - radii.max() * 1.4
    high = centres.max(axis=0) + radii.max() * 1.4
    centre = (low + high) / 2
    half = max(high - low) / 2
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_title("Tocher cluster diagram", loc="left", fontweight="bold", fontsize=spec.title_size, color=COLORS["text"])
    note = ("Circles are clusters: Roman numeral, number of genotypes and, inside, the intra-cluster distance D = √D². Lines "
            "show inter-cluster distances D. Positions approximate the inter-cluster distances.")
    if K > 4:
        note += " Each cluster is linked to its two nearest clusters."
    chart_gallery.footnote(fig, spec, note)
    fig.tight_layout(rect=(0, chart_gallery.footnote_space(fig), 1, 1))
    return _png(fig)


def d2_heatmap_chart(d2, genotypes, clusters, palette=None):
    spec = chart_gallery.analysis_spec(palette=palette)
    order = [g for members in clusters for g in members]
    matrix = np.asarray(d2, dtype=float)[np.ix_(order, order)]
    G = len(order)
    width, _ = figure_size()
    side = max(width, min(16.0, 0.18 * G + 3.0))
    fig, ax = plt.subplots(figsize=(side, side * 0.85))
    image = ax.imshow(matrix, cmap="viridis", aspect="equal")
    if G <= 15:
        for i in range(G):
            for j in range(G):
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=spec.tick_size * 0.7,
                        color=chart_gallery.readable_text(image.cmap(image.norm(matrix[i, j]))))
    for boundary in np.cumsum([len(m) for m in clusters])[:-1] - 0.5:
        ax.axhline(boundary, color="white", linewidth=1.6)
        ax.axvline(boundary, color="white", linewidth=1.6)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    labels = [str(genotypes[g]) for g in order]
    if G <= 80:
        ax.set_xticks(range(G), labels=labels)
        ax.set_yticks(range(G), labels=labels)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("D²")
    colorbar.outline.set_visible(False)
    chart_gallery.decorate(ax, spec, "D² between genotypes", "", "")
    size = min(spec.tick_size, max(4.5, 400 / max(G, 1)))
    for label in ax.get_xticklabels():
        label.set_fontsize(size)
        label.set_rotation(90)
    for label in ax.get_yticklabels():
        label.set_fontsize(size)
    chart_gallery.footnote(fig, spec, "Genotypes are ordered by Tocher cluster; white lines separate the clusters. Small "
                                      "values inside blocks and large values between them mean well-separated clusters.")
    fig.tight_layout(rect=(0, chart_gallery.footnote_space(fig), 1, 1))
    return _png(fig)


def dendrogram_chart(linkage_matrix, genotypes, cluster_of, names, sizes, cophenetic, palette=None):
    spec = chart_gallery.analysis_spec(palette=palette)
    G = len(genotypes)
    horizontal = G > 40
    colors = palette_colors(palette, len(names))
    width, height = figure_size()
    size = (width, min(40.0, max(height, 0.18 * G + 1.5))) if horizontal else (max(width, min(18.0, 0.25 * G + 2.5)), height)
    fig, ax = plt.subplots(figsize=size)
    result = hierarchy.dendrogram(linkage_matrix, ax=ax, labels=[str(g) for g in genotypes],
                                  orientation="left" if horizontal else "top", leaf_rotation=0 if horizontal else 90,
                                  link_color_func=lambda _: "#475569")
    ax.grid(False)
    chart_gallery.decorate(ax, spec, "UPGMA dendrogram of Mahalanobis distances", "D" if horizontal else "",
                           "" if horizontal else "D")
    ticks = ax.get_yticklabels() if horizontal else ax.get_xticklabels()
    font = min(spec.tick_size, max(4.5, 500 / max(G, 1)))
    for tick, leaf in zip(ticks, result["leaves"]):
        tick.set_color(colors[cluster_of[leaf] % len(colors)])
        tick.set_fontsize(font)
    _cluster_legend(ax, spec, names, sizes, colors, loc="upper right")
    chart_gallery.footnote(fig, spec, f"Average linkage (UPGMA) on D = √D². Genotype labels are coloured by Tocher cluster, so "
                                      f"the two groupings can be compared. Cophenetic correlation: {cophenetic:.3f}.")
    fig.tight_layout(rect=(0, chart_gallery.footnote_space(fig), 1, 1))
    return _png(fig)


def contributions_chart(traits, counts, percents, palette=None):
    spec = chart_gallery.analysis_spec(palette=palette)
    percents = np.asarray(percents, dtype=float)
    order = np.argsort(percents)
    width, height = figure_size()
    fig, ax = plt.subplots(figsize=(width, max(height * 0.7, 0.42 * len(traits) + 1.6)))
    ax.barh([traits[i] for i in order], percents[order], color=spec.color(0), height=0.6, zorder=2)
    for position, i in enumerate(order):
        ax.text(percents[i] + percents.max() * 0.02, position, f"{percents[i]:.1f}% ({int(counts[i])})", va="center",
                fontsize=spec.tick_size * 0.85, color=COLORS["text"])
    ax.set_xlim(0, max(percents.max() * 1.3, 1))
    chart_gallery.value_grid(ax, spec, axis="x")
    chart_gallery.decorate(ax, spec, "Contribution of each trait to divergence", "Times ranked first (% of genotype pairs)", "")
    total = int(np.sum(counts))
    chart_gallery.footnote(fig, spec, f"For each of the {total} pairs of genotypes, the trait with the largest difference on "
                                      "the uncorrelated (transformed) scale is ranked first (Singh 1981). The transformation "
                                      "follows the order in which the traits were selected.")
    fig.tight_layout(rect=(0, chart_gallery.footnote_space(fig), 1, 1))
    return _png(fig)


def cluster_means_chart(names, traits, means, palette=None):
    spec = chart_gallery.analysis_spec(palette=palette)
    M = np.asarray(means, dtype=float)
    K, p = M.shape
    sd = M.std(axis=0)
    z = np.where(sd > 0, (M - M.mean(axis=0)) / np.where(sd > 0, sd, 1.0), 0.0)
    width, height = figure_size()
    fig, ax = plt.subplots(figsize=(max(width, 1.0 * p + 3.0), max(height * 0.6, 0.45 * K + 1.9)))
    image = ax.imshow(z, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    best = M.argmax(axis=0)
    for i in range(K):
        for j in range(p):
            ax.text(j, i, chart_gallery.format_value(M[i, j]), ha="center", va="center", fontsize=spec.tick_size * 0.8,
                    fontweight="bold" if best[j] == i and K > 1 else "normal",
                    color=chart_gallery.readable_text(image.cmap(image.norm(z[i, j]))))
    ax.set_xticks(range(p), labels=traits)
    ax.set_yticks(range(K), labels=names)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("SDs from the average cluster")
    colorbar.outline.set_visible(False)
    chart_gallery.decorate(ax, spec, "Cluster means", "", "")
    chart_gallery.rotate_long_labels(ax)
    chart_gallery.footnote(fig, spec, "Values are cluster means; colours compare clusters within each trait (red above, blue "
                                      "below the average of the cluster means). Bold: the highest cluster mean for the trait.")
    fig.tight_layout(rect=(0, chart_gallery.footnote_space(fig), 1, 1))
    return _png(fig)


def canonical_chart(scores, percent, genotypes, cluster_of, names, sizes, palette=None):
    spec = chart_gallery.analysis_spec(palette=palette)
    scores = np.asarray(scores, dtype=float)[:, :2]
    cluster_of = np.asarray(cluster_of)
    colors = palette_colors(palette, len(names))
    width, height = figure_size()
    fig, ax = plt.subplots(figsize=(width, height * 1.05))
    for c in range(len(names)):
        points = scores[cluster_of == c]
        color = colors[c % len(colors)]
        ax.scatter(points[:, 0], points[:, 1], s=30, color=color, edgecolors="white", linewidths=0.5, zorder=3)
        if len(points) >= 3:
            draw_group_hull(ax, points, color)
    low, high = scores.min(axis=0), scores.max(axis=0)
    pad = 0.12 * ((high - low).max() or 1.0)
    ax.set_xlim(low[0] - pad, high[0] + pad)
    ax.set_ylim(low[1] - pad, high[1] + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0, color=COLORS["grid"], linewidth=0.9, zorder=0)
    ax.axvline(0, color=COLORS["grid"], linewidth=0.9, zorder=0)
    chart_gallery.value_grid(ax, spec, axis="both")
    chart_gallery.decorate(ax, spec, "Canonical variates", f"CV1 ({percent[0]:.1f}% of D²)", f"CV2 ({percent[1]:.1f}% of D²)")
    if len(genotypes) <= 60:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        for (x, y), name in zip(chart_gallery.nudge_labels(scores, ax, min_dx=0.07, min_dy=0.035), genotypes):
            ax.text(x + 0.01 * (x1 - x0), y + 0.01 * (y1 - y0), str(name), fontsize=spec.tick_size * 0.78,
                    color=COLORS["muted"], zorder=4)
    _cluster_legend(ax, spec, names, sizes, colors, loc="best")
    chart_gallery.footnote(fig, spec, "Genotype means on the first two canonical variates; Euclidean distances in this space "
                                      "approximate D². Outlines: Tocher clusters.")
    fig.tight_layout(rect=(0, chart_gallery.footnote_space(fig), 1, 1))
    return _png(fig)
