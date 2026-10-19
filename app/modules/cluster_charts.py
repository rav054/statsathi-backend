"""Dendrogram styles for hierarchical clustering (rectangular, circular, heatmap with dendrograms) and the chart for
choosing the number of clusters."""

import base64

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless rendering for the web server
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

from ..figure_style import COLORS, export_figure, figure_size, palette_colors
from . import chart_gallery

UNCLUSTERED = "#94A3B8"
MAX_LEAF_LABELS = 150
MAX_LEGEND_CLUSTERS = 12


def _png(fig):
    try:
        content, _ = export_figure(fig, "png", 150)
    finally:
        plt.close(fig)
    return base64.b64encode(content).decode("utf-8")


def cut_between(linkage_matrix, n_clusters):
    """A height that cuts the tree into n_clusters: midway between the last merge kept and the next one."""
    heights = np.sort(np.asarray(linkage_matrix, dtype=float)[:, 2])
    n = heights.size + 1
    if n_clusters <= 1:
        return float(heights[-1] * 1.05)
    if n_clusters >= n:
        return float(heights[0] / 2)
    return float((heights[n - n_clusters - 1] + heights[n - n_clusters]) / 2)


def link_color_function(linkage_matrix, leaf_clusters, colors):
    """Colour of each tree node: its cluster's colour when all leaves below it belong to one cluster, grey otherwise."""
    n = len(leaf_clusters)
    owner = [int(c) for c in leaf_clusters] + [None] * (n - 1)
    for i, row in enumerate(np.asarray(linkage_matrix)):
        left, right = owner[int(row[0])], owner[int(row[1])]
        owner[n + i] = left if left is not None and left == right else None

    def color(node):
        cluster = owner[node]
        return colors[cluster % len(colors)] if cluster is not None else UNCLUSTERED

    color.owner = owner
    return color


def _legend_handles(clusters, colors):
    counts = np.bincount(clusters)
    return [Patch(facecolor=colors[c % len(colors)], label=f"Cluster {c} (n = {counts[c]})") for c in range(counts.size)]


def dendrogram_chart(linkage_matrix, leaf_labels, leaf_clusters, cut, method_label, cophenetic=None, palette=None):
    spec = chart_gallery.analysis_spec(palette=palette)
    clusters = np.asarray(leaf_clusters, dtype=int)
    K = int(clusters.max()) + 1
    colors = palette_colors(palette, K)
    n = len(leaf_labels)
    horizontal = n > 60
    width, height = figure_size()
    size = (width, min(40.0, max(height, 0.16 * n + 1.5))) if horizontal else (max(width, min(18.0, 0.22 * n + 2.5)), height * 1.05)
    fig, ax = plt.subplots(figsize=size)
    show_labels = n <= MAX_LEAF_LABELS
    result = hierarchy.dendrogram(linkage_matrix, ax=ax, labels=[str(v) for v in leaf_labels],
                                  orientation="left" if horizontal else "top", leaf_rotation=0 if horizontal else 90,
                                  link_color_func=link_color_function(linkage_matrix, clusters, colors),
                                  no_labels=not show_labels)
    (ax.axvline if horizontal else ax.axhline)(cut, color=COLORS["text"], linestyle="--", linewidth=1.0)
    ax.grid(False)
    chart_gallery.decorate(ax, spec, f"Dendrogram ({method_label})", "Height" if horizontal else "", "" if horizontal else "Height")
    if show_labels:
        font = min(spec.tick_size, max(4.5, 600 / max(n, 1)))
        for tick, leaf in zip(ax.get_yticklabels() if horizontal else ax.get_xticklabels(), result["leaves"]):
            tick.set_color(colors[clusters[leaf] % len(colors)])
            tick.set_fontsize(font)
    if K <= MAX_LEGEND_CLUSTERS:
        chart_gallery.add_legend(ax, spec, handles=_legend_handles(clusters, colors), loc="upper right")
    note = f"Branches and labels are coloured by the {K} clusters obtained by cutting the tree at the dashed line (height {cut:.3g})."
    if cophenetic is not None:
        note += f" Cophenetic correlation: {cophenetic:.3f}."
    if not show_labels:
        note += f" Labels are hidden for more than {MAX_LEAF_LABELS} observations."
    chart_gallery.footnote(fig, spec, note)
    fig.tight_layout(rect=(0, chart_gallery.footnote_space(fig), 1, 1))
    return _png(fig)


def circular_dendrogram_chart(linkage_matrix, leaf_labels, leaf_clusters, cut, method_label, palette=None):
    spec = chart_gallery.analysis_spec(palette=palette)
    clusters = np.asarray(leaf_clusters, dtype=int)
    K = int(clusters.max()) + 1
    colors = palette_colors(palette, K)
    tree = hierarchy.dendrogram(linkage_matrix, no_plot=True, labels=[str(v) for v in leaf_labels],
                                link_color_func=link_color_function(linkage_matrix, clusters, colors))
    n = len(leaf_labels)
    top = float(max(max(d) for d in tree["dcoord"])) or 1.0

    def angle(x):
        return 2 * np.pi * (np.asarray(x, dtype=float) - 5.0) / (10.0 * n)

    side = figure_size()[0] * 1.05
    fig = plt.figure(figsize=(side, side))
    ax = fig.add_subplot(projection="polar")
    for xs, ys, color in zip(tree["icoord"], tree["dcoord"], tree["color_list"]):
        a = angle(xs)
        r = top - np.asarray(ys, dtype=float)  # leaves on the outer ring, root at the centre
        ax.plot([a[0], a[1]], [r[0], r[1]], color=color, linewidth=0.9)
        arc = np.linspace(a[1], a[2], max(4, int(abs(a[2] - a[1]) / (2 * np.pi) * 240) + 2))
        ax.plot(arc, np.full(arc.size, r[1]), color=color, linewidth=0.9)
        ax.plot([a[2], a[3]], [r[2], r[3]], color=color, linewidth=0.9)
    ring = np.linspace(0, 2 * np.pi, 361)
    ax.plot(ring, np.full(ring.size, top - cut), color=COLORS["text"], linestyle="--", linewidth=0.8)
    show_labels = n <= MAX_LEAF_LABELS
    if show_labels:
        font = max(4.5, min(spec.tick_size, 900 / n))
        for position, (leaf, label) in enumerate(zip(tree["leaves"], tree["ivl"])):
            theta = float(angle(5.0 + 10.0 * position))
            degrees = np.degrees(theta)
            flip = 90 < degrees < 270
            ax.text(theta, top * 1.03, label, rotation=degrees + 180 if flip else degrees, rotation_mode="anchor",
                    ha="right" if flip else "left", va="center", fontsize=font, color=colors[clusters[leaf] % len(colors)])
    ax.set_ylim(0, top * (1.3 if show_labels else 1.02))
    ax.set_axis_off()
    fig.suptitle(f"Circular dendrogram ({method_label})", x=0.02, ha="left", fontweight="bold", fontsize=spec.title_size,
                 color=COLORS["text"])
    if K <= MAX_LEGEND_CLUSTERS:
        fig.legend(handles=_legend_handles(clusters, colors), loc="upper right", frameon=False,
                   prop={"size": spec.tick_size * 0.85})
    chart_gallery.footnote(fig, spec, f"The same tree drawn radially: the root is at the centre and observations on the outer "
                                      f"ring. The dashed circle is the cut that gives {K} clusters.")
    fig.tight_layout(rect=(0, chart_gallery.footnote_space(fig), 1, 0.95))
    return _png(fig)


def heatmap_dendrogram_chart(scaled, variables, leaf_labels, linkage_matrix, leaf_clusters, palette=None):
    spec = chart_gallery.analysis_spec(palette=palette)
    scaled = np.asarray(scaled, dtype=float)
    clusters = np.asarray(leaf_clusters, dtype=int)
    colors = palette_colors(palette, int(clusters.max()) + 1)
    n, p = scaled.shape
    frame = pd.DataFrame(np.clip(scaled, -3, 3), columns=variables, index=[str(v) for v in leaf_labels])
    width, height = figure_size()
    size = (max(width, 0.65 * p + 3.8), min(30.0, max(height * 1.25, 0.16 * n + 2.6)))
    column_cluster = p >= 3
    column_linkage = hierarchy.linkage(pdist(scaled.T, "correlation"), "average") if column_cluster else None
    grid = sns.clustermap(frame, row_linkage=linkage_matrix, col_linkage=column_linkage, col_cluster=column_cluster,
                          cmap="RdBu_r", center=0, vmin=-3, vmax=3,
                          row_colors=[colors[c % len(colors)] for c in clusters], figsize=size,
                          dendrogram_ratio=(0.2, 0.14), yticklabels=n <= 80, xticklabels=True,
                          cbar_pos=(0.02, 0.84, 0.025, 0.12), cbar_kws={"label": "z-score"})
    try:  # figures are exported at their exact size, so leave room for the title and the column labels
        grid.gs.update(top=0.94, bottom=0.16)
    except Exception:
        pass
    grid.ax_heatmap.set_xlabel("")
    grid.ax_heatmap.set_ylabel("")
    for label in grid.ax_heatmap.get_xticklabels():
        label.set_fontsize(spec.tick_size)
        label.set_rotation(45)
        label.set_ha("right")
    for label in grid.ax_heatmap.get_yticklabels():
        label.set_fontsize(min(spec.tick_size, max(4.0, 500 / max(n, 1))))
    grid.ax_col_dendrogram.set_title("Heatmap with dendrograms (standardized values)", loc="left", fontweight="bold",
                                     fontsize=spec.label_size, color=COLORS["text"], pad=10)
    return _png(grid.figure)


def k_selection_chart(rows, suggested, palette=None):
    spec = chart_gallery.analysis_spec(palette=palette)
    ks = [row["k"] for row in rows]
    width, height = figure_size()
    fig, (left, right) = plt.subplots(1, 2, figsize=(width * 1.2, height * 0.85))
    for position, (method, label) in enumerate((("hierarchical", "Hierarchical"), ("kmeans", "K-means"))):
        values = [np.nan if row[f"{method}_silhouette"] is None else row[f"{method}_silhouette"] for row in rows]
        color = spec.color(position)
        left.plot(ks, values, marker="o", markersize=4, color=color, linewidth=1.6, label=label)
        best = suggested.get(method)
        if best in ks:
            left.scatter([best], [values[ks.index(best)]], s=130, facecolors="none", edgecolors=color, linewidths=1.8, zorder=4)
    left.set_xticks(ks)
    chart_gallery.value_grid(left, spec, axis="both")
    chart_gallery.decorate(left, spec, "Average silhouette width", "Number of clusters (k)", "Silhouette")
    chart_gallery.add_legend(left, spec, loc="best")
    right.plot(ks, [row["kmeans_wss"] for row in rows], marker="o", markersize=4, color=spec.color(1), linewidth=1.6)
    right.set_xticks(ks)
    chart_gallery.value_grid(right, spec, axis="both")
    chart_gallery.decorate(right, spec, "K-means within-cluster SS", "Number of clusters (k)", "WSS")
    for ax in (left, right):
        ax.title.set_fontsize(spec.label_size)
    fig.suptitle("Choosing the number of clusters", x=0.01, ha="left", fontweight="bold", fontsize=spec.title_size,
                 color=COLORS["text"])
    chart_gallery.footnote(fig, spec, f"Circles mark the k with the highest average silhouette (hierarchical: "
                                      f"{suggested.get('hierarchical')}, K-means: {suggested.get('kmeans')}). On the right, look "
                                      "for the elbow where adding clusters stops reducing the within-cluster sum of squares much.")
    fig.tight_layout(rect=(0, chart_gallery.footnote_space(fig), 1, 0.94))
    return _png(fig)
