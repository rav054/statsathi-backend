"""G×E charts: GGE biplot views, AMMI biplots, adaptability regression lines and the genotype × environment heatmap."""

import base64

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless rendering for the web server
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon

from ..figure_style import COLORS, export_figure, figure_size
from . import chart_gallery

MAX_LABELLED = 60  # beyond this, only the most important points are labelled


def _png(fig):
    try:
        content, _ = export_figure(fig, "png", 150)
    finally:
        plt.close(fig)
    return base64.b64encode(content).decode("utf-8")


def _square_figure():
    side = figure_size()[0] * 0.95
    return plt.subplots(figsize=(side, side * 0.92))


def _display_scale(genotypes, environments):
    """Factor applied to environment vectors so both sets are visible; angles and relative lengths are unchanged."""
    g = float(np.linalg.norm(genotypes, axis=1).max())
    e = float(np.linalg.norm(environments, axis=1).max())
    if g <= 0 or e <= 0:
        return 1.0
    ratio = g / e
    return 1.0 if 0.67 <= ratio <= 1.5 else ratio


def _frame(ax, spec, points, title, xlabel, ylabel):
    stacked = np.vstack([np.atleast_2d(np.asarray(p, dtype=float)) for p in points] + [np.zeros((1, 2))])
    low, high = stacked.min(axis=0), stacked.max(axis=0)
    span = float((high - low).max()) or 1.0
    centre = (low + high) / 2
    half = 0.5 * span + 0.14 * span
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_aspect("equal", adjustable="box")  # equal units on both axes, so angles and distances are meaningful
    ax.axhline(0, color=COLORS["grid"], linewidth=0.9, zorder=0)
    ax.axvline(0, color=COLORS["grid"], linewidth=0.9, zorder=0)
    ax.grid(False)
    chart_gallery.decorate(ax, spec, title, xlabel, ylabel)
    return span


def _labels(ax, spec, xy, names, color, bold=False, force=False):
    if len(names) > MAX_LABELLED and not force:
        return
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    if not len(xy):
        return
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    offset_x, offset_y = 0.01 * (x1 - x0), 0.01 * (y1 - y0)
    for (x, y), name in zip(chart_gallery.nudge_labels(xy, ax, min_dx=0.07, min_dy=0.035), names):
        ax.text(x + offset_x, y + offset_y, str(name), ha="left", va="bottom", fontsize=spec.tick_size * 0.82,
                color=color, fontweight="bold" if bold else "normal", zorder=6)


def _genotype_points(ax, spec, G, color=None):
    ax.scatter(G[:, 0], G[:, 1], s=24, color=color or spec.color(0), edgecolors="white", linewidths=0.5, zorder=4)


def _environment_vectors(ax, spec, E):
    for x, y in E:
        ax.annotate("", xy=(x, y), xytext=(0, 0), zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color=spec.color(1), linewidth=1.1, shrinkA=0, shrinkB=0))


def _legend(ax, spec, genotype=True, environment=True, extra=()):
    handles = []
    if genotype:
        handles.append(Line2D([], [], marker="o", linestyle="", color=spec.color(0), label="Genotypes"))
    if environment:
        handles.append(Line2D([], [], color=spec.color(1), label="Environments"))
    handles.extend(extra)
    chart_gallery.add_legend(ax, spec, handles=handles, loc="best")


def _pc_labels(percent, what="G + GE"):
    return f"PC1 ({percent[0]:.1f}% of {what})", f"PC2 ({percent[1]:.1f}% of {what})"


def _scale_note(scale):
    return f" Environment vectors drawn ×{scale:.2g} for display." if scale != 1.0 else ""


# ---------------------------------------------------------------------------
# GGE biplot views
# ---------------------------------------------------------------------------
def gge_biplot_chart(G, E, genotypes, environments, percent, svp_label, scaling_label, palette=None):
    spec = chart_gallery.analysis_spec(palette=palette)
    scale = _display_scale(G, E)
    shown = E * scale
    fig, ax = _square_figure()
    _frame(ax, spec, [G, shown * 1.12], "GGE biplot", *_pc_labels(percent))
    _environment_vectors(ax, spec, shown)
    _genotype_points(ax, spec, G)
    _labels(ax, spec, G, genotypes, COLORS["text"])
    _labels(ax, spec, shown, environments, spec.color(1), bold=True, force=True)
    _legend(ax, spec)
    chart_gallery.footnote(fig, spec, f"Environment-centred data ({scaling_label}), {svp_label} singular value partitioning. "
                                      f"PC1 and PC2 explain {sum(percent):.1f}% of the genotype + genotype × environment "
                                      f"variation.{_scale_note(scale)}")
    fig.tight_layout(rect=(0, chart_gallery.footnote_space(fig), 1, 1))
    return _png(fig)


def which_won_where_chart(G, E, genotypes, environments, www, percent, palette=None):
    spec = chart_gallery.analysis_spec(palette=palette)
    scale = _display_scale(G, E)
    shown = E * scale
    fig, ax = _square_figure()
    span = _frame(ax, spec, [G, shown * 1.12], "Which won where", *_pc_labels(percent))
    vertices = www["vertices"]
    if len(vertices) >= 3:
        ax.add_patch(Polygon(G[vertices], closed=True, fill=False, edgecolor=COLORS["text"], linewidth=1.1, zorder=2))
    elif len(vertices) == 2:
        ax.plot(G[vertices, 0], G[vertices, 1], color=COLORS["text"], linewidth=1.1, zorder=2)
    for ray in www["rays"]:
        ax.plot([0, ray[0] * span * 3], [0, ray[1] * span * 3], color=COLORS["muted"], linestyle="--", linewidth=0.9, zorder=1)
    _genotype_points(ax, spec, G)
    others = [i for i in range(len(genotypes)) if i not in vertices]
    _labels(ax, spec, G[others], [genotypes[i] for i in others], COLORS["muted"])
    _labels(ax, spec, G[vertices], [genotypes[i] for i in vertices], COLORS["text"], bold=True, force=True)
    ax.scatter(shown[:, 0], shown[:, 1], marker="^", s=40, color=spec.color(1), edgecolors="white", linewidths=0.5, zorder=5)
    _labels(ax, spec, shown, environments, spec.color(1), bold=True, force=True)
    _legend(ax, spec, extra=[Line2D([], [], color=COLORS["text"], label="Polygon of vertex genotypes"),
                             Line2D([], [], color=COLORS["muted"], linestyle="--", label="Sector boundaries")])
    groups = {}
    for env, winner in zip(environments, www["winners"]):
        groups.setdefault(genotypes[winner], []).append(str(env))
    summary = "; ".join(f"{winner} in {', '.join(envs)}" for winner, envs in groups.items())
    chart_gallery.footnote(fig, spec, f"The vertex genotype of each sector performs best in the environments inside it. "
                                      f"Winners: {summary}.{_scale_note(scale)}")
    fig.tight_layout(rect=(0, chart_gallery.footnote_space(fig), 1, 1))
    return _png(fig)


def mean_stability_chart(G, E, genotypes, aec, percent, palette=None):
    spec = chart_gallery.analysis_spec(palette=palette)
    unit = np.asarray(aec["unit"], dtype=float)
    perpendicular = np.array([-unit[1], unit[0]])
    reach = float(np.linalg.norm(G, axis=1).max()) or 1.0
    fig, ax = _square_figure()
    span = _frame(ax, spec, [G, unit * reach * 1.1], "Mean vs. stability", *_pc_labels(percent))
    far = span * 3
    ax.plot([-unit[0] * far, unit[0] * far], [-unit[1] * far, unit[1] * far], color=COLORS["text"], linewidth=1.1, zorder=2)
    ax.annotate("", xy=unit * reach * 1.08, xytext=unit * reach * 0.8, zorder=3,
                arrowprops=dict(arrowstyle="-|>", color=COLORS["text"], linewidth=1.4))
    ax.plot([-perpendicular[0] * far, perpendicular[0] * far], [-perpendicular[1] * far, perpendicular[1] * far],
            color=COLORS["muted"], linestyle="--", linewidth=0.9, zorder=1)
    for point, projection in zip(G, aec["projection"]):
        foot = unit * projection
        ax.plot([point[0], foot[0]], [point[1], foot[1]], color=COLORS["muted"], linestyle=":", linewidth=0.9, zorder=2)
    ax.scatter(*(unit * reach * 1.08), s=90, facecolors="none", edgecolors=spec.color(1), linewidths=1.5, zorder=5)
    _genotype_points(ax, spec, G)
    _labels(ax, spec, G, genotypes, COLORS["text"])
    _legend(ax, spec, environment=False, extra=[
        Line2D([], [], color=COLORS["text"], label="Average-environment axis"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=spec.color(1), label="Average environment"),
    ])
    projection = np.asarray(aec["projection"])
    instability = np.asarray(aec["instability"])
    best = int(np.argmax(projection))
    above = np.where(projection > 0)[0]
    stable = int(above[np.argmin(instability[above])]) if above.size else int(np.argmin(instability))
    chart_gallery.footnote(fig, spec, f"The arrow points to better mean performance; dotted projections onto the axis show "
                                      f"instability (longer = less stable). Best mean: {genotypes[best]}. Most stable "
                                      f"above-average genotype: {genotypes[stable]}.")
    fig.tight_layout(rect=(0, chart_gallery.footnote_space(fig), 1, 1))
    return _png(fig)


def discriminativeness_chart(E, environments, aec, percent, palette=None):
    spec = chart_gallery.analysis_spec(palette=palette)
    E = np.asarray(E, dtype=float)
    lengths = np.asarray(aec["environment_length"])
    radius = float(lengths.max()) or 1.0
    unit = np.asarray(aec["unit"], dtype=float)
    fig, ax = _square_figure()
    _frame(ax, spec, [E * 1.15, [[radius, radius], [-radius, -radius]]], "Discriminativeness vs. representativeness",
           *_pc_labels(percent))
    for fraction in (0.25, 0.5, 0.75, 1.0):
        ax.add_patch(Circle((0, 0), radius * fraction, fill=False, edgecolor=COLORS["grid"], linewidth=0.8, zorder=0))
    ax.plot([-unit[0] * radius * 1.1, unit[0] * radius * 1.1], [-unit[1] * radius * 1.1, unit[1] * radius * 1.1],
            color=COLORS["text"], linewidth=1.1, zorder=2)
    ax.scatter(*aec["mean_environment"], s=70, facecolors="none", edgecolors=COLORS["text"], linewidths=1.4, zorder=5)
    _environment_vectors(ax, spec, E)
    _labels(ax, spec, E, environments, spec.color(1), bold=True, force=True)
    _legend(ax, spec, genotype=False, extra=[
        Line2D([], [], color=COLORS["text"], label="Average-environment axis"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=COLORS["text"], label="Average environment"),
    ])
    most_discriminating = environments[int(np.argmax(lengths))]
    most_representative = environments[int(np.argmin(aec["angle"]))]
    chart_gallery.footnote(fig, spec, f"Longer vectors: environments that separate genotypes more clearly. Smaller angle "
                                      f"with the average-environment axis: more representative. Most discriminating: "
                                      f"{most_discriminating}; most representative: {most_representative}.")
    fig.tight_layout(rect=(0, chart_gallery.footnote_space(fig), 1, 1))
    return _png(fig)


def ranking_chart(G, genotypes, aec, percent, palette=None):
    spec = chart_gallery.analysis_spec(palette=palette)
    G = np.asarray(G, dtype=float)
    ideal = np.asarray(aec["ideal_genotype"], dtype=float)
    unit = np.asarray(aec["unit"], dtype=float)
    distances = np.asarray(aec["genotype_distance"])
    length = float(np.linalg.norm(ideal)) or 1.0
    fig, ax = _square_figure()
    span = _frame(ax, spec, [G, ideal * 1.1], "Ranking genotypes relative to the ideal genotype", *_pc_labels(percent))
    step = length / 3
    for k in range(1, min(10, int(np.ceil(distances.max() / step)) + 1)):
        ax.add_patch(Circle(tuple(ideal), step * k, fill=False, edgecolor=COLORS["grid"], linewidth=0.8, zorder=0))
    far = span * 3
    ax.plot([-unit[0] * far, unit[0] * far], [-unit[1] * far, unit[1] * far], color=COLORS["text"], linewidth=1.0, zorder=1)
    ax.scatter(*ideal, marker="*", s=180, color=spec.color(1), edgecolors="white", linewidths=0.6, zorder=6)
    _genotype_points(ax, spec, G)
    _labels(ax, spec, G, genotypes, COLORS["text"])
    _legend(ax, spec, environment=False, extra=[
        Line2D([], [], marker="*", linestyle="", color=spec.color(1), markersize=11, label="Ideal genotype"),
    ])
    order = np.argsort(distances)[:5]
    ranking = ", ".join(f"{i + 1}. {genotypes[g]}" for i, g in enumerate(order))
    chart_gallery.footnote(fig, spec, f"The ideal genotype has the best mean performance and is perfectly stable. Genotypes "
                                      f"on inner circles are closer to it. Closest: {ranking}.")
    fig.tight_layout(rect=(0, chart_gallery.footnote_space(fig), 1, 1))
    return _png(fig)


# ---------------------------------------------------------------------------
# AMMI biplots
# ---------------------------------------------------------------------------
def ammi1_chart(genotype_means, environment_means, genotype_ipca1, environment_ipca1, genotypes, environments, grand_mean,
                percent_ipca1, response, palette=None):
    spec = chart_gallery.analysis_spec(palette=palette)
    fig, ax = plt.subplots(figsize=figure_size())
    ax.axvline(grand_mean, color=COLORS["muted"], linestyle="--", linewidth=1.0, zorder=1)
    ax.axhline(0, color=COLORS["grid"], linewidth=1.0, zorder=1)
    ax.scatter(genotype_means, genotype_ipca1, s=26, color=spec.color(0), edgecolors="white", linewidths=0.5, zorder=4)
    ax.scatter(environment_means, environment_ipca1, marker="^", s=44, color=spec.color(1), edgecolors="white",
               linewidths=0.5, zorder=4)
    xs = np.concatenate([genotype_means, environment_means])
    ys = np.concatenate([genotype_ipca1, environment_ipca1])
    x_pad = 0.1 * ((xs.max() - xs.min()) or 1.0)
    y_pad = 0.12 * ((ys.max() - ys.min()) or 1.0)
    ax.set_xlim(xs.min() - x_pad, xs.max() + x_pad)
    ax.set_ylim(ys.min() - y_pad, ys.max() + y_pad)
    _labels(ax, spec, np.column_stack([genotype_means, genotype_ipca1]), genotypes, COLORS["text"])
    _labels(ax, spec, np.column_stack([environment_means, environment_ipca1]), environments, spec.color(1), bold=True, force=True)
    chart_gallery.value_grid(ax, spec, axis="both")
    chart_gallery.decorate(ax, spec, "AMMI1 biplot", f"Mean {response}", f"IPCA1 ({percent_ipca1:.1f}% of G × E)")
    handles = [Line2D([], [], marker="o", linestyle="", color=spec.color(0), label="Genotypes"),
               Line2D([], [], marker="^", linestyle="", color=spec.color(1), label="Environments"),
               Line2D([], [], color=COLORS["muted"], linestyle="--", label=f"Grand mean ({chart_gallery.format_value(grand_mean)})")]
    chart_gallery.add_legend(ax, spec, handles=handles, loc="best")
    chart_gallery.footnote(fig, spec, "Points right of the dashed line are above the grand mean. IPCA1 scores near zero mean "
                                      "little interaction with environments; genotypes and environments with IPCA1 scores of "
                                      "the same sign interact positively.")
    fig.tight_layout(rect=(0, chart_gallery.footnote_space(fig), 1, 1))
    return _png(fig)


def ammi2_chart(genotype_scores, environment_scores, genotypes, environments, percents, palette=None):
    spec = chart_gallery.analysis_spec(palette=palette)
    G = np.asarray(genotype_scores, dtype=float)
    scale = _display_scale(G, environment_scores)
    shown = np.asarray(environment_scores, dtype=float) * scale
    fig, ax = _square_figure()
    _frame(ax, spec, [G, shown * 1.12], "AMMI2 biplot", f"IPCA1 ({percents[0]:.1f}% of G × E)",
           f"IPCA2 ({percents[1]:.1f}% of G × E)")
    _environment_vectors(ax, spec, shown)
    _genotype_points(ax, spec, G)
    _labels(ax, spec, G, genotypes, COLORS["text"])
    _labels(ax, spec, shown, environments, spec.color(1), bold=True, force=True)
    _legend(ax, spec)
    chart_gallery.footnote(fig, spec, "Genotypes near the origin interact little with the environments (stable). A genotype "
                                      f"and an environment in the same direction interact positively.{_scale_note(scale)}")
    fig.tight_layout(rect=(0, chart_gallery.footnote_space(fig), 1, 1))
    return _png(fig)


# ---------------------------------------------------------------------------
# Adaptability and data overview
# ---------------------------------------------------------------------------
def adaptability_chart(means, genotypes, environments, slopes, response, higher_is_better=True, palette=None):
    spec = chart_gallery.analysis_spec(palette=palette)
    Y = np.asarray(means, dtype=float)
    grand = Y.mean()
    genotype_means, environment_means = Y.mean(axis=1), Y.mean(axis=0)
    xs = np.array([environment_means.min(), environment_means.max()])
    n = len(genotypes)
    order = np.argsort(-genotype_means if higher_is_better else genotype_means)
    highlight = list(order[:5]) if n > 10 else list(range(n))

    width, height = figure_size()
    fig, ax = plt.subplots(figsize=(width * 1.25, height))
    ax.plot(xs, xs, color=COLORS["text"], linestyle="--", linewidth=1.2, zorder=2, label="Average response (b = 1)")
    for i in range(n):
        line = genotype_means[i] + slopes[i] * (xs - grand)
        if i in highlight:
            color = spec.color(highlight.index(i))
            ax.plot(xs, line, color=color, linewidth=1.7, zorder=3, label=f"{genotypes[i]} (b = {slopes[i]:.2f})")
            ax.scatter(environment_means, Y[i], s=14, color=color, alpha=0.75, zorder=3)
        else:
            ax.plot(xs, line, color="#CBD5E1", linewidth=0.8, zorder=1)
    top = ax.secondary_xaxis("top")
    env_order = np.argsort(environment_means)
    top.set_xticks(environment_means[env_order], labels=[str(environments[j]) for j in env_order])
    for label in top.get_xticklabels():
        label.set_fontsize(spec.tick_size * 0.8)
        label.set_rotation(45 if len(environments) > 6 else 0)
        label.set_ha("left" if len(environments) > 6 else "center")
    chart_gallery.value_grid(ax, spec, axis="both")
    chart_gallery.decorate(ax, spec, "Adaptability: response to environment quality", "Environment mean", response)
    chart_gallery.add_legend(ax, spec, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    note = ("Each line is a genotype's regression on the environment mean (Finlay-Wilkinson / Eberhart-Russell). b > 1: "
            "responds strongly to favourable environments; b < 1: relatively better in poor environments.")
    if n > 10:
        note += " The five best genotypes by mean are coloured."
    chart_gallery.footnote(fig, spec, note)
    fig.tight_layout(rect=(0, chart_gallery.footnote_space(fig), 1, 1))
    return _png(fig)


def heatmap_chart(means, genotypes, environments, response, higher_is_better=True, palette=None):
    spec = chart_gallery.analysis_spec(palette=palette)
    Y = np.asarray(means, dtype=float)
    genotype_order = np.argsort(-Y.mean(axis=1) if higher_is_better else Y.mean(axis=1))
    environment_order = np.argsort(Y.mean(axis=0))
    matrix = Y[genotype_order][:, environment_order]
    G, E = matrix.shape
    width, height = figure_size()
    fig, ax = plt.subplots(figsize=(max(width, 0.6 * E + 2.8), max(height, 0.28 * G + 1.9)))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis" if higher_is_better else "viridis_r")
    if G * E <= 300:
        for i in range(G):
            for j in range(E):
                value = matrix[i, j]
                ax.text(j, i, f"{value:.3g}", ha="center", va="center", fontsize=spec.tick_size * 0.72,
                        color=chart_gallery.readable_text(image.cmap(image.norm(value))))
    ax.set_xticks(range(E), labels=[str(environments[j]) for j in environment_order])
    ax.set_yticks(range(G), labels=[str(genotypes[i]) for i in genotype_order])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label(response)
    colorbar.outline.set_visible(False)
    chart_gallery.decorate(ax, spec, f"{response}: genotype × environment means", "Environment (lowest to highest mean)",
                           "Genotype (best mean first)")
    for label in ax.get_yticklabels():
        label.set_fontsize(min(spec.tick_size, max(5.0, 260 / G)))
    chart_gallery.rotate_long_labels(ax)
    chart_gallery.footnote(fig, spec, "Rows are sorted by genotype mean and columns by environment mean; rows whose colour "
                                      "pattern changes across columns show crossover interaction.")
    fig.tight_layout(rect=(0, chart_gallery.footnote_space(fig), 1, 1))
    return _png(fig)
