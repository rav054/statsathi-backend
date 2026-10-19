"""The chart researchers publish from a trial: treatment means, their error bars and the post-hoc letters.

Letters sit above each bar, so the figure says exactly what the comparison table says. Treatments that
share the top letter are filled in the palette's first colour; the rest are a lighter tint of it, because
colour here marks the top group, never a rank.
"""

import base64
import io

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from ..figure_style import (COLORS, category_label_fit, export_figure, figure_size, figure_style, natural_order,
                            palette_colors, pretty_label, rotate_category_labels)


def _tint(hex_color, amount=0.55):
    """`hex_color` mixed with the chart background; 0 keeps the colour, 1 is the background."""
    r, g, b = to_rgb(hex_color)
    br, bg, bb = to_rgb(COLORS["background"])
    mix = (r + (br - r) * amount, g + (bg - g) * amount, b + (bb - b) * amount)
    return "#%02X%02X%02X" % tuple(round(c * 255) for c in mix)


def top_group(letters):
    """Labels sharing a letter with the best treatment (the letters are ordered, so 'a' is the top group)."""
    if not letters:
        return set()
    best = min(letters.values(), key=lambda value: sorted(value)[0] if value else "z")
    marks = set(best)
    return {label for label, value in letters.items() if marks & set(value)}


def mean_separation_figure(descriptives, letters, *, title, xlabel, ylabel, palette=None, preset=None,
                           error_label="± SE", order=None):
    """Bar chart of treatment means with error bars and mean-separation letters. Returns a Matplotlib figure."""
    labels = list(order or natural_order(descriptives.keys()))
    means = np.array([descriptives[label]["mean"] for label in labels], dtype=float)
    errors = np.array([descriptives[label].get("se") or 0.0 for label in labels], dtype=float)

    colors = palette_colors(palette)
    primary = colors[0]
    highlighted = top_group(letters)
    fill = [primary if (not letters or label in highlighted) else _tint(primary) for label in labels]

    with figure_style(preset, palette):
        fig, ax = plt.subplots(figsize=figure_size(preset))
        positions = np.arange(len(labels))
        headroom = max(means.max() + errors.max(), 0) or 1.0

        ax.bar(positions, means, width=0.68, color=fill, linewidth=0)
        if errors.any():
            ax.errorbar(positions, means, yerr=errors, fmt="none", ecolor=COLORS["text"],
                        elinewidth=1.0, capsize=4, capthick=1.0)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(label) for label in labels])
        ax.set_xlim(-0.6, len(labels) - 0.4)

        # Treatment names in a factorial trial ("High / Cultivar_1") are long. Measure how they fit:
        # turn them where that is enough, and lay the bars on their side when it is not.
        fit = category_label_fit(fig, ax, labels)
        if fit == "sideways":
            plt.close(fig)
            return _sideways_figure(labels, means, errors, fill, letters, title=title, xlabel=xlabel,
                                    ylabel=ylabel, preset=preset, error_label=error_label)

        for position, mean, error, label in zip(positions, means, errors, labels):
            mark = letters.get(label, "")
            if mark:
                ax.text(position, mean + error + headroom * 0.035, mark, ha="center", va="bottom",
                        fontweight="bold", color=COLORS["text"])

        if fit == "rotate":
            rotate_category_labels(ax)
        ax.set_ylim(0, headroom * 1.18)
        ax.set_title(title)
        ax.set_xlabel(pretty_label(xlabel))
        ax.set_ylabel(pretty_label(ylabel))
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)
        _caption(fig, letters, error_label)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def _caption(fig, letters, error_label):
    if letters:
        fig.text(0.99, 0.01, f"Means {error_label}. Treatments sharing a letter do not differ.",
                 ha="right", va="bottom", fontsize=plt.rcParams["xtick.labelsize"], color=COLORS["muted"])


def _sideways_figure(labels, means, errors, fill, letters, *, title, xlabel, ylabel, preset, error_label):
    """Bars on their side: long treatment names read straight, one per row, and never collide.

    The figure grows with the number of treatments instead of squeezing them together.
    """
    width, height = figure_size(preset)
    height = max(height, 1.6 + 0.34 * len(labels))
    fig, ax = plt.subplots(figsize=(width, height))
    positions = np.arange(len(labels))[::-1]  # first treatment at the top

    ax.barh(positions, means, height=0.68, color=fill, linewidth=0)
    if errors.any():
        ax.errorbar(means, positions, xerr=errors, fmt="none", ecolor=COLORS["text"],
                    elinewidth=1.0, capsize=4, capthick=1.0)

    reach = max(means.max() + errors.max(), 0) or 1.0
    for position, mean, error, label in zip(positions, means, errors, labels):
        mark = letters.get(label, "")
        if mark:
            ax.text(mean + error + reach * 0.02, position, mark, ha="left", va="center",
                    fontweight="bold", color=COLORS["text"])

    ax.set_yticks(positions)
    ax.set_yticklabels([str(label) for label in labels])
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_xlim(0, reach * 1.12)
    ax.set_title(title)
    ax.set_xlabel(pretty_label(ylabel))   # the measured value now runs along the x axis
    ax.set_ylabel(pretty_label(xlabel))
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    _caption(fig, letters, error_label)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def mean_separation_png(descriptives, letters, **options):
    """The same chart as a base64 PNG, or None when there is nothing to compare."""
    if not descriptives or len(descriptives) < 2:
        return None
    fig = None
    try:
        fig = mean_separation_figure(descriptives, letters or {}, **options)
        content, _ = export_figure(fig, "png", dpi=150)
        return base64.b64encode(content).decode("utf-8")
    except Exception:
        return None  # the analysis itself must never fail because of a chart
    finally:
        if fig is not None:
            plt.close(fig)


def figure_to_png(fig):
    buffer = io.BytesIO()
    content, _ = export_figure(fig, "png", dpi=150)
    buffer.write(content)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
