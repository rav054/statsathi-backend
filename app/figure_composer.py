"""Compose several chart images into one multi-panel figure, the way journals ask for them.

Panels keep their aspect ratio, sit on a grid of equal-width cells, and are labelled (a), (b), (c)... The figure is
built at a journal column width, so the exported file has the right physical size and resolution.
"""

import io
import math
import string
import textwrap

import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")  # headless rendering for the web server
import matplotlib.pyplot as plt

from .figure_style import COLORS, FONT_STACK, MM_PER_INCH, export_figure, get_preset

MAX_PANELS = 12
MAX_PIXELS = 30_000_000  # per image, before it is scaled into its panel
MAX_COLUMNS = 4
LABEL_STYLES = ("lower", "upper", "lower_paren", "upper_paren", "none")
LABEL_POSITIONS = ("outside", "inside", "none")
BACKGROUNDS = ("white", "transparent")
FITS = ("row", "width")
SIDE_MARGIN_IN = 0.04
CAPTION_GAP_IN = 0.04


def panel_label(index, style="lower"):
    """(a), (b), (c)... in the requested style."""
    if style == "none":
        return ""
    letters = string.ascii_lowercase
    text = letters[index] if index < 26 else letters[index // 26 - 1] + letters[index % 26]
    if style.startswith("upper"):
        text = text.upper()
    return f"({text})" if style.endswith("paren") else text


def grid_shape(count, columns=None):
    """Rows and columns for `count` panels; the default keeps the grid close to square, at most 3 across."""
    if columns:
        columns = max(1, min(int(columns), count))
    else:
        columns = 1 if count == 1 else min(3, math.ceil(math.sqrt(count)))
    return math.ceil(count / columns), columns


def wrap_text(text, width_in, font_pt):
    """Caption text wrapped to the width available, as a list of lines."""
    if not text:
        return []
    characters = max(12, int(width_in * 72 / (font_pt * 0.52)))
    return textwrap.fill(str(text), characters).split("\n")


def row_groups(count, columns):
    return [list(range(r * columns, min((r + 1) * columns, count))) for r in range(math.ceil(count / columns))]


def panel_widths(sizes, columns, width_in, spacing_in, fit="row"):
    """Width in inches of every panel.

    fit="width": every panel is as wide as its cell, so columns line up.
    fit="row":   panels in a row share one height and together fill the width exactly, which wastes no space when the
                 charts have different shapes. A short last row falls back to cell widths so a single panel is not
                 stretched across the whole figure.
    """
    available = width_in - 2 * SIDE_MARGIN_IN
    cell = (available - spacing_in * (columns - 1)) / columns
    widths = [cell] * len(sizes)
    if fit == "row":
        for row in row_groups(len(sizes), columns):
            if len(row) < columns:
                continue
            aspects = [sizes[i][0] / sizes[i][1] for i in row]
            height = (available - spacing_in * (len(row) - 1)) / sum(aspects)
            for index, aspect in zip(row, aspects):
                widths[index] = height * aspect
    return widths


def plan_layout(sizes, columns=None, width_in=7.09, spacing_in=0.12, caption_lines=None, label_position="outside",
                figure_caption_lines=0, line_height_in=0.14, fit="row"):
    """Where each panel goes, and how tall the figure has to be.

    sizes: (width, height) in pixels for each image. Panels keep their aspect ratio, so nothing is stretched.
    Returns positions as fractions of the figure, ready for `add_axes`.
    """
    count = len(sizes)
    rows, columns = grid_shape(count, columns)
    caption_lines = caption_lines or [0] * count
    top_pad = 0.22 if label_position == "outside" else SIDE_MARGIN_IN
    widths = panel_widths(sizes, columns, width_in, spacing_in, fit)
    heights = [widths[i] * (sizes[i][1] / sizes[i][0]) for i in range(count)]
    caption_heights = [lines * line_height_in + (CAPTION_GAP_IN if lines else 0.0) for lines in caption_lines]

    grouped = row_groups(count, columns)
    row_heights = [max(heights[i] + caption_heights[i] for i in row) for row in grouped]
    caption_block = figure_caption_lines * line_height_in + (0.10 if figure_caption_lines else 0.0)
    total = top_pad + sum(row_heights) + spacing_in * (rows - 1) + SIDE_MARGIN_IN + caption_block

    panels, top = [], top_pad
    for row, row_height in zip(grouped, row_heights):
        left = SIDE_MARGIN_IN
        for index in row:
            panels.append({
                "index": index,
                "left": left / width_in,
                "bottom": (total - top - heights[index]) / total,
                "width": widths[index] / width_in,
                "height": heights[index] / total,
                "caption_top": (total - top - heights[index] - CAPTION_GAP_IN) / total,
            })
            left += widths[index] + spacing_in
        top += row_height + spacing_in
    return {"rows": rows, "columns": columns, "figure_size": (width_in, total), "cell_width_in": widths,
            "caption_block_in": caption_block, "panels": panels}


def load_images(streams):
    """Open uploaded images, with the transparency flattened onto white."""
    images = []
    for name, content in streams:
        try:
            image = Image.open(io.BytesIO(content))
            image.load()
        except Exception:
            raise ValueError(f"'{name}' is not an image file (PNG, JPEG or TIFF).")
        if image.width * image.height > MAX_PIXELS:
            raise ValueError(f"'{name}' is too large ({image.width}×{image.height} pixels).")
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGBA")
            flattened = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(flattened, image).convert("RGB")
        else:
            image = image.convert("RGB")
        images.append(image)
    return images


def compose(images, columns=None, preset="double_column", spacing_mm=3.0, label_style="lower",
            label_position="outside", labels=None, captions=None, figure_caption="", background="white", fit="row"):
    """Build the multi-panel figure. Returns (figure, layout)."""
    if not images:
        raise ValueError("Add at least one image.")
    if len(images) > MAX_PANELS:
        raise ValueError(f"At most {MAX_PANELS} panels can be combined into one figure.")
    if label_style not in LABEL_STYLES:
        raise ValueError(f"Panel label style must be one of: {', '.join(LABEL_STYLES)}.")
    if label_position not in LABEL_POSITIONS:
        raise ValueError(f"Panel label position must be one of: {', '.join(LABEL_POSITIONS)}.")
    if background not in BACKGROUNDS:
        raise ValueError("Background must be white or transparent.")
    if columns is not None and not 1 <= int(columns) <= MAX_COLUMNS:
        raise ValueError(f"Choose between 1 and {MAX_COLUMNS} columns.")
    if fit not in FITS:
        raise ValueError("Panel fit must be 'row' (fill each row) or 'width' (equal widths).")

    settings = get_preset(preset)
    width_in = settings["width_mm"] / MM_PER_INCH
    spacing_in = max(0.0, float(spacing_mm)) / MM_PER_INCH
    caption_size = settings["font_pt"]
    label_size = settings["title_pt"]
    line_height_in = caption_size * 1.4 / 72

    sizes = [(image.width, image.height) for image in images]
    _, column_count = grid_shape(len(images), columns)
    widths = panel_widths(sizes, column_count, width_in, spacing_in, fit)  # captions wrap to their own panel
    captions = list(captions or [""] * len(images)) + [""] * len(images)
    wrapped = [wrap_text(captions[i], widths[i], caption_size) for i in range(len(images))]
    figure_lines = wrap_text(figure_caption, width_in - 2 * SIDE_MARGIN_IN, caption_size)

    layout = plan_layout(sizes, columns, width_in, spacing_in, [len(lines) for lines in wrapped], label_position,
                         len(figure_lines), line_height_in, fit)

    figure = plt.figure(figsize=layout["figure_size"])
    if background == "white":
        figure.patch.set_facecolor("white")
    else:
        figure.patch.set_alpha(0.0)

    for panel in layout["panels"]:
        index = panel["index"]
        ax = figure.add_axes((panel["left"], panel["bottom"], panel["width"], panel["height"]))
        ax.imshow(np.asarray(images[index]), interpolation="antialiased", aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        text = (labels[index] if labels and index < len(labels) and labels[index] else panel_label(index, label_style))
        if text and label_position != "none":
            if label_position == "outside":
                ax.text(0, 1.012, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=label_size,
                        fontweight="bold", family=FONT_STACK, color=COLORS["text"])
            else:
                ax.text(0.012, 0.988, text, transform=ax.transAxes, ha="left", va="top", fontsize=label_size,
                        fontweight="bold", family=FONT_STACK, color=COLORS["text"],
                        bbox=dict(boxstyle="square,pad=0.22", facecolor="white", edgecolor="none", alpha=0.85))
        if wrapped[index]:
            figure.text(panel["left"] + panel["width"] / 2, panel["caption_top"], "\n".join(wrapped[index]),
                        ha="center", va="top", fontsize=caption_size, family=FONT_STACK, color=COLORS["text"],
                        linespacing=1.4)
    if figure_lines:
        figure.text(SIDE_MARGIN_IN / width_in, 0.012, "\n".join(figure_lines), ha="left", va="bottom",
                    fontsize=caption_size, family=FONT_STACK, color=COLORS["text"], linespacing=1.4)
    return figure, layout


def render(images, fmt="png", dpi=None, **options):
    """Compose and export in one step. Returns (content, media type, layout)."""
    settings = get_preset(options.get("preset", "double_column"))
    figure, layout = compose(images, **options)
    try:
        content, media_type = export_figure(figure, fmt, dpi or settings["dpi"])
    finally:
        plt.close(figure)
    return content, media_type, layout
