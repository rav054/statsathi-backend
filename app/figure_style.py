"""Shared figure style for Matplotlib charts: palettes, journal size presets and export.

The definitions live in assets/figure_style.json; the frontend keeps an identical copy
(frontend/src/theme/figure_style.json) so the interactive Plotly charts match the server figures.
"""

import io
import json
import re
import logging
import threading
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering for the web server
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
from matplotlib import font_manager

STYLE = json.loads((Path(__file__).parent / "assets" / "figure_style.json").read_text(encoding="utf-8"))
PALETTES = STYLE["palettes"]
PRESETS = STYLE["presets"]
COLORS = STYLE["colors"]
MM_PER_INCH = 25.4

EXPORT_MEDIA_TYPES = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "tiff": "image/tiff",
    "svg": "image/svg+xml",
    "pdf": "application/pdf",
    "eps": "application/postscript",
}
_FORMAT_ALIASES = {"jpg": "jpeg", "tif": "tiff"}

# Missing fonts in the fallback stack are expected on Linux hosts; don't flood the log.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

_installed_fonts = {f.name for f in font_manager.fontManager.ttflist}
FONT_STACK = [name for name in STYLE["font"]["matplotlib"] if name in _installed_fonts] or ["DejaVu Sans"]

# rcParams are process-wide, so figures rendered with a non-default preset are drawn one at a time.
_STYLE_LOCK = threading.RLock()


# Brand colormaps: one hue light->dark for magnitude, two hues around a neutral grey for polarity.
# A rainbow colormap encodes nothing, so neither of these is one.
SEQUENTIAL_STEPS = ["#F2F3FE", "#D9DCFA", "#B5B9F2", "#8A8BE9", "#6366F1", "#4F46E5", "#3730A3", "#272066"]
# Discrete steps for ordinal categories (bins, ranks): every step clears the white surface, unlike the pale
# end of the continuous map above, which is only ever read as a filled cell beside a colour bar.
ORDINAL_STEPS = ["#A5A8F2", "#8486EB", "#6366F1", "#4F46E5", "#3A32AB", "#272066"]
DIVERGING_STEPS = ["#7C2D12", "#C2410C", "#EA580C", "#F7B486", "#F1F2F6", "#A5B4FC", "#6366F1", "#4F46E5", "#312E81"]


def _register_colormaps():
    from matplotlib.colors import LinearSegmentedColormap

    for name, steps in (("statsathi_seq", SEQUENTIAL_STEPS), ("statsathi_div", DIVERGING_STEPS)):
        if name in matplotlib.colormaps:
            continue
        cmap = LinearSegmentedColormap.from_list(name, steps)
        matplotlib.colormaps.register(cmap=cmap, name=name)
        matplotlib.colormaps.register(cmap=cmap.reversed(), name=f"{name}_r")


_register_colormaps()


def pretty_label(name):
    """A column name as a reader sees it: 'grain_yield' -> 'Grain yield'; 'PC1' and 'DAS' stay as they are."""
    text = str(name).replace("_", " ").replace(".", " ").strip()
    if not text:
        return str(name)
    words = text.split()
    if all(word.isupper() or any(c.isdigit() for c in word) for word in words):
        return " ".join(words)
    joined = " ".join(words)
    return joined[0].upper() + joined[1:]


def natural_order(labels):
    """Labels in the order a person reads them: N0, N40, N80, N120 - not N0, N120, N40."""
    def key(label):
        return [(int(part), "") if part.isdigit() else (float("inf"), part.lower())
                for part in re.split(r"(\d+)", str(label)) if part != ""]
    return sorted(labels, key=key)



def category_label_fit(fig, ax, labels=None):
    """How category labels fit along the x axis, measured rather than guessed.

    Returns "flat" when they fit as they are, "rotate" when turning them 30 degrees is enough, and
    "sideways" when even that would overlap, which is the signal to draw the bars horizontally.
    """
    ticks = [t for t in ax.get_xticklabels() if t.get_text()]
    if not ticks:
        return "flat"
    try:
        renderer = fig.canvas.get_renderer()
    except AttributeError:  # a backend without a renderer: fall back to counting characters
        longest = max(len(t.get_text()) for t in ticks)
        return "flat" if longest <= 8 else "rotate" if longest <= 14 else "sideways"
    widths = [t.get_window_extent(renderer).width for t in ticks]
    slot = ax.get_window_extent(renderer).width / max(len(ticks), 1)
    widest = max(widths)
    if widest <= slot * 0.95:
        return "flat"
    # A label turned 30 degrees needs about cos(30) of its width along the axis
    if widest * 0.87 <= slot * 1.9:
        return "rotate"
    return "sideways"


def rotate_category_labels(ax, degrees=30):
    """Turn the x tick labels so they stop overlapping, anchored at their right-hand end."""
    for label in ax.get_xticklabels():
        label.set_rotation(degrees)
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")


def get_preset(name=None):
    return PRESETS.get(name) or PRESETS[STYLE["default_preset"]]


def palette_colors(name=None, n=None):
    """Hex colours for a StatSathi palette key.

    Older screens send seaborn / matplotlib palette names ("Oranges", "coolwarm", ...), which are
    still accepted; unknown names fall back to the default palette.
    """
    if name in PALETTES or not name:
        colors = PALETTES[name or STYLE["default_palette"]]["colors"]
        if n is None:
            return list(colors)
        return [colors[i % len(colors)] for i in range(n)]
    try:
        return sns.color_palette(name, n).as_hex()
    except (ValueError, TypeError):
        return palette_colors(None, n)


def colormap_name(name=None):
    """Continuous colormap for a palette key, or the name itself if it is already a colormap."""
    if name in PALETTES:
        return PALETTES[name]["cmap"]
    if name and name in matplotlib.colormaps:
        return name
    return PALETTES[STYLE["default_palette"]]["cmap"]


def diverging_cmap(name=None):
    """Two-hue colormap with a neutral middle, for values centred on zero (correlations, differences)."""
    palette = PALETTES.get(name) or PALETTES[STYLE["default_palette"]]
    return palette.get("diverging_cmap") or "statsathi_div"


def figure_size(preset=None, aspect=None):
    """Figure size in inches for a preset; `aspect` (height / width) overrides the preset's ratio."""
    p = get_preset(preset)
    width = p["width_mm"] / MM_PER_INCH
    return width, width * (aspect if aspect else p["aspect"])


def preview_dpi(preset=None, screen_dpi=150):
    """DPI that renders any preset at about the pixel width of the screen preset, for on-screen previews."""
    ratio = PRESETS["screen"]["width_mm"] / get_preset(preset)["width_mm"]
    return max(72, min(600, round(screen_dpi * ratio)))


def rc_params(preset=None, palette=None):
    p = get_preset(preset)
    font = p["font_pt"]
    small = round(font * 0.9, 1)
    return {
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK + ["DejaVu Sans"],
        "font.size": font,
        "axes.titlesize": p["title_pt"],
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": font * 0.9,
        "axes.labelsize": font,
        "axes.labelpad": font * 0.4,
        "xtick.labelsize": small,
        "ytick.labelsize": small,
        "legend.fontsize": small,
        "legend.title_fontsize": font,
        "legend.frameon": False,
        "text.color": COLORS["text"],
        "axes.labelcolor": COLORS["text"],
        "axes.edgecolor": COLORS["axis"],
        "axes.facecolor": COLORS["background"],
        "axes.linewidth": p["axis_width"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": COLORS["grid"],
        "grid.linestyle": "-",
        "grid.linewidth": p["axis_width"] * 0.8,
        "axes.prop_cycle": cycler(color=palette_colors(palette)),
        "image.cmap": colormap_name(palette),
        "xtick.color": COLORS["axis"],
        "ytick.color": COLORS["axis"],
        "xtick.labelcolor": COLORS["muted"],
        "ytick.labelcolor": COLORS["muted"],
        "xtick.major.width": p["axis_width"],
        "ytick.major.width": p["axis_width"],
        "xtick.major.size": font * 0.35,
        "ytick.major.size": font * 0.35,
        "lines.linewidth": p["line_width"],
        "lines.markersize": p["marker_size"],
        "patch.linewidth": p["axis_width"],
        "figure.facecolor": COLORS["background"],
        "figure.figsize": figure_size(preset),
        "savefig.facecolor": COLORS["background"],
        "savefig.dpi": p["dpi"],
        # Keep text as text in vector files so journals and editors can edit it.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }


@contextmanager
def figure_style(preset=None, palette=None):
    """Apply the StatSathi style for a size preset and palette to figures created and saved inside the block."""
    with _STYLE_LOCK, plt.rc_context(rc_params(preset, palette)):
        yield


def normalize_format(fmt):
    """Canonical export format name; raises ValueError for formats that cannot be exported."""
    fmt = (fmt or "png").strip().lower()
    fmt = _FORMAT_ALIASES.get(fmt, fmt)
    if fmt not in EXPORT_MEDIA_TYPES:
        raise ValueError(f"Unsupported figure format '{fmt}'. Use one of: {', '.join(EXPORT_MEDIA_TYPES)}.")
    return fmt


def export_figure(fig, fmt="png", dpi=None):
    """Render a figure to bytes. Returns (content, media_type).

    The figure keeps its exact size (no bbox cropping), so a single-column preset really is 85 mm wide.
    """
    fmt = normalize_format(fmt)
    options = {"format": fmt, "dpi": dpi or plt.rcParams["savefig.dpi"]}
    if fmt == "tiff":
        options["pil_kwargs"] = {"compression": "tiff_lzw"}
    elif fmt == "jpeg":
        options["pil_kwargs"] = {"quality": 95}
    buf = io.BytesIO()
    fig.savefig(buf, **options)
    return buf.getvalue(), EXPORT_MEDIA_TYPES[fmt]


# Every Matplotlib figure in the app uses the default StatSathi style unless a block overrides it.
matplotlib.rcParams.update(rc_params())
