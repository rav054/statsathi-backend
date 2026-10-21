"""Charts exported as PowerPoint slides that stay editable.

A PNG on a slide is a picture of a result: a co-author who wants the bars in their own house colours,
or who spots a mislabelled treatment an hour before a talk, has to come back to StatSathi and export
again. So this module writes a *native* PowerPoint chart instead - the shape carries its own small
worksheet, and PowerPoint lets anyone change the numbers, the colours, the fonts, even the chart type.

Not every figure has a PowerPoint counterpart: there is no box plot or violin in Office, and a biplot's
loading vectors are not series. Those export as the publication-quality image plus the numbers behind
it on the slide, which is still more use than an image alone.
"""

import io

import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.util import Emu, Inches, Pt

from ..figure_style import COLORS, natural_order, palette_colors, pretty_label

MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

# A 16:9 slide, and a chart inset that leaves room for the title above it
SLIDE_WIDTH, SLIDE_HEIGHT = Inches(13.333), Inches(7.5)
CHART_BOX = (Inches(0.9), Inches(1.5), Inches(11.5), Inches(5.4))

# Which plot types become a real chart object, and what PowerPoint calls them
NATIVE_CHARTS = {
    "barplot": XL_CHART_TYPE.COLUMN_CLUSTERED,
    "line": XL_CHART_TYPE.LINE_MARKERS,
    "multiline": XL_CHART_TYPE.LINE_MARKERS,
    "scatter": XL_CHART_TYPE.XY_SCATTER,
    "qqplot": XL_CHART_TYPE.XY_SCATTER,
    "histogram": XL_CHART_TYPE.COLUMN_CLUSTERED,
    "pie": XL_CHART_TYPE.PIE,
}

# Why the remaining types arrive as an image, in words a user can act on
IMAGE_ONLY = {
    "boxplot": "PowerPoint has no box plot, so the figure is placed as an image with its five-number "
               "summary beside it.",
    "violin": "PowerPoint has no violin plot, so the figure is placed as an image with its five-number "
              "summary beside it.",
    "pcabiplot": "A biplot's loading vectors are not a chart series, so the figure is placed as an image.",
}


def _rgb(hex_color):
    return RGBColor.from_string(hex_color.lstrip("#").upper())


def _clean(df, columns):
    """Rows where every column the chart needs has a value."""
    present = [c for c in columns if c]
    return df.dropna(subset=present) if present else df


def _categories(values):
    return natural_order([str(v) for v in pd.unique(values)])


def series_for(df, plot_type, *, x_var=None, y_var=None, hue_var=None, bins=10, y_cols=None):
    """The numbers behind the figure, ready for a chart object.

    Returns a dict with `kind` ("category" or "xy"), the categories or point lists, the series, and the
    axis titles - or None when this plot type has no PowerPoint counterpart.
    """
    if plot_type not in NATIVE_CHARTS:
        return None

    if plot_type == "pie":
        counts = df[x_var].dropna().value_counts()
        if len(counts) > 8:  # the figure itself groups the tail, so the slide must agree
            tail = counts.iloc[7:].sum()
            counts = counts.iloc[:7]
            counts["Others"] = tail
        return {
            "kind": "category",
            "categories": [str(i) for i in counts.index],
            "series": [("Count", [float(v) for v in counts.values])],
            "x_title": pretty_label(x_var),
            "y_title": "Count",
        }

    if plot_type == "histogram":
        values = pd.to_numeric(df[x_var], errors="coerce").dropna()
        if values.empty:
            return None
        counts, edges = np.histogram(values, bins=int(bins or 10))
        centres = [f"{(edges[i] + edges[i + 1]) / 2:.4g}" for i in range(len(counts))]
        return {
            "kind": "category",
            "categories": centres,
            "series": [(pretty_label(x_var), [float(c) for c in counts])],
            "x_title": pretty_label(x_var),
            "y_title": "Frequency",
        }

    if plot_type == "qqplot":
        from scipy import stats

        values = pd.to_numeric(df[x_var], errors="coerce").dropna().sort_values()
        if len(values) < 3:
            return None
        quantiles = stats.norm.ppf((np.arange(1, len(values) + 1) - 0.5) / len(values))
        return {
            "kind": "xy",
            "series": [("Sample", list(zip([float(q) for q in quantiles], [float(v) for v in values])))],
            "x_title": "Theoretical quantiles",
            "y_title": pretty_label(x_var),
        }

    if plot_type == "scatter":
        frame = _clean(df, [x_var, y_var, hue_var])
        xs = pd.to_numeric(frame[x_var], errors="coerce")
        ys = pd.to_numeric(frame[y_var], errors="coerce")
        keep = xs.notna() & ys.notna()
        frame, xs, ys = frame[keep], xs[keep], ys[keep]
        if frame.empty:
            return None
        if hue_var:
            series = [(str(level), list(zip(xs[frame[hue_var].astype(str) == str(level)].astype(float),
                                            ys[frame[hue_var].astype(str) == str(level)].astype(float))))
                      for level in _categories(frame[hue_var])]
        else:
            series = [(pretty_label(y_var), list(zip(xs.astype(float), ys.astype(float))))]
        return {"kind": "xy", "series": series,
                "x_title": pretty_label(x_var), "y_title": pretty_label(y_var)}

    if plot_type == "multiline":
        frame = _clean(df, [x_var])
        categories = [str(v) for v in frame[x_var]]
        series = []
        for column in (y_cols or []):
            values = pd.to_numeric(frame[column], errors="coerce")
            series.append((pretty_label(column), [None if pd.isna(v) else float(v) for v in values]))
        if not series:
            return None
        return {"kind": "category", "categories": categories, "series": series,
                "x_title": pretty_label(x_var), "y_title": "Value"}

    # barplot and line both show the mean of y at each x, which is what seaborn draws
    frame = _clean(df, [x_var, y_var, hue_var])
    frame = frame.assign(_value=pd.to_numeric(frame[y_var], errors="coerce")).dropna(subset=["_value"])
    if frame.empty:
        return None
    categories = _categories(frame[x_var])
    if hue_var:
        series = []
        for level in _categories(frame[hue_var]):
            means = (frame[frame[hue_var].astype(str) == level]
                     .groupby(frame[x_var].astype(str))["_value"].mean())
            series.append((level, [None if c not in means else float(means[c]) for c in categories]))
    else:
        means = frame.groupby(frame[x_var].astype(str))["_value"].mean()
        series = [(pretty_label(y_var), [None if c not in means else float(means[c]) for c in categories])]
    return {"kind": "category", "categories": categories, "series": series,
            "x_title": pretty_label(x_var), "y_title": pretty_label(y_var)}


def five_number_summary(df, x_var, y_var):
    """Minimum, lower quartile, median, upper quartile and maximum per group.

    A box plot has no PowerPoint equivalent, so its slide carries the numbers it was drawn from.
    """
    if not y_var or y_var not in df.columns:
        return None
    frame = df.assign(_value=pd.to_numeric(df[y_var], errors="coerce")).dropna(subset=["_value"])
    if frame.empty:
        return None
    groups = frame.groupby(frame[x_var].astype(str))["_value"] if x_var in df.columns else None
    if groups is None:
        return None
    categories = natural_order([str(name) for name, _ in groups])
    rows = {str(name): values for name, values in groups}
    labels = ("Minimum", "Lower quartile", "Median", "Upper quartile", "Maximum")
    quantiles = (0.0, 0.25, 0.5, 0.75, 1.0)
    return {
        "kind": "category",
        "categories": categories,
        "series": [(label, [float(rows[c].quantile(q)) for c in categories])
                   for label, q in zip(labels, quantiles)],
        "x_title": pretty_label(x_var),
        "y_title": pretty_label(y_var),
    }


def _add_title(slide, text, subtitle=None):
    box = slide.shapes.add_textbox(Inches(0.9), Inches(0.45), Inches(11.5), Inches(0.9))
    frame = box.text_frame
    frame.word_wrap = True
    paragraph = frame.paragraphs[0]
    run = paragraph.add_run()
    run.text = text
    run.font.size = Pt(26)
    run.font.bold = True
    run.font.color.rgb = _rgb(COLORS["text"])
    if subtitle:
        note = frame.add_paragraph()
        run = note.add_run()
        run.text = subtitle
        run.font.size = Pt(12)
        run.font.color.rgb = _rgb(COLORS["muted"])


def _style_chart(chart, colors, *, legend):
    chart.font.size = Pt(13)
    chart.font.color.rgb = _rgb(COLORS["text"])
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout = False
    for index, plot_series in enumerate(chart.series):
        colour = _rgb(colors[index % len(colors)])
        try:
            plot_series.format.fill.solid()
            plot_series.format.fill.fore_color.rgb = colour
        except (NotImplementedError, ValueError):
            pass  # a line series has no fill; its stroke is set below
        try:
            plot_series.format.line.color.rgb = colour
        except (NotImplementedError, ValueError):
            pass


def _add_native_chart(slide, spec, data, palette):
    kind = NATIVE_CHARTS[spec["plot_type"]]
    colors = palette_colors(palette, max(len(data["series"]), 1))

    if data["kind"] == "xy":
        chart_data = XyChartData()
        for name, points in data["series"]:
            group = chart_data.add_series(name)
            for x, y in points:
                group.add_data_point(x, y)
    else:
        chart_data = CategoryChartData()
        chart_data.categories = data["categories"]
        for name, values in data["series"]:
            chart_data.add_series(name, values)

    graphic = slide.shapes.add_chart(kind, *CHART_BOX, chart_data)
    chart = graphic.chart
    _style_chart(chart, colors, legend=len(data["series"]) > 1)

    if spec["plot_type"] != "pie":
        try:
            chart.category_axis.axis_title.text_frame.text = data["x_title"]
            chart.value_axis.axis_title.text_frame.text = data["y_title"]
        except (ValueError, AttributeError):
            pass  # an XY chart names its axes differently across Office versions; the labels are optional
    return chart


def _add_table(slide, data, top):
    """The numbers under the picture, so a slide without a chart object still carries its data."""
    series = data["series"][:4]
    rows = min(len(data["categories"]), 12) + 1
    table = slide.shapes.add_table(rows, len(series) + 1, Inches(8.6), top,
                                   Inches(3.8), Inches(0.3) * rows).table
    table.cell(0, 0).text = data["x_title"] or "Category"
    for column, (name, _) in enumerate(series, start=1):
        table.cell(0, column).text = str(name)
    for row, category in enumerate(data["categories"][:rows - 1], start=1):
        table.cell(row, 0).text = str(category)
        for column, (_, values) in enumerate(series, start=1):
            value = values[row - 1] if row - 1 < len(values) else None
            table.cell(row, column).text = "" if value is None else f"{value:.4g}"
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.text_frame.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(10)
    return table


def build_deck(spec, data, image=None):
    """A one-slide deck: a native chart where PowerPoint has one, otherwise the figure and its numbers.

    `spec` carries plot_type, title and palette; `data` is `series_for(...)`; `image` is PNG bytes.
    """
    deck = Presentation()
    deck.slide_width, deck.slide_height = SLIDE_WIDTH, SLIDE_HEIGHT
    slide = deck.slides.add_slide(deck.slide_layouts[6])  # blank

    title = spec.get("title") or pretty_label(spec["plot_type"])
    note = IMAGE_ONLY.get(spec["plot_type"])

    if data and not note:
        _add_title(slide, title, "Double-click the chart to edit its data, colours or type in PowerPoint.")
        _add_native_chart(slide, spec, data, spec.get("palette"))
    else:
        _add_title(slide, title, note or "Exported from StatSathi.")
        if image:
            slide.shapes.add_picture(io.BytesIO(image), Inches(0.9), Inches(1.5), height=Inches(5.4))
        if data:
            _add_table(slide, data, Inches(1.5))

    out = io.BytesIO()
    deck.save(out)
    out.seek(0)
    return out


def chart_is_editable(plot_type):
    """True when this plot type exports as a chart PowerPoint can edit, rather than a picture."""
    return plot_type in NATIVE_CHARTS and plot_type not in IMAGE_ONLY
