"""How charts look: one colour per measure, gridlines only across the measured axis, readable labels.

Run from backend/:  python -m unittest discover -s tests -t .
"""
import io
import math
import unittest
from unittest import mock

import pandas as pd

from tests.support import post_csv  # must come before other app imports

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from app import figure_style as fs  # noqa: E402
from app.figure_style import natural_order, pretty_label  # noqa: E402
from app.modules.mean_chart import mean_separation_figure, mean_separation_png, top_group  # noqa: E402
from app.modules.plots import generate_plot_internal  # noqa: E402

TRIAL = pd.DataFrame({
    "n_dose": ["N0", "N40", "N80", "N120"] * 4,
    "rep": sum(([f"R{r}"] * 4 for r in range(1, 5)), []),
    "grain_yield": [3.1, 4.0, 4.9, 5.0, 3.2, 3.9, 4.8, 5.1, 2.9, 4.1, 5.0, 4.9, 3.0, 4.0, 4.9, 5.2],
    "plant_height": [62, 74, 86, 95, 64, 73, 88, 96, 61, 75, 87, 94, 63, 74, 86, 97],
})


class _Upload:
    """The little of UploadFile that the chart code reads."""

    def __init__(self, frame):
        self.filename = "trial.csv"
        self.content_type = "text/csv"
        self.file = io.BytesIO(frame.to_csv(index=False).encode())


def draw(frame=None, **options):
    """Render a chart and hand back its Axes, which the renderer would otherwise close."""
    plt.close("all")
    with mock.patch.object(plt, "close"):
        generate_plot_internal(file=_Upload(TRIAL if frame is None else frame), **options)
    figure = plt.figure(plt.get_fignums()[-1])
    return figure.axes[0]


class ChartColourTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_default_palette_is_the_shared_one(self):
        self.assertEqual(fs.palette_colors(None)[0], fs.PALETTES["statsathi"]["colors"][0])
        # The brand colormaps exist, so a heatmap never falls back to a rainbow
        self.assertIn(fs.colormap_name(None), matplotlib.colormaps)
        self.assertIn(fs.diverging_cmap(None), matplotlib.colormaps)

    def test_bars_and_boxes_use_one_colour_until_a_grouping_variable_is_given(self):
        status, body = post_csv("/analyze/plot", TRIAL, plot_type="barplot", x_var="n_dose", y_var="grain_yield")
        self.assertEqual(status, 200)
        self.assertIn("plot", body)

        ax = draw(plot_type="boxplot", x_var="n_dose", y_var="grain_yield")
        boxes = [p for p in ax.patches if p.get_facecolor()[3] > 0]
        self.assertGreaterEqual(len(boxes), 4)
        self.assertEqual(len({tuple(round(c, 3) for c in p.get_facecolor()) for p in boxes}), 1)

        grouped = draw(plot_type="boxplot", x_var="n_dose", y_var="grain_yield", hue_var="rep")
        colours = {tuple(round(c, 3) for c in p.get_facecolor()) for p in grouped.patches if p.get_facecolor()[3] > 0}
        self.assertGreater(len(colours), 1, "a grouping variable is what makes colour mean something")

    def test_gridlines_only_cross_the_measured_axis(self):
        bars = draw(plot_type="barplot", x_var="n_dose", y_var="grain_yield")
        self.assertTrue(any(line.get_visible() for line in bars.yaxis.get_gridlines()))
        self.assertFalse(any(line.get_visible() for line in bars.xaxis.get_gridlines()))

        scatter = draw(plot_type="scatter", x_var="plant_height", y_var="grain_yield")
        self.assertTrue(any(line.get_visible() for line in scatter.xaxis.get_gridlines()))

        off = draw(plot_type="barplot", x_var="n_dose", y_var="grain_yield", show_grid=False)
        self.assertFalse(any(line.get_visible() for line in off.yaxis.get_gridlines()))


class ChartLabelTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_column_names_are_written_as_words(self):
        self.assertEqual(pretty_label("grain_yield"), "Grain yield")
        self.assertEqual(pretty_label("plant.height"), "Plant height")
        self.assertEqual(pretty_label("PC1"), "PC1")
        self.assertEqual(pretty_label("DAS"), "DAS")
        self.assertEqual(pretty_label(""), "")

        ax = draw(plot_type="scatter", x_var="plant_height", y_var="grain_yield")
        self.assertEqual(ax.get_xlabel(), "Plant height")
        self.assertEqual(ax.get_ylabel(), "Grain yield")

    def test_a_label_the_user_typed_is_left_alone(self):
        ax = draw(plot_type="scatter", x_var="plant_height", y_var="grain_yield",
                  xlabel="Plant height (cm)", ylabel="Grain yield (t/ha)")
        self.assertEqual(ax.get_xlabel(), "Plant height (cm)")
        self.assertEqual(ax.get_ylabel(), "Grain yield (t/ha)")


class UnknownChartTests(unittest.TestCase):
    def test_an_unknown_chart_type_says_so_instead_of_drawing_empty_axes(self):
        status, body = post_csv("/analyze/plot", TRIAL, plot_type="bar", x_var="n_dose", y_var="grain_yield")
        self.assertEqual(status, 400)
        self.assertIn("barplot", body["detail"])


if __name__ == "__main__":
    unittest.main()


class MeanSeparationChartTests(unittest.TestCase):
    """The chart a trial is published with: means, error bars and post-hoc letters."""

    DESCRIPTIVES = {
        "N0": {"mean": 3.1, "se": 0.14, "n": 4},
        "N40": {"mean": 3.9, "se": 0.12, "n": 4},
        "N80": {"mean": 5.0, "se": 0.10, "n": 4},
        "N120": {"mean": 5.1, "se": 0.13, "n": 4},
    }
    LETTERS = {"N0": "c", "N40": "b", "N80": "a", "N120": "a"}

    def tearDown(self):
        plt.close("all")

    def test_letters_sit_above_the_bars_in_reading_order(self):
        fig = mean_separation_figure(self.DESCRIPTIVES, self.LETTERS, title="Yield", xlabel="n_dose",
                                     ylabel="grain_yield")
        ax = fig.axes[0]
        self.assertEqual([t.get_text() for t in ax.get_xticklabels()], ["N0", "N40", "N80", "N120"])
        marks = {t.get_text(): t.get_position() for t in ax.texts}
        self.assertEqual(set(marks), {"a", "b", "c"})
        # every letter clears the bar and its error bar
        for label, letter in (("N0", "c"), ("N40", "b")):
            top = self.DESCRIPTIVES[label]["mean"] + self.DESCRIPTIVES[label]["se"]
            self.assertGreater(marks[letter][1], top)
        self.assertEqual(ax.get_xlabel(), "N dose")
        self.assertEqual(ax.get_ylabel(), "Grain yield")

    def test_colour_marks_the_top_group_not_the_rank(self):
        self.assertEqual(top_group(self.LETTERS), {"N80", "N120"})
        fig = mean_separation_figure(self.DESCRIPTIVES, self.LETTERS, title="Yield", xlabel="dose", ylabel="yield")
        fills = [bar.get_facecolor() for bar in fig.axes[0].patches]
        self.assertEqual(len({tuple(f) for f in fills}), 2, "the top group is filled, the rest are tinted")
        self.assertEqual(tuple(fills[2]), tuple(fills[3]))

    def test_the_anova_chart_carries_the_letters_the_table_reports(self):
        status, body = post_csv("/analyze/anova", TRIAL, test_type="oneway", dep_var="grain_yield",
                                ind_var1="n_dose", posthoc_method="duncan")
        self.assertEqual(status, 200)
        self.assertTrue(body["posthoc_letters"])
        self.assertTrue(body["plot"])
        self.assertIn("plot_classic", body)
        self.assertNotEqual(body["plot"], body["plot_classic"])

    def test_a_chart_problem_never_fails_the_analysis(self):
        self.assertIsNone(mean_separation_png({}, {}, title="t", xlabel="x", ylabel="y"))
        broken = {"A": {"mean": "not a number", "se": 0.1}, "B": {"mean": 2.0, "se": 0.1}}
        self.assertIsNone(mean_separation_png(broken, {"A": "a", "B": "b"}, title="t", xlabel="x", ylabel="y"))


class CategoryLabelTests(unittest.TestCase):
    """Long treatment names must never be printed on top of each other."""

    def tearDown(self):
        plt.close("all")

    @staticmethod
    def means(labels):
        return {label: {"mean": 40 + i, "se": 0.8, "n": 4} for i, label in enumerate(labels)}

    @staticmethod
    def overlapping(ax, fig):
        """True when neighbouring tick labels would print on top of each other.

        Flat labels collide when their boxes meet; turned labels run alongside each other instead, so
        what matters there is the gap across the slant, not the width.
        """
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ticks = [t for t in ax.get_xticklabels() if t.get_text()]
        if not ticks:
            return False
        boxes = sorted((t.get_window_extent(renderer) for t in ticks), key=lambda b: b.x0)
        angle = math.radians(ticks[0].get_rotation())
        if angle:
            # Turned labels sit on parallel lines: they clash when the gap across the slant is
            # narrower than a line of text
            slot = ax.get_window_extent(renderer).width / len(ticks)
            line_height = ticks[0].get_fontsize() * fig.dpi / 72 * 1.15
            return slot * math.sin(angle) < line_height
        return any(b.x1 > boxes[i + 1].x0 + 0.5 for i, b in enumerate(boxes[:-1]))

    def test_short_names_are_left_flat(self):
        labels = ["N0", "N40", "N80", "N120"]
        fig = mean_separation_figure(self.means(labels), {}, title="t", xlabel="dose", ylabel="yield")
        ax = fig.axes[0]
        self.assertEqual([t.get_rotation() for t in ax.get_xticklabels()], [0] * len(labels))
        self.assertFalse(self.overlapping(ax, fig))

    def test_factorial_names_are_turned_rather_than_overlapped(self):
        labels = [f"{level} / Cultivar_{n}" for level in ("Low", "Medium", "High") for n in (1, 2, 3)]
        fig = mean_separation_figure(self.means(labels), {}, title="t", xlabel="Treatment", ylabel="yield")
        ax = fig.axes[0]
        self.assertTrue(all(t.get_rotation() > 0 for t in ax.get_xticklabels()))
        self.assertFalse(self.overlapping(ax, fig))

    def test_very_long_names_lay_the_bars_on_their_side(self):
        labels = [f"Irrigated {when} / Vermicompost {n}" for n, when in
                  enumerate(("weekly", "fortnightly", "monthly", "at flowering", "at tillering", "at booting"), 1)]
        letters = {label: "a" for label in labels}
        fig = mean_separation_figure(self.means(labels), letters, title="t", xlabel="Treatment", ylabel="Grain yield")
        ax = fig.axes[0]
        self.assertEqual(ax.get_ylabel(), "Treatment", "the names moved to the y axis")
        self.assertEqual(ax.get_xlabel(), "Grain yield")
        self.assertEqual([t.get_text() for t in ax.get_yticklabels()], natural_order(labels))
        self.assertGreaterEqual(fig.get_size_inches()[1], 1.6 + 0.34 * len(labels) - 0.01)
        self.assertEqual(len(ax.patches), len(labels))

    def test_the_plots_screen_turns_long_category_labels_too(self):
        frame = pd.DataFrame({
            "Treatment": [f"{level} / Cultivar_{n}" for level in ("Low", "Medium", "High") for n in (1, 2, 3)] * 2,
            "yield": [40 + i * 0.7 for i in range(18)],
        })
        ax = draw(frame=frame, plot_type="boxplot", x_var="Treatment", y_var="yield")
        self.assertTrue(all(t.get_rotation() > 0 for t in ax.get_xticklabels() if t.get_text()))
