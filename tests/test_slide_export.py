"""Charts exported to PowerPoint stay editable: a chart object with its own data, not a picture.

Run from backend/:  python -m unittest discover -s tests -t .
"""
import io
import unittest
import zipfile

import pandas as pd

from tests.support import post_file  # must come before other app imports

from pptx import Presentation  # noqa: E402
from pptx.enum.shapes import MSO_SHAPE_TYPE  # noqa: E402

from app.modules import slide_export  # noqa: E402

TRIAL = pd.DataFrame({
    "n_dose": ["N0", "N40", "N80", "N120"] * 4,
    "rep": sum(([f"R{r}"] * 4 for r in range(1, 5)), []),
    "grain_yield": [3.1, 4.0, 4.9, 5.0, 3.2, 3.9, 4.8, 5.1, 2.9, 4.1, 5.0, 4.9, 3.0, 4.0, 4.9, 5.2],
    "plant_height": [62, 74, 86, 95, 64, 73, 88, 96, 61, 75, 87, 94, 63, 74, 86, 97],
})


def tiny_png():
    """A real one-pixel PNG: python-pptx reads the image to size it, so bytes alone will not do."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new('RGB', (4, 3), '#4F46E5').save(buffer, format='PNG')
    return buffer.getvalue()


def deck_from(content):
    return Presentation(io.BytesIO(content))


def only_slide(content):
    deck = deck_from(content)
    assert len(deck.slides) == 1
    return deck.slides[0]


class SeriesTests(unittest.TestCase):
    """The numbers put on the slide are the ones the figure was drawn from."""

    def test_a_bar_chart_carries_the_group_means(self):
        data = slide_export.series_for(TRIAL, "barplot", x_var="n_dose", y_var="grain_yield")
        self.assertEqual(data["categories"], ["N0", "N40", "N80", "N120"])
        self.assertEqual(data["x_title"], "N dose")
        self.assertEqual(data["y_title"], "Grain yield")
        name, values = data["series"][0]
        self.assertEqual(name, "Grain yield")
        self.assertAlmostEqual(values[0], TRIAL[TRIAL.n_dose == "N0"].grain_yield.mean(), places=6)
        self.assertAlmostEqual(values[2], TRIAL[TRIAL.n_dose == "N80"].grain_yield.mean(), places=6)

    def test_a_grouping_variable_becomes_one_series_each(self):
        data = slide_export.series_for(TRIAL, "barplot", x_var="n_dose", y_var="grain_yield", hue_var="rep")
        self.assertEqual([name for name, _ in data["series"]], ["R1", "R2", "R3", "R4"])
        self.assertTrue(all(len(values) == 4 for _, values in data["series"]))

    def test_a_scatter_keeps_every_point_as_an_x_y_pair(self):
        data = slide_export.series_for(TRIAL, "scatter", x_var="plant_height", y_var="grain_yield")
        self.assertEqual(data["kind"], "xy")
        points = data["series"][0][1]
        self.assertEqual(len(points), len(TRIAL))
        self.assertIn((62.0, 3.1), points)

    def test_a_histogram_is_binned_into_counts(self):
        data = slide_export.series_for(TRIAL, "histogram", x_var="plant_height", bins=4)
        self.assertEqual(len(data["categories"]), 4)
        self.assertEqual(sum(data["series"][0][1]), len(TRIAL))
        self.assertEqual(data["y_title"], "Frequency")

    def test_a_pie_groups_its_tail_the_way_the_figure_does(self):
        many = pd.DataFrame({"kind": [f"T{i}" for i in range(12)] * 2})
        data = slide_export.series_for(many, "pie", x_var="kind")
        self.assertEqual(len(data["categories"]), 8)
        self.assertEqual(data["categories"][-1], "Others")
        self.assertEqual(sum(data["series"][0][1]), 24)

    def test_plots_powerpoint_cannot_draw_report_no_series(self):
        for plot_type in ("boxplot", "violin", "pcabiplot"):
            self.assertIsNone(slide_export.series_for(TRIAL, plot_type, x_var="n_dose", y_var="grain_yield"))
            self.assertFalse(slide_export.chart_is_editable(plot_type))
        self.assertTrue(slide_export.chart_is_editable("barplot"))

    def test_the_editable_list_matches_what_the_plots_screen_promises(self):
        # EDITABLE_IN_POWERPOINT in frontend/src/components/PlotsModal.jsx must say the same
        self.assertEqual(sorted(slide_export.NATIVE_CHARTS),
                         sorted(["barplot", "line", "multiline", "scatter", "qqplot",
                                 "histogram", "pie"]))

    def test_a_box_plot_still_gets_the_numbers_it_was_drawn_from(self):
        summary = slide_export.five_number_summary(TRIAL, "n_dose", "grain_yield")
        self.assertEqual(summary["categories"], ["N0", "N40", "N80", "N120"])
        self.assertEqual([name for name, _ in summary["series"]],
                         ["Minimum", "Lower quartile", "Median", "Upper quartile", "Maximum"])
        minimums = dict(zip(summary["categories"], summary["series"][0][1]))
        self.assertAlmostEqual(minimums["N0"], 2.9)


class DeckTests(unittest.TestCase):
    """The file really is a PowerPoint deck, and the chart really is a chart."""

    def test_the_slide_holds_an_editable_chart_with_its_own_worksheet(self):
        data = slide_export.series_for(TRIAL, "barplot", x_var="n_dose", y_var="grain_yield")
        content = slide_export.build_deck({"plot_type": "barplot", "title": "Yield by dose"}, data).getvalue()

        charts = [s for s in only_slide(content).shapes if s.has_chart]
        self.assertEqual(len(charts), 1, "the figure is a chart object, not a picture")
        chart = charts[0].chart
        self.assertEqual([c for c in chart.plots[0].categories], ["N0", "N40", "N80", "N120"])

        # The embedded workbook is what makes the chart editable in PowerPoint
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            embedded = [n for n in archive.namelist() if n.endswith(".xlsx")]
        self.assertTrue(embedded, "an editable chart ships the worksheet behind it")

    def test_the_title_and_the_how_to_edit_note_reach_the_slide(self):
        data = slide_export.series_for(TRIAL, "scatter", x_var="plant_height", y_var="grain_yield")
        content = slide_export.build_deck({"plot_type": "scatter", "title": "Height vs yield"}, data).getvalue()
        text = " ".join(shape.text_frame.text for shape in only_slide(content).shapes if shape.has_text_frame)
        self.assertIn("Height vs yield", text)
        self.assertIn("edit its data", text)

    def test_a_box_plot_slide_carries_the_picture_and_the_table(self):
        summary = slide_export.five_number_summary(TRIAL, "n_dose", "grain_yield")
        png = tiny_png()
        content = slide_export.build_deck({"plot_type": "boxplot", "title": "Spread"}, summary, image=png).getvalue()
        slide = only_slide(content)
        self.assertEqual(len([s for s in slide.shapes if s.has_chart]), 0)
        self.assertEqual(len([s for s in slide.shapes if s.has_table]), 1)
        text = " ".join(s.text_frame.text for s in slide.shapes if s.has_text_frame)
        self.assertIn("no box plot", text, "the slide says why it is not editable")


class DownloadEndpointTests(unittest.TestCase):
    @staticmethod
    def download(**form):
        return post_file("/analyze/plot/download", "data.csv", TRIAL.to_csv(index=False).encode(),
                         "text/csv", **form)

    def test_asking_for_pptx_returns_a_deck_with_a_chart(self):
        response = self.download(plot_type="barplot", x_var="n_dose", y_var="grain_yield",
                                 download_format="pptx")
        self.assertEqual(response.status_code, 200)
        self.assertIn("presentationml", response.headers["content-type"])
        self.assertIn(".pptx", response.headers["content-disposition"])
        self.assertEqual(response.content[:2], b"PK", "a .pptx is a zip archive")
        self.assertEqual(len([s for s in only_slide(response.content).shapes if s.has_chart]), 1)

    def test_a_box_plot_comes_back_as_a_slide_too(self):
        response = self.download(plot_type="boxplot", x_var="n_dose", y_var="grain_yield",
                                 download_format="pptx")
        self.assertEqual(response.status_code, 200)
        slide = only_slide(response.content)
        self.assertEqual(len([s for s in slide.shapes if s.shape_type == MSO_SHAPE_TYPE.PICTURE]), 1)
        self.assertEqual(len([s for s in slide.shapes if s.has_table]), 1)

    def test_the_other_formats_still_download_as_before(self):
        response = self.download(plot_type="barplot", x_var="n_dose", y_var="grain_yield",
                                 download_format="png")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
