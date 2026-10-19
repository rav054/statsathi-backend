"""Figure composer: grid layout, panel labels, composition and the /figure endpoints.

Run from backend/:  python -m unittest discover -s tests -t .
"""
import io
import json
import unittest

from tests.support import client_and_headers  # must come before other app imports
from tests.test_endpoints_smoke import PNG_SIGNATURE  # noqa: E402

from PIL import Image  # noqa: E402

from app import figure_composer as composer  # noqa: E402
from app.figure_style import MM_PER_INCH, get_preset  # noqa: E402


def image_bytes(width, height, color=(70, 90, 200), fmt="PNG"):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format=fmt)
    return buffer.getvalue()


def upload(count=3, width=800, height=600):
    return [("files", (f"chart{i + 1}.png", image_bytes(width, height), "image/png")) for i in range(count)]


class LayoutTests(unittest.TestCase):
    def test_grid_shape(self):
        self.assertEqual(composer.grid_shape(1), (1, 1))
        self.assertEqual(composer.grid_shape(2), (1, 2))
        self.assertEqual(composer.grid_shape(4), (2, 2))
        self.assertEqual(composer.grid_shape(5), (2, 3))
        self.assertEqual(composer.grid_shape(9), (3, 3))
        self.assertEqual(composer.grid_shape(10), (4, 3))  # never wider than three by default
        self.assertEqual(composer.grid_shape(6, columns=2), (3, 2))
        self.assertEqual(composer.grid_shape(3, columns=9), (1, 3))  # more columns than panels

    def test_panel_labels(self):
        self.assertEqual([composer.panel_label(i) for i in range(3)], ["a", "b", "c"])
        self.assertEqual(composer.panel_label(0, "upper"), "A")
        self.assertEqual(composer.panel_label(1, "lower_paren"), "(b)")
        self.assertEqual(composer.panel_label(2, "upper_paren"), "(C)")
        self.assertEqual(composer.panel_label(3, "none"), "")

    def test_panels_keep_their_aspect_ratio_and_fit_the_figure(self):
        sizes = [(800, 600), (400, 400), (1000, 500), (600, 900)]
        width_in, spacing_in = 7.09, 0.12
        layout = composer.plan_layout(sizes, columns=2, width_in=width_in, spacing_in=spacing_in, fit="width")
        self.assertEqual((layout["rows"], layout["columns"]), (2, 2))
        figure_width, figure_height = layout["figure_size"]
        self.assertAlmostEqual(figure_width, width_in)
        expected_cell = (width_in - 2 * composer.SIDE_MARGIN_IN - spacing_in) / 2
        for panel, (pixel_width, pixel_height) in zip(layout["panels"], sizes):
            with self.subTest(panel=panel["index"]):
                self.assertAlmostEqual(panel["width"] * figure_width, expected_cell)
                drawn_aspect = (panel["width"] * figure_width) / (panel["height"] * figure_height)
                self.assertAlmostEqual(drawn_aspect, pixel_width / pixel_height, places=8)
                self.assertGreaterEqual(panel["bottom"], 0)
                self.assertLessEqual(panel["bottom"] + panel["height"], 1)
        first_row = layout["panels"][:2]
        tops = {round(p["bottom"] + p["height"], 9) for p in first_row}
        self.assertEqual(len(tops), 1)  # panels in a row are aligned at the top
        lefts = [p["left"] for p in first_row]
        self.assertAlmostEqual(lefts[1] - lefts[0], (expected_cell + spacing_in) / figure_width)

    def test_captions_make_the_figure_taller(self):
        sizes = [(600, 400), (600, 400)]
        plain = composer.plan_layout(sizes, columns=2, width_in=7.0, fit="width")
        captioned = composer.plan_layout(sizes, columns=2, width_in=7.0, caption_lines=[2, 1], figure_caption_lines=2,
                                         fit="width")
        self.assertGreater(captioned["figure_size"][1], plain["figure_size"][1])
        self.assertAlmostEqual(captioned["caption_block_in"], 2 * 0.14 + 0.10)

    def test_justified_rows_fill_the_width(self):
        sizes = [(1000, 500), (400, 600), (800, 600)]
        width_in, spacing_in = 7.0, 0.1
        layout = composer.plan_layout(sizes, columns=2, width_in=width_in, spacing_in=spacing_in, fit="row")
        figure_width, figure_height = layout["figure_size"]
        first_row = layout["panels"][:2]
        heights = {round(p["height"] * figure_height, 9) for p in first_row}
        self.assertEqual(len(heights), 1)  # panels in a justified row share one height
        used = sum(p["width"] * figure_width for p in first_row) + spacing_in
        self.assertAlmostEqual(used, width_in - 2 * composer.SIDE_MARGIN_IN, places=8)
        for panel, (pixel_width, pixel_height) in zip(layout["panels"], sizes):
            drawn = (panel["width"] * figure_width) / (panel["height"] * figure_height)
            self.assertAlmostEqual(drawn, pixel_width / pixel_height, places=8)
        last = layout["panels"][2]  # a short last row keeps the cell width instead of stretching
        cell = (width_in - 2 * composer.SIDE_MARGIN_IN - spacing_in) / 2
        self.assertAlmostEqual(last["width"] * figure_width, cell, places=8)

    def test_wrapping(self):
        lines = composer.wrap_text("Effect of nitrogen on grain yield across three seasons at two locations", 2.0, 8)
        self.assertGreater(len(lines), 1)
        self.assertEqual(composer.wrap_text("", 2.0, 8), [])


class CompositionTests(unittest.TestCase):
    def setUp(self):
        self.images = composer.load_images([(f"chart{i}.png", image_bytes(800, 600)) for i in range(3)])

    def test_rendered_size_matches_the_preset(self):
        for preset in ("single_column", "double_column"):
            with self.subTest(preset=preset):
                content, media_type, layout = composer.render(self.images, "png", 300, preset=preset, columns=2)
                self.assertEqual(media_type, "image/png")
                self.assertTrue(content.startswith(PNG_SIGNATURE))
                rendered = Image.open(io.BytesIO(content))
                expected_width = get_preset(preset)["width_mm"] / MM_PER_INCH
                self.assertAlmostEqual(layout["figure_size"][0], expected_width, places=8)
                self.assertAlmostEqual(rendered.width / 300, expected_width, places=1)
                self.assertAlmostEqual(rendered.height / 300, layout["figure_size"][1], places=1)

    def test_other_formats(self):
        for fmt, signature in (("tiff", b"II*\x00"), ("pdf", b"%PDF"), ("svg", b"<?xml")):
            with self.subTest(fmt=fmt):
                content, media_type, _ = composer.render(self.images, fmt, 150, columns=3)
                self.assertTrue(content.startswith(signature), media_type)

    def test_transparency_is_flattened_onto_white(self):
        buffer = io.BytesIO()
        Image.new("RGBA", (100, 80), (255, 0, 0, 0)).save(buffer, format="PNG")
        loaded = composer.load_images([("clear.png", buffer.getvalue())])[0]
        self.assertEqual(loaded.mode, "RGB")
        self.assertEqual(loaded.getpixel((10, 10)), (255, 255, 255))

    def test_invalid_options(self):
        cases = [
            ({}, []),  # no images
            ({}, composer.load_images([("a.png", image_bytes(50, 50))] * (composer.MAX_PANELS + 1))),
            ({"label_style": "roman"}, self.images),
            ({"label_position": "middle"}, self.images),
            ({"background": "blue"}, self.images),
            ({"columns": 9}, self.images),
            ({"fit": "square"}, self.images),
        ]
        for options, images in cases:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    composer.compose(images, **options)

    def test_a_file_that_is_not_an_image(self):
        with self.assertRaises(ValueError):
            composer.load_images([("notes.txt", b"just text, not a picture")])


class FigureEndpointTests(unittest.TestCase):
    def post(self, path, files=None, **form):
        client, headers = client_and_headers()
        return client.post(path, headers=headers, files=files if files is not None else upload(), data=form)

    def test_compose_preview(self):
        response = self.post("/analyze/figure/compose", columns="2", preset="double_column",
                             captions_json=json.dumps(["Yield by treatment", "Residuals", ""]),
                             figure_caption="Figure 1. Field trial results, 2024-25.")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual((body["rows"], body["columns"]), (2, 2))
        self.assertEqual([p["label"] for p in body["panels"]], ["a", "b", "c"])
        self.assertEqual([p["filename"] for p in body["panels"]], ["chart1.png", "chart2.png", "chart3.png"])
        self.assertEqual(body["size_mm"]["width"], 180.0)
        self.assertGreater(body["size_mm"]["height"], 90)
        self.assertTrue(base64_png(body["image"]))

    def test_download_formats(self):
        for fmt, media_type, extension in (("png", "image/png", "png"), ("tiff", "image/tiff", "tiff"),
                                           ("pdf", "application/pdf", "pdf")):
            with self.subTest(fmt=fmt):
                response = self.post("/analyze/figure/download", download_format=fmt, dpi="150", preset="single_column")
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.headers["content-type"], media_type)
                self.assertIn(f"statsathi_figure.{extension}", response.headers["content-disposition"])
                self.assertGreater(len(response.content), 1000)

    def test_single_panel_and_label_overrides(self):
        response = self.post("/analyze/figure/compose", files=upload(1, 1200, 400), label_style="upper_paren",
                             labels_json=json.dumps(["Panel one"]), label_position="inside")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual((body["rows"], body["columns"]), (1, 1))
        self.assertEqual(body["panels"][0]["pixels"], [1200, 400])

    def test_validation(self):
        client, headers = client_and_headers()
        no_files = client.post("/analyze/figure/compose", headers=headers, data={})
        self.assertEqual(no_files.status_code, 422, no_files.text)  # the upload itself is required
        cases = [
            (upload(13), {}),
            (upload(2), {"preset": "billboard"}),
            (upload(2), {"columns": "7"}),
            (upload(2), {"label_style": "roman"}),
            (upload(2), {"captions_json": "not json"}),
            ([("files", ("notes.txt", b"plain text", "text/plain"))], {}),
        ]
        for files, form in cases:
            with self.subTest(form=form, files=len(files)):
                response = self.post("/analyze/figure/compose", files=files, **form)
                self.assertEqual(response.status_code, 400, response.text)

    def test_download_validation(self):
        for form in ({"download_format": "docx"}, {"dpi": "5000"}):
            with self.subTest(form=form):
                self.assertEqual(self.post("/analyze/figure/download", **form).status_code, 400)


def base64_png(text):
    import base64
    return base64.b64decode(text).startswith(PNG_SIGNATURE)


if __name__ == "__main__":
    unittest.main()
