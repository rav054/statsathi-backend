"""Shared figure style: palettes, size presets and export formats.

Run from backend/:  python -m unittest discover -s tests -t .
"""
import base64
import io
import json
import re
import unittest
from pathlib import Path

from tests.support import post_csv, post_file  # must come before other app imports
from tests.test_endpoints_smoke import PNG_SIGNATURE, sample_frame  # noqa: E402

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from PIL import Image  # noqa: E402

from app import figure_style as fs  # noqa: E402

SIGNATURES = {
    "png": (PNG_SIGNATURE,),
    "jpeg": (b"\xff\xd8\xff",),
    "tiff": (b"II*\x00", b"MM\x00*"),
    "pdf": (b"%PDF",),
    "eps": (b"%!PS",),
    "svg": (b"<?xml", b"<svg"),
}
FRONTEND_COPY = Path(__file__).resolve().parents[2] / "frontend" / "src" / "theme" / "figure_style.json"


def pixels_for(width_mm, dpi):
    return round(width_mm / 25.4 * dpi)


class FigureStyleDefinitionTests(unittest.TestCase):
    @unittest.skipUnless(FRONTEND_COPY.exists(), "frontend checkout not next to the backend")
    def test_frontend_copy_is_identical(self):
        self.assertEqual(json.loads(FRONTEND_COPY.read_text(encoding="utf-8")), fs.STYLE)

    def test_palettes_have_hex_colours_and_known_colormaps(self):
        for key, palette in fs.PALETTES.items():
            with self.subTest(palette=key):
                self.assertGreaterEqual(len(palette["colors"]), 6)
                for color in palette["colors"]:
                    self.assertRegex(color, r"^#[0-9A-F]{6}$")
                self.assertIn(palette["cmap"], matplotlib.colormaps)

    def test_presets_are_complete(self):
        keys = {"label", "width_mm", "aspect", "font_pt", "title_pt", "line_width", "axis_width", "marker_size", "dpi"}
        for key, preset in fs.PRESETS.items():
            with self.subTest(preset=key):
                self.assertEqual(set(preset), keys)

    def test_palette_colors(self):
        okabe = fs.PALETTES["okabe_ito"]["colors"]
        ten = fs.palette_colors("okabe_ito", 10)
        self.assertEqual(len(ten), 10)
        self.assertEqual(ten[8], okabe[0])
        # Names sent by older screens still resolve through seaborn
        self.assertEqual(fs.palette_colors("Oranges", 3), sns.color_palette("Oranges", 3).as_hex())
        # Unknown names fall back to the default palette
        self.assertEqual(fs.palette_colors("no-such-palette"), fs.PALETTES[fs.STYLE["default_palette"]]["colors"])

    def test_colormap_name(self):
        self.assertEqual(fs.colormap_name("viridis"), "viridis")
        self.assertEqual(fs.colormap_name("okabe_ito"), "cividis")
        self.assertEqual(fs.colormap_name("YlOrRd"), "YlOrRd")
        self.assertEqual(fs.colormap_name("not-a-map"), fs.PALETTES[fs.STYLE["default_palette"]]["cmap"])


class FigureExportTests(unittest.TestCase):
    def draw(self, preset):
        with fs.figure_style(preset, "okabe_ito"):
            fig, ax = plt.subplots(figsize=fs.figure_size(preset))
            ax.plot([1, 2, 3], [2, 1, 3], label="series")
            ax.set_title("Title")
            ax.legend()
            fig.tight_layout()
        return fig

    def test_every_format_has_the_right_file_signature(self):
        fig = self.draw("screen")
        try:
            for fmt, signatures in SIGNATURES.items():
                with self.subTest(fmt=fmt):
                    content, media_type = fs.export_figure(fig, fmt, dpi=100)
                    self.assertTrue(content.startswith(signatures), content[:20])
                    self.assertEqual(media_type, fs.EXPORT_MEDIA_TYPES[fmt])
        finally:
            plt.close(fig)

    def test_single_column_is_85_mm_wide(self):
        fig = self.draw("single_column")
        try:
            content, _ = fs.export_figure(fig, "tiff", dpi=600)
            width, height = Image.open(io.BytesIO(content)).size
        finally:
            plt.close(fig)
        self.assertAlmostEqual(width, pixels_for(85, 600), delta=1)
        self.assertAlmostEqual(height / width, fs.PRESETS["single_column"]["aspect"], delta=0.01)

    def test_vector_text_stays_editable(self):
        fig = self.draw("screen")
        try:
            svg, _ = fs.export_figure(fig, "svg")
        finally:
            plt.close(fig)
        self.assertIn(b"<text", svg)  # text elements, not glyph outlines

    def test_style_block_restores_the_defaults(self):
        before = plt.rcParams["font.size"]
        with fs.figure_style("poster"):
            self.assertEqual(plt.rcParams["font.size"], fs.PRESETS["poster"]["font_pt"])
        self.assertEqual(plt.rcParams["font.size"], before)

    def test_format_names(self):
        self.assertEqual(fs.normalize_format("JPG"), "jpeg")
        self.assertEqual(fs.normalize_format("tif"), "tiff")
        with self.assertRaises(ValueError):
            fs.normalize_format("bmp")

    def test_preview_dpi_keeps_previews_a_similar_pixel_width(self):
        screen_px = pixels_for(fs.PRESETS["screen"]["width_mm"], fs.preview_dpi("screen"))
        column_px = pixels_for(fs.PRESETS["single_column"]["width_mm"], fs.preview_dpi("single_column"))
        self.assertAlmostEqual(column_px / screen_px, 1.0, delta=0.02)


class StyledEndpointTests(unittest.TestCase):
    def download(self, **form):
        content = sample_frame().to_csv(index=False).encode()
        return post_file("/analyze/plot/download", "data.csv", content, "text/csv",
                         plot_type="boxplot", x_var="Treatment", y_var="Yield", **form)

    def test_journal_tiff_download_has_preset_width(self):
        response = self.download(download_format="tiff", dpi="300", preset="single_column", palette="okabe_ito")
        self.assertEqual(response.status_code, 200, response.text[:200])
        self.assertEqual(response.headers["content-type"], "image/tiff")
        width, _ = Image.open(io.BytesIO(response.content)).size
        self.assertAlmostEqual(width, pixels_for(85, 300), delta=1)

    def test_pdf_and_eps_downloads(self):
        for fmt, media_type in (("pdf", "application/pdf"), ("eps", "application/postscript")):
            with self.subTest(fmt=fmt):
                response = self.download(download_format=fmt, preset="double_column")
                self.assertEqual(response.status_code, 200, response.text[:200])
                self.assertEqual(response.headers["content-type"], media_type)
                self.assertTrue(response.content.startswith(SIGNATURES[fmt]))

    def test_unsupported_format_is_rejected(self):
        response = self.download(download_format="bmp")
        self.assertEqual(response.status_code, 400)

    def test_new_palettes_work_in_analysis_modules(self):
        df = sample_frame()
        cases = [
            ("/analyze/plot", dict(plot_type="scatter", x_var="Height", y_var="Yield", hue_var="Treatment", palette="tol_bright", preset="thesis")),
            ("/analyze/anova", dict(test_type="oneway", dep_var="Yield", ind_var1="Treatment", palette="okabe_ito")),
            ("/analyze/pca", dict(columns_str="Yield,Height,Protein", hue_var="Treatment", palette="viridis")),
            ("/analyze/correlation", dict(palette="cividis")),
        ]
        for path, form in cases:
            with self.subTest(path=path):
                status, res = post_csv(path, df, **form)
                self.assertEqual(status, 200, res)
                self.assertTrue(base64.b64decode(res["plot"]).startswith(PNG_SIGNATURE))


if __name__ == "__main__":
    unittest.main()
