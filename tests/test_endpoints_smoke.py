"""Smoke tests for endpoints without numeric reference tests: dataset helpers and chart rendering.

Run from backend/:  python -m unittest discover -s tests -t .
"""
import base64
import io
import unittest

from tests.support import post_csv, post_file  # must come before other app imports

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def sample_frame():
    rng = np.random.default_rng(21)
    return pd.DataFrame({
        "Treatment": np.repeat(["T1", "T2", "T3"], 8),
        "Yield": rng.normal(40, 4, 24).round(2),
        "Height": rng.normal(90, 8, 24).round(1),
        "Protein": rng.normal(12, 1.5, 24).round(2),
    })


class DataIoSmokeTests(unittest.TestCase):
    def test_columns(self):
        status, res = post_csv("/analyze/columns", sample_frame())
        self.assertEqual(status, 200, res)
        self.assertEqual(res["columns"], ["Treatment", "Yield", "Height", "Protein"])
        self.assertEqual(res["numeric_columns"], ["Yield", "Height", "Protein"])

    def test_preview_serializes_missing_values_as_null(self):
        df = sample_frame()
        df.loc[3, "Height"] = np.nan
        status, res = post_csv("/analyze/preview", df)
        self.assertEqual(status, 200, res)
        self.assertEqual(res["total_rows"], 24)
        self.assertIsNone(res["data"][3]["Height"])

    def test_upload_edited_data(self):
        status, res = post_csv("/analyze/upload-edited-data", sample_frame())
        self.assertEqual(status, 200, res)
        self.assertEqual(res["rows"], 24)

    def test_excel_to_csv_round_trip(self):
        df = sample_frame()
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False)
        response = post_file("/analyze/excel-to-csv", "field.xlsx", buffer.getvalue(),
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.headers["content-type"].startswith("text/csv"))
        pd.testing.assert_frame_equal(pd.read_csv(io.StringIO(response.text)), df)


    def test_analysis_endpoints_accept_excel_uploads(self):
        buffer = io.BytesIO()
        sample_frame().to_excel(buffer, index=False)
        xlsx = buffer.getvalue()
        media = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        response = post_file("/analyze/descriptive", "field.xlsx", xlsx, media, columns_str="Yield")
        self.assertEqual(response.status_code, 200, response.text[:300])
        self.assertEqual(response.json()["variables"]["Yield"]["overall"]["n"], 24)
        response = post_file("/analyze/anova", "field.xlsx", xlsx, media,
                             test_type="oneway", dep_var="Yield", ind_var1="Treatment")
        self.assertEqual(response.status_code, 200, response.text[:300])


class PlotSmokeTests(unittest.TestCase):
    CASES = [
        dict(plot_type="boxplot", x_var="Treatment", y_var="Yield"),
        dict(plot_type="histogram", x_var="Yield", kde="true"),
        dict(plot_type="scatter", x_var="Height", y_var="Yield", fit_reg="true"),
        dict(plot_type="line", x_var="Treatment", y_var="Yield", errorbar_toggle="true", errorbar_type="se"),
        dict(plot_type="barplot", x_var="Treatment", y_var="Yield", errorbar_toggle="true", errorbar_type="ci"),
        dict(plot_type="violin", x_var="Treatment", y_var="Yield"),
        dict(plot_type="qqplot", x_var="Yield"),
        dict(plot_type="pie", x_var="Treatment"),
        dict(plot_type="multiline", x_var="Height", y_vars_str="Yield,Protein"),
        dict(plot_type="pcabiplot", x_var="Yield", y_vars_str="Yield,Height,Protein", hue_var="Treatment"),
    ]

    def test_every_plot_type_renders_a_png(self):
        df = sample_frame()
        for form in self.CASES:
            with self.subTest(plot=form["plot_type"]):
                status, res = post_csv("/analyze/plot", df, **form)
                self.assertEqual(status, 200, res)
                self.assertTrue(base64.b64decode(res["plot"]).startswith(PNG_SIGNATURE))

    def test_download_png_and_svg(self):
        content = sample_frame().to_csv(index=False).encode()
        png = post_file("/analyze/plot/download", "data.csv", content, "text/csv",
                        plot_type="boxplot", x_var="Treatment", y_var="Yield", download_format="png", dpi="300")
        self.assertEqual(png.status_code, 200, png.text[:200])
        self.assertEqual(png.headers["content-type"], "image/png")
        self.assertTrue(png.content.startswith(PNG_SIGNATURE))

        svg = post_file("/analyze/plot/download", "data.csv", content, "text/csv",
                        plot_type="scatter", x_var="Height", y_var="Yield", download_format="svg")
        self.assertEqual(svg.status_code, 200, svg.text[:200])
        self.assertTrue(svg.headers["content-type"].startswith("image/svg+xml"))
        self.assertIn(b"<svg", svg.content[:500])

    def test_correlation_heatmap_respects_dpi(self):
        from PIL import Image

        df = sample_frame()[["Yield", "Height", "Protein"]]
        sizes = {}
        for dpi in ("150", "300"):
            status, res = post_csv("/analyze/correlation", df, dpi=dpi)
            self.assertEqual(status, 200, res)
            sizes[dpi] = Image.open(io.BytesIO(base64.b64decode(res["plot"]))).size
        self.assertAlmostEqual(sizes["300"][0] / sizes["150"][0], 2.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
