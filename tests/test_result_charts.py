"""Alternative result charts (/analyze/result-chart) and the intervals the estimation plots draw.

Run from backend/:  python -m unittest discover -s tests -t .
"""
import base64
import json
import unittest

from tests.support import post_csv  # must come before other app imports
from tests.test_chart_gallery import field_frame  # noqa: E402
from tests.test_endpoints_smoke import PNG_SIGNATURE  # noqa: E402

import numpy as np  # noqa: E402
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW  # noqa: E402

from app.modules import chart_gallery as cg  # noqa: E402

LETTERS = json.dumps({"T1": "b", "T2": "b", "T3": "a", "T10": "b"})

CASES = [
    # t-tests: one column per group
    dict(chart="raincloud", layout="wide", cols_str="Before,After", p_value="0.012", test_label="Welch's t-Test"),
    dict(chart="boxswarm", layout="wide", cols_str="Before,After", p_value="0.2", test_label="Student's t-Test"),
    dict(chart="meanci", layout="wide", cols_str="Before,After", p_value="0.0004"),
    dict(chart="estimation", layout="wide", cols_str="Before,After", p_value="0.03", test_label="Welch's t-Test", equal_var="false"),
    dict(chart="estimation", layout="wide", cols_str="Before,After", equal_var="true"),
    # one-sample test with the hypothesized mean as a reference line
    dict(chart="raincloud", layout="wide", cols_str="Yield", reference="42", reference_label="Hypothesized mean = 42"),
    dict(chart="meanci", layout="wide", cols_str="Yield", reference="42", reference_label="Hypothesized mean = 42"),
    # paired tests
    dict(chart="paired", layout="long", x_var="Before", y_var="After", p_value="0.001", test_label="Paired t-Test"),
    dict(chart="paired_estimation", layout="long", x_var="Before", y_var="After"),
    # Friedman / Kruskal-Wallis with three columns
    dict(chart="boxswarm", layout="wide_complete", cols_str="Yield,Height,Protein", p_value="0.001", test_label="Friedman Test"),
    # chi-square
    dict(chart="mosaic", layout="long", x_var="Treatment", hue_var="Location"),
    # ANOVA: letters for one-factor designs, factor B for two-factor designs
    dict(chart="raincloud", layout="long", x_var="Treatment", y_var="Yield", letters_json=LETTERS, palette="Oranges"),
    dict(chart="meanci", layout="long", x_var="Treatment", y_var="Yield", letters_json=LETTERS, test_label="One-Way ANOVA"),
    dict(chart="meanci", layout="long", x_var="Treatment", y_var="Yield", hue_var="Location"),
    dict(chart="meanheatmap", layout="long", x_var="Treatment", y_var="Yield", hue_var="Location"),
]


class ResultChartEndpointTests(unittest.TestCase):
    def test_every_result_chart_renders(self):
        df = field_frame()
        for form in CASES:
            with self.subTest(form=form):
                status, res = post_csv("/analyze/result-chart", df, **form)
                self.assertEqual(status, 200, res)
                self.assertTrue(base64.b64decode(res["plot"]).startswith(PNG_SIGNATURE))

    def test_invalid_requests(self):
        df = field_frame()
        cases = [
            dict(chart="violin", layout="wide", cols_str="Yield,Height"),  # not an offered chart
            dict(chart="raincloud", layout="sideways", cols_str="Yield"),
            dict(chart="raincloud", layout="wide", cols_str="Yield,Nope"),
            dict(chart="raincloud", layout="long", x_var="Treatment", y_var="Yield", letters_json="[1, 2]"),
            dict(chart="estimation", layout="long", x_var="Treatment", y_var="Yield"),  # four groups
            dict(chart="estimation", layout="wide", cols_str=""),
        ]
        for form in cases:
            with self.subTest(form=form):
                status, res = post_csv("/analyze/result-chart", df, **form)
                self.assertEqual(status, 400, res)


class EstimationIntervalTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(8)
        self.a = rng.normal(42, 4, 15)
        self.b = rng.normal(39, 6, 11)

    def test_welch_and_student_intervals_match_statsmodels(self):
        compare = CompareMeans(DescrStatsW(self.a), DescrStatsW(self.b))
        for equal_var, usevar in ((False, "unequal"), (True, "pooled")):
            with self.subTest(equal_var=equal_var):
                difference, low, high, _, _ = cg.difference_interval(self.a, self.b, equal_var)
                expected_low, expected_high = compare.tconfint_diff(alpha=0.05, usevar=usevar)
                self.assertAlmostEqual(difference, self.a.mean() - self.b.mean(), places=12)
                self.assertAlmostEqual(low, expected_low, places=10)
                self.assertAlmostEqual(high, expected_high, places=10)

    def test_paired_interval_matches_statsmodels(self):
        a = self.a[:11]
        mean, low, high, _, dof = cg.paired_difference_interval(a, self.b)
        expected_low, expected_high = DescrStatsW(a - self.b).tconfint_mean(alpha=0.05)
        self.assertAlmostEqual(mean, (a - self.b).mean(), places=12)
        self.assertAlmostEqual(low, expected_low, places=10)
        self.assertAlmostEqual(high, expected_high, places=10)
        self.assertEqual(dof, 10)


if __name__ == "__main__":
    unittest.main()
