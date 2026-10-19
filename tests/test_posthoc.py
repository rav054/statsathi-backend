"""Reference tests for the post-hoc engine (app/posthoc.py).

Run from backend/:  python -m unittest discover -s tests -t .
"""
import unittest
from itertools import combinations
from math import sqrt

import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from app.posthoc import (
    compact_letter_groups,
    pairwise_comparisons,
    pooled_pair_error,
    splitplot_pair_error,
    studentized_range_cdf,
    welch_pair_error,
)

# Corrected Duncan significant studentized ranges, alpha = 0.05 (Harter 1960), p = 2..6
PUBLISHED_DUNCAN_SSR = {
    10: [3.151, 3.293, 3.376, 3.430, 3.465],
    20: [2.950, 3.097, 3.190, 3.255, 3.303],
    60: [2.829, 2.976, 3.073, 3.143, 3.198],
}


def one_way_summary(groups):
    labels = list(groups)
    ns = [len(groups[g]) for g in labels]
    means = [float(np.mean(groups[g])) for g in labels]
    df_error = sum(ns) - len(labels)
    mse = sum(((v - v.mean()) ** 2).sum() for v in groups.values()) / df_error
    return labels, ns, means, mse, df_error


class StudentizedRangeTests(unittest.TestCase):
    def test_cdf_matches_scipy(self):
        grid = [(q, k, df) for q in (0.3, 1.0, 2.0, 2.8, 3.5, 4.5, 6.0)
                for k in (2, 3, 6, 12, 30) for df in (2, 4, 9, 20, 37.3, 120, 5000)]
        q, k, df = (np.array(v) for v in zip(*grid))
        reference = np.array([stats.studentized_range.cdf(*point) for point in grid])
        np.testing.assert_allclose(studentized_range_cdf(q, k, df), reference, atol=1e-8)


class PairwiseMethodTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        sizes = [4, 6, 5, 7, 5]
        self.groups = {f"G{i}": rng.normal(10 + 0.6 * i, 1.0, n) for i, n in enumerate(sizes)}

    def test_tukey_kramer_matches_scipy_and_statsmodels(self):
        labels, ns, means, mse, df_error = one_way_summary(self.groups)
        mine = pairwise_comparisons(labels, means, pooled_pair_error(ns, mse, df_error), "tukey")

        exact = stats.tukey_hsd(*self.groups.values())
        y = np.concatenate(list(self.groups.values()))
        g = np.repeat(labels, ns)
        sm = pairwise_tukeyhsd(y, g)
        for n, (i, j) in enumerate(combinations(range(len(labels)), 2)):
            self.assertAlmostEqual(mine[n]["p_value"], exact.pvalue[i, j], places=7)
            self.assertAlmostEqual(mine[n]["mean_diff"], sm.meandiffs[n], places=10)
            self.assertEqual(mine[n]["significant"], bool(sm.reject[n]))

    def test_two_groups_tukey_equals_student_t(self):
        a, b = self.groups["G0"], self.groups["G3"]
        labels, ns, means, mse, df_error = one_way_summary({"a": a, "b": b})
        mine = pairwise_comparisons(labels, means, pooled_pair_error(ns, mse, df_error), "tukey")
        self.assertAlmostEqual(mine[0]["p_value"], stats.ttest_ind(a, b).pvalue, places=8)

    def test_two_groups_games_howell_equals_welch_t(self):
        a, b = self.groups["G1"], self.groups["G4"] * 3
        labels = ["a", "b"]
        pair_error = welch_pair_error([len(a), len(b)], [a.var(ddof=1), b.var(ddof=1)])
        mine = pairwise_comparisons(labels, [a.mean(), b.mean()], pair_error, "games_howell")
        self.assertAlmostEqual(mine[0]["p_value"], stats.ttest_ind(a, b, equal_var=False).pvalue, places=8)

    def test_lsd_matches_pooled_t_and_bonferroni_multiplies(self):
        labels, ns, means, mse, df_error = one_way_summary(self.groups)
        provider = pooled_pair_error(ns, mse, df_error)
        lsd = pairwise_comparisons(labels, means, provider, "lsd")
        bonf = pairwise_comparisons(labels, means, provider, "bonferroni")
        m = len(lsd)
        for n, (i, j) in enumerate(combinations(range(len(labels)), 2)):
            t = (means[j] - means[i]) / sqrt(mse * (1 / ns[i] + 1 / ns[j]))
            self.assertAlmostEqual(lsd[n]["p_value"], 2 * stats.t.sf(abs(t), df_error), places=12)
            self.assertAlmostEqual(bonf[n]["p_value"], min(1.0, lsd[n]["p_value"] * m), places=12)

    def test_duncan_critical_ranges_match_published_table(self):
        mse, n = 4.0, 5
        se_mean = sqrt(mse / n)
        for df_error, ssr_values in PUBLISHED_DUNCAN_SSR.items():
            for span, ssr in enumerate(ssr_values, start=2):
                for factor, expect_significant in ((1.003, True), (0.997, False)):
                    gap = ssr * se_mean * factor
                    means = list(np.linspace(gap, 0.0, span))
                    labels = [f"T{i}" for i in range(span)]
                    comps = pairwise_comparisons(labels, means, pooled_pair_error([n] * span, mse, df_error), "duncan")
                    extreme = next(c for c in comps if {c["group1"], c["group2"]} == {labels[0], labels[-1]})
                    self.assertEqual(extreme["significant"], expect_significant,
                                     f"df={df_error} p={span} factor={factor}")

    def test_lsd_critical_difference_boundary(self):
        mse, n, df_error = 2.5, 4, 15
        cd = stats.t.ppf(0.975, df_error) * sqrt(2 * mse / n)
        for factor, expected in ((1.001, True), (0.999, False)):
            comps = pairwise_comparisons(["x", "y"], [0.0, cd * factor], pooled_pair_error([n, n], mse, df_error), "lsd")
            self.assertEqual(comps[0]["significant"], expected)


class SplitPlotErrorTests(unittest.TestCase):
    def test_gomez_gomez_standard_errors(self):
        ms_a, df_a, ms_b, df_b, b, r = 3.2, 6, 1.1, 27, 4, 3
        labels = ["A1 / B1", "A1 / B2", "A2 / B1"]
        main = {"A1 / B1": "A1", "A1 / B2": "A1", "A2 / B1": "A2"}
        provider = splitplot_pair_error(labels, main, ms_a, df_a, ms_b, df_b, b, r)
        self.assertEqual(provider(0, 1), (sqrt(2 * ms_b / r), df_b))
        se, df = provider(0, 2)
        combined = (b - 1) * ms_b + ms_a
        self.assertAlmostEqual(se, sqrt(2 * combined / (r * b)), places=12)
        self.assertAlmostEqual(df, combined ** 2 / (((b - 1) * ms_b) ** 2 / df_b + ms_a ** 2 / df_a), places=12)


class LetterDisplayTests(unittest.TestCase):
    def test_share_a_letter_iff_not_significant(self):
        rng = np.random.default_rng(11)
        for _ in range(400):
            k = int(rng.integers(2, 14))
            labels = [f"T{i}" for i in range(k)]
            means = rng.normal(0, 1, k).tolist()
            comps = [{"group1": labels[i], "group2": labels[j], "significant": bool(rng.random() < 0.5)}
                     for i, j in combinations(range(k), 2)]
            groups = compact_letter_groups(labels, means, comps)
            top = labels[int(np.argmax(means))]
            self.assertIn("a", groups[top])
            for c in comps:
                share = bool(set(groups[c["group1"]]) & set(groups[c["group2"]]))
                self.assertEqual(share, not c["significant"])

    def test_no_differences_gives_single_letter(self):
        comps = [{"group1": a, "group2": b, "significant": False} for a, b in combinations("ABC", 2)]
        self.assertEqual(compact_letter_groups(list("ABC"), [3, 2, 1], comps), {"A": ["a"], "B": ["a"], "C": ["a"]})


if __name__ == "__main__":
    unittest.main()
