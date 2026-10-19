"""Model-based imputation: single fills, multiple imputation, Rubin's rules and the /prep endpoints.

Rubin's rules are checked against the formulas of van Buuren (FIMD, 2nd ed., section 2.3) worked by hand, and
against their limiting cases; the imputers against scikit-learn called directly; multiple imputation against data
simulated from a known model.
Run from backend/:  python -m unittest discover -s tests -t .
"""
import io
import json
import math
import unittest

from tests.support import post_csv  # must come before other app imports

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.stats as stats  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from sklearn.experimental import enable_iterative_imputer  # noqa: E402,F401
from sklearn.impute import IterativeImputer, KNNImputer  # noqa: E402
from sklearn.linear_model import BayesianRidge  # noqa: E402

from app import dataprep as dp  # noqa: E402
from app import imputation as imp  # noqa: E402


def simulated(n=300, missing=0.3, seed=11):
    """y = 2 + 1.5 x1 - 0.8 x2 + noise, with y and x2 missing at random depending on the fully observed x1."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(10, 2, n)
    x2 = 0.5 * x1 + rng.normal(0, 1, n)
    y = 2 + 1.5 * x1 - 0.8 * x2 + rng.normal(0, 1, n)
    full = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    data = full.copy()
    chance = missing * 2 / (1 + np.exp(-(x1 - 10)))  # more gaps where x1 is large (MAR)
    data.loc[rng.random(n) < chance, "y"] = np.nan
    data.loc[rng.random(n) < missing / 2, "x2"] = np.nan
    return full, data


class RubinTests(unittest.TestCase):
    def test_hand_worked_example(self):
        q, u, m, dfcom = [10.0, 12.0, 11.0], [1.0, 1.2, 0.8], 3, 20
        pooled = imp.rubin(q, u, df_complete=dfcom)

        q_bar, u_bar, b = 11.0, 1.0, 1.0
        t = u_bar + (1 + 1 / m) * b
        lam = (1 + 1 / m) * b / t
        riv = (1 + 1 / m) * b / u_bar
        df_old = (m - 1) / lam ** 2
        df_obs = (dfcom + 1) / (dfcom + 3) * dfcom * (1 - lam)
        df = df_old * df_obs / (df_old + df_obs)
        fmi = (riv + 2 / (df + 3)) / (riv + 1)

        self.assertAlmostEqual(pooled["estimate"], q_bar)
        self.assertAlmostEqual(pooled["within"], u_bar)
        self.assertAlmostEqual(pooled["between"], b)
        self.assertAlmostEqual(pooled["total"], t)
        self.assertAlmostEqual(pooled["se"], math.sqrt(t))
        self.assertAlmostEqual(pooled["lambda"], lam)
        self.assertAlmostEqual(pooled["riv"], riv)
        self.assertAlmostEqual(pooled["df"], df)
        self.assertAlmostEqual(pooled["fmi"], fmi)
        crit = stats.t.ppf(0.975, df)
        self.assertAlmostEqual(pooled["ci_low"], q_bar - crit * math.sqrt(t))
        self.assertAlmostEqual(pooled["p_value"], 2 * stats.t.sf(q_bar / math.sqrt(t), df))

    def test_identical_imputations_reduce_to_the_complete_data_analysis(self):
        pooled = imp.rubin([5.0] * 10, [0.04] * 10, df_complete=30)
        self.assertEqual(pooled["between"], 0.0)
        self.assertAlmostEqual(pooled["se"], 0.2)
        self.assertAlmostEqual(pooled["lambda"], 0.0)
        # with no between-imputation variance the Barnard-Rubin df approaches the complete-data df (31/33 * 30)
        self.assertAlmostEqual(pooled["df"], 31 / 33 * 30, delta=0.01)
        self.assertLess(pooled["fmi"], 0.1)

    def test_large_samples_use_the_classic_df(self):
        pooled = imp.rubin([1.0, 2.0, 3.0, 2.0], [0.5, 0.5, 0.5, 0.5])
        lam = (1 + 1 / 4) * pooled["between"] / pooled["total"]
        self.assertAlmostEqual(pooled["df"], 3 / lam ** 2)

    def test_needs_two_estimates(self):
        with self.assertRaises(imp.ImputationError):
            imp.rubin([1.0], [0.1])


class SingleImputationTests(unittest.TestCase):
    def test_iterative_matches_scikit_learn_and_leaves_observed_values(self):
        _, data = simulated()
        filled = imp.impute(data, ["y", "x2"], ["x1"], method="iterative", seed=3, max_iter=10)
        reference = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=3, keep_empty_features=True)
        expected = reference.fit_transform(data[["y", "x2", "x1"]].to_numpy())
        np.testing.assert_allclose(filled["y"].to_numpy(), expected[:, 0], rtol=1e-6)
        observed = data["y"].notna()
        np.testing.assert_array_equal(filled.loc[observed, "y"], data.loc[observed, "y"])
        self.assertFalse(filled[["y", "x2"]].isna().any().any())
        self.assertTrue((filled["x1"] == data["x1"]).all())

    def test_knn_matches_scikit_learn_on_standardised_columns(self):
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0, 5.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0]})
        filled = imp.impute(df, ["a"], ["b"], method="knn", n_neighbors=2)
        standard = (df - df.mean()) / df.std(ddof=0)
        expected = KNNImputer(n_neighbors=2).fit_transform(standard.to_numpy())[:, 0] * df["a"].std(ddof=0) + df["a"].mean()
        self.assertAlmostEqual(filled["a"][2], expected[2])
        self.assertAlmostEqual(filled["a"][2], 3.0)  # neighbours are rows 2 and 4 (b = 20, 40)

    def test_bounds(self):
        df = pd.DataFrame({"yield": [0.2, 0.1, np.nan, 0.3, np.nan, 0.15], "dose": [1.0, 0.0, -9.0, 2.0, 9.0, 0.5]})
        free = imp.impute(df, ["yield"], ["dose"], method="iterative", seed=1)
        clipped = imp.impute(df, ["yield"], ["dose"], method="iterative", seed=1, bounds="observed_range")
        self.assertTrue(free["yield"].min() < 0.1 or free["yield"].max() > 0.3)
        self.assertGreaterEqual(clipped["yield"].min(), 0.1)
        self.assertLessEqual(clipped["yield"].max(), 0.3)

    def test_category_predictors_are_dummy_coded_and_must_be_complete(self):
        df = pd.DataFrame({"Treatment": ["A", "A", "A", "B", "B", "B"] * 3,
                           "y": [1.0, 1.1, np.nan, 5.0, np.nan, 5.2] * 3})
        filled = imp.impute(df, ["y"], ["Treatment"], method="iterative", seed=0)
        gap_a, gap_b = filled.loc[2, "y"], filled.loc[4, "y"]
        self.assertLess(abs(gap_a - 1.05), 0.5)
        self.assertLess(abs(gap_b - 5.1), 0.5)
        broken = df.copy()
        broken.loc[0, "Treatment"] = None
        with self.assertRaisesRegex(imp.ImputationError, "has gaps"):
            imp.impute(broken, ["y"], ["Treatment"])

    def test_refusals(self):
        df = pd.DataFrame({"t": ["a", "b", None], "y": [1.0, np.nan, np.nan]})
        with self.assertRaisesRegex(imp.ImputationError, "not numeric"):
            imp.impute(df, ["t"])
        with self.assertRaisesRegex(imp.ImputationError, "fewer than 3"):
            imp.impute(df, ["y"])

    def test_row_identifiers_are_not_default_predictors(self):
        df = pd.DataFrame({"Plot": [101, 102, 103, 104, 105], "Rep": [1, 1, 2, 2, 3],
                           "Dose": [0.0, 0.5, 1.0, 1.5, 2.0], "y": [1.0, np.nan, 2.1, 2.4, np.nan]})
        predictors, identifiers = imp.default_predictors(df, ["y"])
        self.assertEqual(identifiers, ["Plot"])
        self.assertEqual(predictors, ["Rep", "Dose"])  # Dose is distinct on every row but not whole numbers
        _, log = dp.apply_steps(df, [{"type": "impute_model", "columns": ["y"], "method": "knn", "n_neighbors": 2}])
        self.assertIn("Plot left out", log[0]["message"])

    def test_recipe_step(self):
        _, data = simulated(n=80)
        out, log = dp.apply_steps(data, [{"type": "impute_model", "columns": ["y"], "method": "iterative"}])
        self.assertEqual(int(out["y"].isna().sum()), 0)
        self.assertIn("single imputation", log[0]["message"])
        self.assertIn("using x1, x2", log[0]["message"])


class MultipleImputationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full, cls.data = simulated()
        cls.completed = imp.multiple_impute(cls.data, ["y", "x2"], ["x1"], m=20, seed=7)

    def test_draws_differ_but_observed_values_never_change(self):
        gaps = self.data["y"].isna()
        first, second = self.completed[0], self.completed[1]
        self.assertFalse(np.allclose(first.loc[gaps, "y"], second.loc[gaps, "y"]))
        for table in self.completed:
            np.testing.assert_array_equal(table.loc[~gaps, "y"], self.data.loc[~gaps, "y"])
            self.assertFalse(table[["y", "x2"]].isna().any().any())

    def test_reproducible_with_the_same_seed(self):
        again = imp.multiple_impute(self.data, ["y", "x2"], ["x1"], m=3, seed=7)
        for a, b in zip(again, self.completed[:3]):
            pd.testing.assert_frame_equal(a, b)

    def test_pooled_regression_recovers_the_model_and_is_honest_about_uncertainty(self):
        pooled = imp.pooled_regression(self.completed, "y", ["x1", "x2"])
        terms = {c["term"]: c for c in pooled["coefficients"]}
        for term, truth in (("x1", 1.5), ("x2", -0.8)):
            self.assertLess(terms[term]["ci_low"], truth, term)
            self.assertGreater(terms[term]["ci_high"], truth, term)
            self.assertGreater(terms[term]["between"], 0)
            self.assertGreater(terms[term]["fmi"], 0)

        # the pooled standard errors are larger than those from the full data before any values were removed ...
        X = sm.add_constant(self.full[["x1", "x2"]])
        full_fit = sm.OLS(self.full["y"], X).fit()
        self.assertGreater(terms["x1"]["se"], full_fit.bse["x1"])
        # ... and larger than pretending one completed table was measured
        self.assertGreater(terms["x1"]["se"], sm.OLS(self.completed[0]["y"], sm.add_constant(self.completed[0][["x1", "x2"]])).fit().bse["x1"])
        self.assertTrue(0 < pooled["r_squared"] < 1)

    def test_pooled_means_cover_the_full_data_mean(self):
        pooled = imp.pooled_means(self.completed, ["y"])["y"]
        self.assertLess(pooled["ci_low"], self.full["y"].mean())
        self.assertGreater(pooled["ci_high"], self.full["y"].mean())
        self.assertEqual(pooled["m"], 20)

    def test_mean_ignoring_the_missingness_mechanism_is_biased_but_imputation_corrects_it(self):
        # gaps in y are more common where x1 (and so y) is large, so the observed-only mean is too low
        complete_case = self.data["y"].mean()
        pooled = imp.pooled_means(self.completed, ["y"])["y"]["estimate"]
        truth = self.full["y"].mean()
        self.assertLess(complete_case, truth)
        self.assertLess(abs(pooled - truth), abs(complete_case - truth))

    def test_diagnostics_and_long_format(self):
        report = imp.imputation_diagnostics(self.data, self.completed, ["y"])[0]
        self.assertEqual(report["missing"], int(self.data["y"].isna().sum()))
        self.assertEqual(len(report["imputed_mean_by_draw"]), 20)
        long = imp.long_format(self.data, self.completed[:3])
        self.assertEqual(len(long), 4 * len(self.data))
        self.assertEqual(sorted(long[".imp"].unique()), [0, 1, 2, 3])
        self.assertTrue(long.loc[long[".imp"] == 0, "y"].isna().any())
        self.assertFalse(long.loc[long[".imp"] == 2, "y"].isna().any())

    def test_number_of_imputations_is_checked(self):
        with self.assertRaisesRegex(imp.ImputationError, "between 2 and 100"):
            imp.multiple_impute(self.data, ["y"], m=1)


class EndpointTests(unittest.TestCase):
    def test_multiple_imputation_endpoint(self):
        _, data = simulated(n=120)
        code, body = post_csv("/analyze/prep/multiple-imputation", data, columns_str="y,x2", m="5", seed="1",
                              regression_dependent="y", regression_predictors_str="x1,x2")
        self.assertEqual(code, 200, body)
        self.assertEqual(body["settings"]["predictors"], ["x1"])
        self.assertEqual([c["column"] for c in body["columns"]], ["y", "x2"])
        self.assertIn("fmi", body["columns"][0]["pooled_mean"])
        self.assertEqual([c["term"] for c in body["regression"]["coefficients"]], ["Intercept", "x1", "x2"])
        long = pd.read_csv(io.StringIO(body["long_csv"]))
        self.assertEqual(len(long), 6 * len(data))
        json.dumps(body)

    def test_endpoint_refusals(self):
        _, data = simulated(n=60)
        for form, message in [
            ({"columns_str": "x1"}, "no missing values"),
            ({"columns_str": "rainfall"}, "not in the dataset"),
            ({"columns_str": "y", "m": "1"}, "between 2 and 100"),
            ({"columns_str": "y", "regression_dependent": "y", "regression_predictors_str": "soil"}, "not in the dataset"),
        ]:
            code, body = post_csv("/analyze/prep/multiple-imputation", data, **form)
            self.assertEqual(code, 400, form)
            self.assertIn(message, body["detail"], form)


if __name__ == "__main__":
    unittest.main()
