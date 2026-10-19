"""SEM engines validated against lavaan's published results and exact algebraic identities.

Run from backend/:  python -m unittest discover -s tests -t .
"""
import json
import math
import unittest
from itertools import combinations

from tests.support import post_csv  # must come before other app imports

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from semopy.examples import political_democracy  # noqa: E402

from app.sem_engine import SemError, fit_cb_sem, fit_pls_sem  # noqa: E402

# Bollen (1989) Political Democracy model, as in the lavaan tutorial
PD_LATENTS = [
    {"id": "ind60", "name": "ind60", "indicators": ["x1", "x2", "x3"]},
    {"id": "dem60", "name": "dem60", "indicators": ["y1", "y2", "y3", "y4"]},
    {"id": "dem65", "name": "dem65", "indicators": ["y5", "y6", "y7", "y8"]},
]
PD_PATHS = [{"from": "ind60", "to": "dem60"}, {"from": "ind60", "to": "dem65"}, {"from": "dem60", "to": "dem65"}]
PD_RESIDUAL_COVARIANCES = [("y1", "y5"), ("y2", "y4"), ("y2", "y6"), ("y3", "y7"), ("y4", "y8"), ("y6", "y8")]

# lavaan ML output for this model (lavaan tutorial, "SEM" section; fitMeasures())
LAVAAN_ESTIMATES = {
    ("ind60", "x2"): 2.180, ("ind60", "x3"): 1.819,
    ("dem60", "y2"): 1.257, ("dem60", "y3"): 1.058, ("dem60", "y4"): 1.265,
    ("dem65", "y6"): 1.186, ("dem65", "y7"): 1.280, ("dem65", "y8"): 1.266,
}
LAVAAN_PATHS = {("ind60", "dem60"): 1.483, ("ind60", "dem65"): 0.572, ("dem60", "dem65"): 0.837}
LAVAAN_FIT = {"chi_square": 38.125, "df": 35, "p_value": 0.329, "cfi": 0.995, "tli": 0.993, "rmsea": 0.035, "srmr": 0.044}


def three_factor_data(n=300, seed=0, prefixes=("a", "b", "c")):
    rng = np.random.default_rng(seed)
    f1 = rng.normal(size=n)
    f2 = 0.5 * f1 + math.sqrt(0.75) * rng.normal(size=n)
    f3 = 0.4 * f1 + 0.3 * f2 + rng.normal(scale=0.8, size=n)
    columns = {}
    for prefix, factor, loadings in ((prefixes[0], f1, (0.9, 0.8, 0.7)),
                                     (prefixes[1], f2, (0.85, 0.75, 0.8)),
                                     (prefixes[2], f3, (0.8, 0.9, 0.7))):
        for i, lam in enumerate(loadings):
            columns[f"{prefix}{i + 1}"] = lam * factor + rng.normal(scale=math.sqrt(1 - lam ** 2), size=n)
    return pd.DataFrame(columns)


def three_factor_spec(prefixes=("a", "b", "c")):
    latents = [{"id": f"L{k}", "name": f"Construct {k}", "indicators": [f"{p}{i}" for i in (1, 2, 3)]}
               for k, p in enumerate(prefixes, start=1)]
    return latents, [{"from": "L1", "to": "L3"}, {"from": "L2", "to": "L3"}]


def standardized(df):
    return (df - df.mean()) / df.std(ddof=1)


class CbSemReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = fit_cb_sem(political_democracy.get_data(), PD_LATENTS, PD_PATHS,
                                residual_covariances=PD_RESIDUAL_COVARIANCES)

    def test_global_fit_matches_lavaan(self):
        fit = self.result["fit_indices"]
        self.assertEqual(fit["df"], LAVAAN_FIT["df"])
        self.assertAlmostEqual(fit["chi_square"], LAVAAN_FIT["chi_square"], places=2)
        for key in ("p_value", "cfi", "tli", "rmsea", "srmr"):
            self.assertAlmostEqual(fit[key], LAVAAN_FIT[key], delta=0.0015, msg=key)

    def test_estimates_match_lavaan(self):
        loadings = {(row["latent"], row["indicator"]): row for row in self.result["outer_loadings"]}
        for key, expected in LAVAAN_ESTIMATES.items():
            self.assertAlmostEqual(loadings[key]["estimate"], expected, delta=0.002, msg=str(key))
        paths = {(row["from"], row["to"]): row for row in self.result["path_coefficients"]}
        for key, expected in LAVAAN_PATHS.items():
            self.assertAlmostEqual(paths[key]["estimate"], expected, delta=0.002, msg=str(key))

    def test_standardized_solution_is_internally_consistent(self):
        corr = self.result["latent_correlations"]
        for lv in corr:
            self.assertAlmostEqual(corr[lv][lv], 1.0, places=6)
        single_predictor = next(p for p in self.result["path_coefficients"] if p["to"] == "dem60")
        self.assertAlmostEqual(self.result["r2_values"]["dem60"], single_predictor["coefficient"] ** 2, places=5)
        loadings = {(row["latent"], row["indicator"]): row["loading"] for row in self.result["outer_loadings"]}
        for row in self.result["reliability_indices"]:
            lam = np.array([loadings[(row["latent"], i)] for i in next(lv["indicators"] for lv in PD_LATENTS if lv["id"] == row["latent"])])
            self.assertAlmostEqual(row["ave"], float(np.mean(lam ** 2)), places=10)
            self.assertAlmostEqual(row["composite_reliability"], lam.sum() ** 2 / (lam.sum() ** 2 + np.sum(1 - lam ** 2)), places=10)

    def test_single_indicator_construct_is_rejected(self):
        latents = PD_LATENTS[:2] + [{"id": "solo", "name": "solo", "indicators": ["y5"]}]
        with self.assertRaisesRegex(SemError, "at least 2 indicators"):
            fit_cb_sem(political_democracy.get_data(), latents, PD_PATHS[:1])

    def test_feedback_loop_is_rejected(self):
        loop = PD_PATHS + [{"from": "dem65", "to": "ind60"}]
        with self.assertRaisesRegex(SemError, "feedback loop"):
            fit_cb_sem(political_democracy.get_data(), PD_LATENTS, loop)


class PlsIdentityTests(unittest.TestCase):
    def test_single_indicator_constructs_reduce_to_standardized_ols(self):
        rng = np.random.default_rng(1)
        n = 250
        x1 = rng.normal(size=n)
        x2 = 0.4 * x1 + rng.normal(size=n)
        m = 0.4 * x1 + 0.3 * x2 + rng.normal(size=n)
        y = 0.5 * m + 0.2 * x1 + rng.normal(size=n)
        df = pd.DataFrame({"X1": x1, "X2": x2, "M": m, "Y": y})
        latents = [{"id": c, "name": c, "indicators": [c]} for c in df.columns]
        paths = [{"from": "X1", "to": "M"}, {"from": "X2", "to": "M"}, {"from": "X1", "to": "Y"}, {"from": "M", "to": "Y"}]
        result = fit_pls_sem(df, latents, paths, bootstrap_samples=200)

        z = standardized(df)
        coefficients = {(p["from"], p["to"]): p["coefficient"] for p in result["path_coefficients"]}
        for outcome, predictors in (("M", ["X1", "X2"]), ("Y", ["X1", "M"])):
            beta, *_ = np.linalg.lstsq(z[predictors].values, z[outcome].values, rcond=None)
            for predictor, b in zip(predictors, beta):
                self.assertAlmostEqual(coefficients[(predictor, outcome)], b, places=10)
            residual = z[outcome].values - z[predictors].values @ beta
            self.assertAlmostEqual(result["r2_values"][outcome], 1 - residual @ residual / (n - 1), places=10)
        for row in result["outer_loadings"]:
            self.assertAlmostEqual(row["loading"], 1.0, places=12)

    def test_two_block_outer_weights_are_leading_singular_vectors(self):
        df = three_factor_data(seed=2)
        latents, _ = three_factor_spec()
        result = fit_pls_sem(df, latents[:2], [{"from": "L1", "to": "L2"}], bootstrap_samples=100)
        z = standardized(df)
        cross = z[["a1", "a2", "a3"]].values.T @ z[["b1", "b2", "b3"]].values / (len(df) - 1)
        u, _, vt = np.linalg.svd(cross)
        for latent, direction in (("L1", u[:, 0]), ("L2", vt[0])):
            w = np.array([row["weight"] for row in result["outer_weights"] if row["latent"] == latent])
            cosine = abs(w @ direction) / (np.linalg.norm(w) * np.linalg.norm(direction))
            self.assertAlmostEqual(cosine, 1.0, places=9)

    def test_reliability_and_htmt_follow_their_definitions(self):
        df = three_factor_data(seed=3)
        latents, paths = three_factor_spec()
        result = fit_pls_sem(df, latents, paths, bootstrap_samples=100)
        z = standardized(df)
        abs_corr = z.corr().abs()
        loadings = {(r["latent"], r["indicator"]): r["loading"] for r in result["outer_loadings"]}
        blocks = {lv["id"]: lv["indicators"] for lv in latents}
        for row in result["reliability_indices"]:
            items = blocks[row["latent"]]
            lam = np.array([loadings[(row["latent"], i)] for i in items])
            k = len(items)
            alpha = k / (k - 1) * (1 - z[items].var(ddof=1).sum() / z[items].sum(axis=1).var(ddof=1))
            self.assertAlmostEqual(row["cronbach_alpha"], alpha, places=10)
            self.assertAlmostEqual(row["ave"], float(np.mean(lam ** 2)), places=10)
        for row in result["discriminant_validity"]["htmt"]:
            a, b = blocks[row["latent1"]], blocks[row["latent2"]]
            hetero = np.mean([abs_corr.loc[i, j] for i in a for j in b])
            mono_a = np.mean([abs_corr.loc[i, j] for i, j in combinations(a, 2)])
            mono_b = np.mean([abs_corr.loc[i, j] for i, j in combinations(b, 2)])
            self.assertAlmostEqual(row["htmt"], hetero / math.sqrt(mono_a * mono_b), places=10)

    def test_bootstrap_standard_errors_are_close_to_ols(self):
        rng = np.random.default_rng(4)
        n = 400
        x1 = rng.normal(size=n)
        x2 = 0.3 * x1 + rng.normal(size=n)
        y = 0.4 * x1 + 0.25 * x2 + rng.normal(size=n)
        df = pd.DataFrame({"X1": x1, "X2": x2, "Y": y})
        latents = [{"id": c, "name": c, "indicators": [c]} for c in df.columns]
        result = fit_pls_sem(df, latents, [{"from": "X1", "to": "Y"}, {"from": "X2", "to": "Y"}], bootstrap_samples=2000)
        z = standardized(df)
        ols = sm.OLS(z["Y"], sm.add_constant(z[["X1", "X2"]])).fit()
        for path in result["path_coefficients"]:
            self.assertAlmostEqual(path["std_error"] / ols.bse[path["from"]], 1.0, delta=0.15)
            self.assertLess(path["ci_lower"], path["coefficient"])
            self.assertGreater(path["ci_upper"], path["coefficient"])

    def test_bootstrap_is_reproducible(self):
        df = three_factor_data(seed=5)
        latents, paths = three_factor_spec()
        first = fit_pls_sem(df, latents, paths, bootstrap_samples=150)
        second = fit_pls_sem(df, latents, paths, bootstrap_samples=150)
        self.assertEqual(first["path_coefficients"], second["path_coefficients"])


class SemEndpointTests(unittest.TestCase):
    PREFIXES = ("Soil pH ", "Organic C ", "Yield t/ha ")

    @classmethod
    def setUpClass(cls):
        cls.data = three_factor_data(seed=6, prefixes=cls.PREFIXES)
        cls.latents, cls.paths = three_factor_spec(cls.PREFIXES)

    def post(self, sem_type, latents=None, paths=None):
        spec = {"latent_variables": latents or self.latents, "paths": paths or self.paths}
        return post_csv("/analyze/sem", self.data, sem_type=sem_type, specification=json.dumps(spec))

    def test_pls_endpoint(self):
        status, res = self.post("pls")
        self.assertEqual(status, 200, res)
        self.assertEqual(res["sem_type"], "pls")
        self.assertIsNone(res["fit_indices"])
        self.assertEqual(res["bootstrap"]["samples"], 1000)
        self.assertTrue(all(p["p_value"] is not None for p in res["path_coefficients"]))

    def test_cb_endpoint_handles_column_names_with_spaces(self):
        status, res = self.post("cb")
        self.assertEqual(status, 200, res)
        direct = fit_cb_sem(self.data, self.latents, self.paths)
        self.assertAlmostEqual(res["fit_indices"]["chi_square"], direct["fit_indices"]["chi_square"], places=8)
        self.assertEqual({row["indicator"] for row in res["outer_loadings"]}, set(self.data.columns))

    def test_lavaan_syntax_endpoint_reports_srmr(self):
        model = "ind60 =~ x1 + x2 + x3\ndem60 =~ y1 + y2 + y3 + y4\ndem60 ~ ind60"
        status, res = post_csv("/api/sem/fit", political_democracy.get_data(), model=model)
        self.assertEqual(status, 200, res)
        self.assertIsNotNone(res["fit_indices"]["SRMR"])

    def test_unknown_sem_type_is_rejected(self):
        status, res = self.post("gsca")
        self.assertEqual(status, 400, res)

    def test_path_to_undefined_construct_is_rejected(self):
        status, res = self.post("pls", paths=[{"from": "L1", "to": "nope"}])
        self.assertEqual(status, 422, res)
        self.assertIn("undefined latent variable", res["detail"])


if __name__ == "__main__":
    unittest.main()
