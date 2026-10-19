"""Genetic divergence: error SSCP, Wilks' Λ, Mahalanobis D², Tocher clustering, contributions and the /divergence endpoint.

Checked against statsmodels (OLS residuals, MANOVA), scipy (Mahalanobis distance) and a hand-worked Tocher example.
Run from backend/:  python -m unittest discover -s tests -t .
"""
import base64
import unittest

from tests.support import post_csv  # must come before other app imports
from tests.test_endpoints_smoke import PNG_SIGNATURE  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy.spatial.distance import mahalanobis  # noqa: E402
from statsmodels.multivariate.manova import MANOVA  # noqa: E402

from app import divergence as div  # noqa: E402

TRAITS = ["Yield", "Height", "Tillers", "Protein"]


def trial_frame(G=12, r=3, seed=3):
    rng = np.random.default_rng(seed)
    centres = np.array([[0, 0, 0, 0], [6, 2, -3, 1], [-4, 5, 2, -2]], dtype=float)
    effects = centres[np.arange(G) % 3] + rng.normal(0, 0.8, (G, 4))
    covariance = [[1, .5, .2, 0], [.5, 1, .3, .1], [.2, .3, 1, .4], [0, .1, .4, 1]]
    rows = []
    for k in range(r):
        block = rng.normal(0, 1, 4)
        noise = rng.multivariate_normal(np.zeros(4), covariance, G)
        for i in range(G):
            values = 20 + effects[i] + block + noise[i]
            rows.append({"Genotype": f"V{i + 1}", "Rep": k + 1, **dict(zip(TRAITS, values))})
    return pd.DataFrame(rows)


def codes(df):
    return df["Genotype"].str[1:].astype(int).to_numpy() - 1, df["Rep"].to_numpy() - 1


class TestStatistics(unittest.TestCase):
    def setUp(self):
        self.df = trial_frame()
        self.genotype, self.block = codes(self.df)
        self.Y = self.df[TRAITS].to_numpy()
        self.test = div.manova_wilks(self.Y, self.genotype, self.block)

    def test_error_sscp_matches_statsmodels_residuals(self):
        residuals = np.column_stack([smf.ols(f"{t} ~ C(Genotype) + C(Rep)", data=self.df).fit().resid for t in TRAITS])
        np.testing.assert_allclose(self.test["error_sscp"], residuals.T @ residuals, rtol=1e-8, atol=1e-8)
        self.assertEqual(self.test["df_error"], (12 - 1) * (3 - 1))

    def test_wilks_lambda_matches_statsmodels_manova(self):
        manova = MANOVA.from_formula(" + ".join(TRAITS) + " ~ C(Genotype) + C(Rep)", data=self.df).mv_test()
        row = manova.results["C(Genotype)"]["stat"].loc["Wilks' lambda"]
        self.assertAlmostEqual(self.test["lambda"], float(row["Value"]), places=10)
        self.assertAlmostEqual(self.test["f"], float(row["F Value"]), places=8)
        self.assertAlmostEqual(self.test["df1"], float(row["Num DF"]), places=8)
        self.assertAlmostEqual(self.test["df2"], float(row["Den DF"]), places=8)
        self.assertLess(self.test["p_value"], 0.001)

    def test_d2_contributions_and_canonical_variates(self):
        covariance = self.test["error_sscp"] / self.test["df_error"]
        means = pd.DataFrame(self.Y).groupby(self.genotype).mean().to_numpy()
        d2, transformed = div.mahalanobis_d2(means, covariance)
        inverse = np.linalg.inv(covariance)
        for i, j in [(0, 1), (2, 7), (5, 11)]:
            self.assertAlmostEqual(d2[i, j], mahalanobis(means[i], means[j], inverse) ** 2, places=8)
        rebuilt = ((transformed[:, None, :] - transformed[None, :, :]) ** 2).sum(axis=2)
        np.testing.assert_allclose(rebuilt, d2, atol=1e-10)  # D² is the squared distance of the transformed means
        counts, percent = div.character_contributions(transformed)
        self.assertEqual(counts.sum(), 12 * 11 // 2)
        self.assertAlmostEqual(percent.sum(), 100.0)
        scores, share = div.canonical_variates(transformed)
        rebuilt = ((scores[:, None, :] - scores[None, :, :]) ** 2).sum(axis=2)
        np.testing.assert_allclose(rebuilt, d2, atol=1e-8)  # canonical space reproduces D² exactly
        self.assertAlmostEqual(share.sum(), 100.0)

    def test_tocher_worked_example(self):
        positions = np.array([0.0, 1.0, 2.0, 10.0, 11.0])
        d2 = (positions[:, None] - positions[None, :]) ** 2
        clusters, theta = div.tocher(d2)
        self.assertEqual(theta, 1.0)  # every genotype's nearest neighbour is at D² = 1
        self.assertEqual(clusters, [[0, 1], [3, 4], [2]])  # genotype 3 would average D² = 2.5 > θ with cluster I

    def test_tocher_properties_and_cluster_distances(self):
        covariance = self.test["error_sscp"] / self.test["df_error"]
        means = pd.DataFrame(self.Y).groupby(self.genotype).mean().to_numpy()
        d2, _ = div.mahalanobis_d2(means, covariance)
        clusters, theta = div.tocher(d2)
        off = d2 + np.diag(np.full(12, np.inf))
        self.assertAlmostEqual(theta, off.min(axis=1).max())
        self.assertEqual(sorted(g for c in clusters for g in c), list(range(12)))
        remaining = set(range(12))
        for cluster in clusters:
            if len(cluster) > 1:
                pool = sorted(remaining)
                sub = off[np.ix_(pool, pool)]
                a, b = np.unravel_index(np.argmin(sub), sub.shape)
                self.assertEqual(set(cluster[:2]), {pool[a], pool[b]})  # starts with the closest unclustered pair
                for position in range(2, len(cluster)):
                    self.assertLessEqual(d2[cluster[position], cluster[:position]].mean(), theta + 1e-12)
            remaining -= set(cluster)
        self.assertGreaterEqual(len(clusters), 3)
        for cluster in clusters:  # no cluster mixes genotypes from different simulated groups
            self.assertEqual(len({g % 3 for g in cluster}), 1)
        distances = div.cluster_distances(d2, clusters)
        first = clusters[0]
        pairs = [d2[i, j] for n, i in enumerate(first) for j in first[n + 1:]]
        self.assertAlmostEqual(distances[0, 0], np.mean(pairs))
        self.assertAlmostEqual(distances[0, 1], d2[np.ix_(clusters[0], clusters[1])].mean())
        self.assertTrue(np.all(np.diag(distances) < distances[0, 1]))


class DivergenceEndpointTests(unittest.TestCase):
    def post(self, df, **form):
        fields = dict(genotype_col="Genotype", traits_str=",".join(TRAITS), rep_col="Rep")
        fields.update(form)
        return post_csv("/analyze/divergence", df, **fields)

    def test_replicated_trial(self):
        status, res = self.post(trial_frame())
        self.assertEqual(status, 200, res)
        self.assertEqual(res["covariance_source"], "error")
        self.assertLess(res["wilks"]["p_value"], 0.001)
        members = sorted((g for c in res["clusters"] for g in c["members"]), key=lambda g: int(g[1:]))
        self.assertEqual(members, [f"V{i + 1}" for i in range(12)])
        self.assertEqual([c["name"] for c in res["clusters"]][:2], ["Cluster I", "Cluster II"])
        self.assertAlmostEqual(sum(c["percent"] for c in res["contributions"]), 100.0)
        self.assertEqual(len(res["canonical"]["scores"]), 12)
        for key, image in res["charts"].items():
            with self.subTest(chart=key):
                self.assertTrue(base64.b64decode(image).startswith(PNG_SIGNATURE))

    def test_unreplicated_genotypes_use_their_covariance(self):
        df = trial_frame(G=15)
        means = df.groupby("Genotype", as_index=False)[TRAITS].mean()
        status, res = self.post(means, rep_col="")
        self.assertEqual(status, 200, res)
        self.assertEqual(res["covariance_source"], "genotypes")
        self.assertIsNone(res["wilks"])
        self.assertIsNone(res["charts"]["canonical"])
        self.assertTrue(res["notes"])

    def test_without_block_column(self):
        status, res = self.post(trial_frame(), rep_col="")
        self.assertEqual(status, 200, res)
        self.assertIsNotNone(res["wilks"])

    def test_validation(self):
        df = trial_frame()
        cases = [
            (df, dict(traits_str="Yield")),
            (df[df["Genotype"].isin(["V1", "V2"])], {}),
            (df.assign(Yield=df["Yield"].astype(str).where(df.index != 4, "lost")), {}),
            (df.assign(Copy=df["Yield"] * 2), dict(traits_str="Yield,Copy,Height")),
            (df, dict(traits_str="Yield,Nope")),
            (df, dict(traits_str="Yield,Genotype")),
        ]
        for frame, form in cases:
            with self.subTest(form=form, rows=len(frame)):
                status, res = self.post(frame, **form)
                self.assertEqual(status, 400, res)


if __name__ == "__main__":
    unittest.main()
