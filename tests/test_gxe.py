"""Genotype × environment analysis: AMMI ANOVA and scores, stability statistics, GGE biplot geometry, /gxe endpoint.

Checked against statsmodels (combined ANOVA), numpy regression and exact algebraic identities.
Run from backend/:  python -m unittest discover -s tests -t .
"""
import base64
import unittest

from tests.support import post_csv  # must come before other app imports
from tests.test_endpoints_smoke import PNG_SIGNATURE  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

from app import gxe  # noqa: E402


def trial_frame(G=8, E=5, r=3, seed=21):
    rng = np.random.default_rng(seed)
    genotype, environment = rng.normal(0, 4, G), rng.normal(0, 10, E)
    u, v = rng.normal(size=G), rng.normal(size=E)
    rows = []
    for j in range(E):
        blocks = rng.normal(0, 1.5, r)
        for k in range(r):
            for i in range(G):
                rows.append({"Genotype": f"G{i + 1}", "Location": f"L{j + 1}", "Rep": k + 1,
                             "Yield": 50 + genotype[i] + environment[j] + 3 * u[i] * v[j] + blocks[k] + rng.normal(0, 1.2)})
    return pd.DataFrame(rows)


def indices(df):
    return (df["Genotype"].str[1:].astype(int).to_numpy() - 1, df["Location"].str[1:].astype(int).to_numpy() - 1,
            df["Rep"].to_numpy() - 1)


def means_of(df):
    return df.pivot_table(index="Genotype", columns="Location", values="Yield", aggfunc="mean").loc[
        [f"G{i + 1}" for i in range(df["Genotype"].nunique())]].to_numpy()


def statsmodels_term(table, *factors):
    for name in table.index:
        if set(name.split(":")) == set(factors):
            return table.loc[name, "sum_sq"], table.loc[name, "df"]
    raise KeyError(factors)


class AmmiTests(unittest.TestCase):
    def setUp(self):
        self.df = trial_frame()
        g, e, b = indices(self.df)
        self.result = gxe.ammi_anova(self.df["Yield"], g, e, 8, 5, 3, b)

    def test_combined_anova_matches_statsmodels(self):
        model = smf.ols("Yield ~ C(Location) + C(Location):C(Rep) + C(Genotype) + C(Genotype):C(Location)", data=self.df).fit()
        table = sm.stats.anova_lm(model, typ=1)
        rows = {row["source"]: row for row in self.result["rows"]}
        for source, factors in [("Environments", ("C(Location)",)), ("Replications within environments", ("C(Location)", "C(Rep)")),
                                ("Genotypes", ("C(Genotype)",)), ("Genotype × environment", ("C(Genotype)", "C(Location)"))]:
            ss, df = statsmodels_term(table, *factors)
            with self.subTest(source=source):
                self.assertAlmostEqual(rows[source]["ss"], ss, places=6)
                self.assertEqual(rows[source]["df"], df)
        self.assertAlmostEqual(rows["Pooled error"]["ss"], table.loc["Residual", "sum_sq"], places=6)
        self.assertEqual(rows["Pooled error"]["df"], table.loc["Residual", "df"])
        # genotypes are tested against the pooled error, environments against replicates within environments
        self.assertAlmostEqual(rows["Genotypes"]["f"], rows["Genotypes"]["ms"] / rows["Pooled error"]["ms"])
        self.assertAlmostEqual(rows["Environments"]["f"], rows["Environments"]["ms"] / rows["Replications within environments"]["ms"])

    def test_ipca_terms_partition_the_interaction(self):
        ipca = self.result["ipca"]
        self.assertEqual([t["df"] for t in ipca], [10, 8, 6, 4])  # Gollob: G + E - 1 - 2k
        self.assertAlmostEqual(sum(t["ss"] for t in ipca), self.result["ss_ge"], places=8)
        singular = np.linalg.svd(gxe.double_centre(means_of(self.df)), compute_uv=False)
        for term, s in zip(ipca, singular):
            self.assertAlmostEqual(term["ss"], 3 * s ** 2, places=8)
        self.assertAlmostEqual(ipca[-1]["cumulative"], 100.0, places=8)
        self.assertGreater(ipca[0]["percent"], 50)  # the simulated interaction has rank one

    def test_scores_reconstruct_the_interaction_and_asv(self):
        interaction = self.result["interaction"]
        g_scores, e_scores, _ = gxe.ammi_scores(interaction, 4)
        np.testing.assert_allclose(g_scores @ e_scores.T, interaction, atol=1e-10)
        ss1, ss2 = self.result["ipca"][0]["ss"], self.result["ipca"][1]["ss"]
        asv = gxe.ammi_stability_value(g_scores[:, :2], ss1, ss2)
        np.testing.assert_allclose(asv, np.sqrt((ss1 / ss2 * g_scores[:, 0]) ** 2 + g_scores[:, 1] ** 2))

    def test_unreplicated_trial_has_no_f_tests(self):
        means = means_of(self.df)
        result = gxe.ammi_anova(means.ravel(), np.repeat(np.arange(8), 5), np.tile(np.arange(5), 8), 8, 5)
        self.assertIsNone(result["ms_error"])
        self.assertTrue(all(row["f"] is None for row in result["rows"]))
        self.assertAlmostEqual(result["ss_ge"], self.result["ss_ge"] / 3, places=8)


class StabilityTests(unittest.TestCase):
    def setUp(self):
        self.df = trial_frame(G=7, E=6, r=2, seed=4)
        self.means = means_of(self.df)
        g, e, b = indices(self.df)
        self.anova = gxe.ammi_anova(self.df["Yield"], g, e, 7, 6, 2, b)
        self.stats = gxe.stability_statistics(self.means, self.anova["ms_error"], 2, self.anova["df_error"])

    def test_eberhart_russell_regression(self):
        environment_means = self.means.mean(axis=0)
        for i in range(7):
            slope, intercept = np.polyfit(environment_means, self.means[i], 1)
            residual = self.means[i] - (slope * environment_means + intercept)
            with self.subTest(genotype=i):
                self.assertAlmostEqual(self.stats["bi"][i], slope, places=10)
                ms_deviation = (residual ** 2).sum() / 4
                self.assertAlmostEqual(self.stats["s2di"][i], ms_deviation - self.anova["ms_error"] / 2, places=10)
        self.assertAlmostEqual(self.stats["bi"].mean(), 1.0, places=10)  # slopes average to one

    def test_ecovalence_and_shukla_identities(self):
        W = self.stats["ecovalence"]
        self.assertAlmostEqual(W.sum(), self.anova["ss_ge"] / 2, places=8)
        self.assertAlmostEqual(self.stats["shukla"].sum(), 7 * W.sum() / (6 * 5), places=8)
        self.assertAlmostEqual(self.stats["ecovalence_percent"].sum(), 100.0, places=8)

    def test_coefficient_of_variation(self):
        row = self.means[2]
        self.assertAlmostEqual(self.stats["cv"][2], 100 * row.std(ddof=1) / row.mean(), places=10)


class GgeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(9)
        self.means = 40 + rng.normal(0, 5, (12, 6)) + rng.normal(0, 8, 6)

    def test_two_environment_table_is_reproduced_exactly(self):
        table = self.means[:, :2]
        decomposition = gxe.gge_decomposition(table)
        for svp in gxe.SVP_EXPONENTS:
            G, E = gxe.gge_coordinates(decomposition, svp)
            with self.subTest(svp=svp):
                np.testing.assert_allclose(G @ E.T, table - table.mean(axis=0), atol=1e-10)
        self.assertAlmostEqual(sum(decomposition["percent"]), 100.0, places=8)

    def test_sd_scaling_and_genotype_centroid(self):
        decomposition = gxe.gge_decomposition(self.means, "sd")
        np.testing.assert_allclose(decomposition["centred"].std(axis=0, ddof=1), 1.0)
        G, _ = gxe.gge_coordinates(decomposition, "symmetric")
        np.testing.assert_allclose(G.mean(axis=0), 0.0, atol=1e-10)  # origin lies inside the genotype polygon
        self.assertLess(sum(decomposition["percent"]), 100.0)

    def test_which_won_where_geometry(self):
        G, E = gxe.gge_coordinates(gxe.gge_decomposition(self.means), "symmetric")
        result = gxe.which_won_where(G, E)
        vertices = result["vertices"]
        self.assertTrue(set(result["winners"]) <= set(vertices))
        np.testing.assert_array_equal(result["winners"], (G @ E.T).argmax(axis=0))
        self.assertEqual(len(result["rays"]), len(vertices))
        for k, ray in enumerate(result["rays"]):
            a, b = G[vertices[k]], G[vertices[(k + 1) % len(vertices)]]
            with self.subTest(edge=k):
                self.assertAlmostEqual(float(np.dot(ray, b - a)), 0.0, places=10)  # sector boundary ⟂ polygon side
                self.assertGreaterEqual(float(np.dot(ray, a)), 0.0)
                self.assertAlmostEqual(float(np.dot(ray, a)), float(np.dot(ray, b)), places=10)

    def test_average_environment_axis(self):
        G, E = gxe.gge_coordinates(gxe.gge_decomposition(self.means), "genotype")
        aec = gxe.average_environment_axis(G, E)
        # the projection on the axis ranks genotypes exactly as their mean predicted value across environments
        np.testing.assert_array_equal(np.argsort(aec["projection"]), np.argsort((G @ E.T).mean(axis=1)))
        self.assertAlmostEqual(float(np.linalg.norm(aec["unit"])), 1.0)
        self.assertAlmostEqual(float(np.linalg.norm(aec["ideal_genotype"])), float(np.linalg.norm(G, axis=1).max()))
        self.assertTrue(np.all(aec["representativeness"] <= 1 + 1e-12))


class GxeEndpointTests(unittest.TestCase):
    def post(self, df, **form):
        fields = dict(genotype_col="Genotype", environment_col="Location", response_col="Yield")
        fields.update(form)
        return post_csv("/analyze/gxe", df, **fields)

    def test_replicated_trial_with_blocks(self):
        status, res = self.post(trial_frame(), rep_col="Rep")
        self.assertEqual(status, 200, res)
        sources = [row["source"] for row in res["anova"]["rows"]]
        self.assertIn("Replications within environments", sources)
        self.assertIn("IPCA1", sources)
        self.assertTrue(res["anova"]["blocks"])
        self.assertEqual(res["replicates"], 3)
        self.assertEqual(len(res["genotype_stats"]), 8)
        self.assertEqual(sorted(e for m in res["gge"]["mega_environments"] for e in m["environments"]),
                         sorted(res["environments"]))
        for key, image in res["charts"].items():
            with self.subTest(chart=key):
                self.assertTrue(base64.b64decode(image).startswith(PNG_SIGNATURE))
        best = min(res["genotype_stats"], key=lambda row: row["rank"])
        self.assertEqual(best["mean"], max(res["means"]["genotype_means"]))
        self.assertEqual(res["gge"]["svp"]["mean_stability"], "genotype")

    def test_replicates_without_block_column(self):
        status, res = self.post(trial_frame(G=6, E=4, r=2), scaling="sd", svp="symmetric")
        self.assertEqual(status, 200, res)
        self.assertFalse(res["anova"]["blocks"])
        error = next(row for row in res["anova"]["rows"] if row["source"] == "Error")
        self.assertEqual(error["df"], 6 * 4 * (2 - 1))
        self.assertEqual(set(res["gge"]["svp"].values()), {"symmetric"})

    def test_unreplicated_and_unbalanced_trials(self):
        means = trial_frame().groupby(["Genotype", "Location"], as_index=False)["Yield"].mean()
        status, res = self.post(means)
        self.assertEqual(status, 200, res)
        self.assertTrue(all(row["f"] is None for row in res["anova"]["rows"]))
        self.assertIsNone(res["genotype_stats"][0]["s2di"])
        self.assertTrue(res["notes"])
        unbalanced = trial_frame().drop(index=[0])
        status, res = self.post(unbalanced, rep_col="Rep")
        self.assertEqual(status, 200, res)
        self.assertFalse(res["balanced"])
        self.assertTrue(any("unequal" in note for note in res["notes"]))

    def test_lower_is_better_reverses_ranks(self):
        df = trial_frame()
        status, res = self.post(df, higher_is_better="false")
        self.assertEqual(status, 200, res)
        best = min(res["genotype_stats"], key=lambda row: row["rank"])
        self.assertEqual(best["mean"], min(res["means"]["genotype_means"]))

    def test_validation(self):
        df = trial_frame()
        cases = [
            (df[df["Location"].isin(["L1", "L2"])], {}),
            (df[~((df["Genotype"] == "G2") & (df["Location"] == "L3"))], {}),
            (df, dict(svp="diagonal")),
            (df, dict(scaling="range")),
            (df, dict(response_col="Genotype")),
            (df.assign(Yield=df["Yield"].astype(str).where(df.index != 5, "lost")), {}),
            (df, dict(rep_col="Missing")),
        ]
        for frame, form in cases:
            with self.subTest(form=form, rows=len(frame)):
                status, res = self.post(frame, **form)
                self.assertEqual(status, 400, res)


if __name__ == "__main__":
    unittest.main()
