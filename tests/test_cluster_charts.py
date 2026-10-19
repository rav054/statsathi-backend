"""Data behind the clustering charts: silhouette widths and standardized cluster profiles.

Run from backend/:  python -m unittest discover -s tests -t .
"""
import base64
import unittest

from tests.support import post_csv  # must come before other app imports
from tests.test_chart_gallery import field_frame  # noqa: E402
from tests.test_endpoints_smoke import PNG_SIGNATURE  # noqa: E402

import numpy as np  # noqa: E402
from scipy.cluster import hierarchy  # noqa: E402
from scipy.spatial.distance import pdist  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import silhouette_samples, silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from app.modules import cluster_charts  # noqa: E402

VARIABLES = ["Yield", "Height", "Protein"]


class ClusterChartDataTests(unittest.TestCase):
    def setUp(self):
        self.df = field_frame()
        status, self.res = post_csv("/analyze/cluster", self.df, variables_str=",".join(VARIABLES), method="kmeans", k="3")
        self.assertEqual(status, 200, self.res)
        self.scaled = StandardScaler().fit_transform(self.df[VARIABLES].to_numpy(dtype=float))

    def test_silhouette_widths_match_sklearn(self):
        for method in ("kmeans", "hierarchical"):
            with self.subTest(method=method):
                labels = self.res[method]["labels"]
                np.testing.assert_allclose(self.res["silhouette"][method], silhouette_samples(self.scaled, labels), rtol=1e-10)
                self.assertAlmostEqual(np.mean(self.res["silhouette"][method]), silhouette_score(self.scaled, labels), places=10)

    def test_scaled_profiles_are_cluster_means_of_z_scores(self):
        labels = np.asarray(self.res["kmeans"]["labels"])
        profiles = self.res["profiles_scaled"]["kmeans"]
        for cluster, means in profiles.items():
            members = self.scaled[labels == int(cluster)]
            np.testing.assert_allclose([means[v] for v in VARIABLES], members.mean(axis=0), rtol=1e-10, atol=1e-12)
        weights = np.bincount(labels)
        overall = sum(weights[int(c)] * np.array([m[v] for v in VARIABLES]) for c, m in profiles.items()) / labels.size
        np.testing.assert_allclose(overall, 0, atol=1e-10)  # z-scores average to zero across clusters


class ClusterTreeUpgradeTests(unittest.TestCase):
    def setUp(self):
        self.df = field_frame()
        self.scaled = StandardScaler().fit_transform(self.df[VARIABLES].to_numpy(dtype=float))

    def post(self, **form):
        fields = dict(variables_str=",".join(VARIABLES), method="hierarchical", k="3")
        fields.update(form)
        return post_csv("/analyze/cluster", self.df, **fields)

    def test_default_ward_tree_is_unchanged(self):
        status, res = self.post()
        self.assertEqual(status, 200, res)
        tree = hierarchy.linkage(self.scaled, method="ward")
        expected = hierarchy.fcluster(tree, 3, "maxclust")
        np.testing.assert_array_equal(res["hierarchical"]["labels"], expected - expected.min())
        self.assertAlmostEqual(res["quality"]["cophenetic_correlation"], hierarchy.cophenet(tree, pdist(self.scaled))[0], places=10)
        cut = res["hierarchical"]["cut_height_used"]
        self.assertEqual(np.unique(hierarchy.fcluster(tree, cut, "distance")).size, 3)  # the drawn cut gives the 3 clusters

    def test_linkage_distance_and_labels(self):
        status, res = self.post(linkage="average", distance="cityblock", label_col="Treatment")
        self.assertEqual(status, 200, res)
        distances = pdist(self.scaled, "cityblock")
        tree = hierarchy.linkage(distances, "average")
        expected = hierarchy.fcluster(tree, 3, "maxclust")
        np.testing.assert_array_equal(res["hierarchical"]["labels"], expected - expected.min())
        self.assertAlmostEqual(res["quality"]["cophenetic_correlation"], hierarchy.cophenet(tree, distances)[0], places=10)
        np.testing.assert_allclose(res["silhouette"]["hierarchical"], silhouette_samples(self.scaled, expected, metric="cityblock"),
                                   rtol=1e-10)
        comparison = {row["linkage"]: row["cophenetic_correlation"] for row in res["linkage_comparison"]}
        self.assertNotIn("ward", comparison)  # Ward needs Euclidean distances
        self.assertAlmostEqual(comparison["complete"], hierarchy.cophenet(hierarchy.linkage(distances, "complete"), distances)[0],
                               places=10)
        self.assertEqual(res["leaf_labels"][:4], self.df["Treatment"].astype(str).tolist()[:4])
        for key in ("dendrogram", "circular", "heatmap", "k_selection"):
            with self.subTest(chart=key):
                self.assertTrue(base64.b64decode(res["charts"][key]).startswith(PNG_SIGNATURE))

    def test_choosing_k(self):
        status, res = self.post(method="kmeans")
        self.assertEqual(status, 200, res)
        tree = hierarchy.linkage(self.scaled, method="ward")
        rows = res["k_selection"]["rows"]
        self.assertEqual([row["k"] for row in rows], list(range(2, 11)))
        for row in rows[:4]:
            with self.subTest(k=row["k"]):
                labels = hierarchy.fcluster(tree, row["k"], "maxclust")
                self.assertAlmostEqual(row["hierarchical_silhouette"], silhouette_score(self.scaled, labels), places=10)
                search = KMeans(n_clusters=row["k"], random_state=42, n_init="auto").fit(self.scaled)
                self.assertAlmostEqual(row["kmeans_wss"], search.inertia_, places=6)
                self.assertAlmostEqual(row["kmeans_silhouette"], silhouette_score(self.scaled, search.labels_), places=10)
        best = max(rows, key=lambda row: row["hierarchical_silhouette"])["k"]
        self.assertEqual(res["k_selection"]["suggested"]["hierarchical"], best)

    def test_link_colours_follow_clusters(self):
        tree = hierarchy.linkage(self.scaled, method="ward")
        labels = hierarchy.fcluster(tree, 4, "maxclust") - 1
        color = cluster_charts.link_color_function(tree, labels, ["a", "b", "c", "d"])
        n = len(labels)
        for node in range(n, 2 * n - 1):
            leaves = hierarchy.to_tree(tree).pre_order() if node == 2 * n - 2 else None
            if leaves is not None:
                self.assertIsNone(color.owner[node])  # the root spans several clusters
        for i, row in enumerate(tree):
            owner = color.owner[n + i]
            if owner is not None:
                self.assertEqual(color.owner[int(row[0])], owner)
                self.assertEqual(color.owner[int(row[1])], owner)
        self.assertAlmostEqual(cluster_charts.cut_between(tree, 4), (np.sort(tree[:, 2])[n - 5] + np.sort(tree[:, 2])[n - 4]) / 2)

    def test_invalid_tree_options(self):
        for form in (dict(linkage="ward", distance="correlation"), dict(linkage="bogus"), dict(distance="bogus"),
                     dict(label_col="Nope")):
            with self.subTest(form=form):
                status, res = self.post(**form)
                self.assertEqual(status, 400, res)


if __name__ == "__main__":
    unittest.main()
