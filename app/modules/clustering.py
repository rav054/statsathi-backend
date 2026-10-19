"""K-means and hierarchical clustering: linkage methods, distance measures, dendrogram styles and choosing k."""

from typing import Optional

import pandas as pd
import numpy as np
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status, Form

from ..auth import get_current_user
from ..models import User
from ..measures import cluster_quality

from .common import read_uploaded_table
from . import chart_gallery, cluster_charts

router = APIRouter()

LINKAGES = {
    "ward": "Ward",
    "complete": "complete linkage",
    "average": "average linkage (UPGMA)",
    "weighted": "weighted average (WPGMA)",
    "single": "single linkage",
    "centroid": "centroid (UPGMC)",
    "median": "median (WPGMC)",
}
DISTANCES = {"euclidean": "Euclidean", "cityblock": "Manhattan", "correlation": "1 − Pearson r", "cosine": "cosine"}
EUCLIDEAN_ONLY = ("ward", "centroid", "median")
MAX_K_SEARCH = 10
MAX_TREE_CHART_ROWS = 1500
SILHOUETTE_SAMPLE = 2000


def bad_request(message):
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)


@router.post("/cluster")
def analyze_clustering(
    file: UploadFile = File(...),
    variables_str: str = Form(...),
    method: str = Form(...),
    k: int = Form(3),
    cut_height: Optional[float] = Form(None),
    linkage: str = Form("ward"),
    distance: str = Form("euclidean"),
    label_col: Optional[str] = Form(None),  # labels for the dendrogram leaves
    palette: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user)
):
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import calinski_harabasz_score, silhouette_samples, silhouette_score
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import pdist

    if linkage not in LINKAGES:
        raise bad_request(f"Linkage must be one of: {', '.join(LINKAGES)}.")
    if distance not in DISTANCES:
        raise bad_request(f"Distance must be one of: {', '.join(DISTANCES)}.")
    if linkage in EUCLIDEAN_ONLY and distance != "euclidean":
        raise bad_request(f"{LINKAGES[linkage].capitalize()} needs Euclidean distances.")

    df = read_uploaded_table(file)

    selected_cols = [c.strip() for c in variables_str.split(',') if c.strip()]
    if not selected_cols:
        raise bad_request("No variables selected for clustering.")
    missing_cols = [c for c in selected_cols if c not in df.columns]
    if missing_cols:
        raise bad_request(f"Variables not found in dataset: {', '.join(missing_cols)}")
    label_col = label_col or None
    if label_col and label_col not in df.columns:
        raise bad_request(f"Label column '{label_col}' not found in dataset.")

    # Clean missing values
    df_clean = df.dropna(subset=selected_cols).copy()
    if len(df_clean) < 3:
        raise bad_request("Clustering requires at least 3 valid observations after removing missing values.")

    X = df_clean[selected_cols].values
    n = len(df_clean)
    if not 2 <= k <= n - 1:
        raise bad_request(f"Number of clusters must be between 2 and {n - 1} for {n} observations.")
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    except Exception as e:
        raise bad_request(f"Failed to standardize variables: {str(e)}")

    distances = pdist(X_scaled, metric=distance)
    if not np.all(np.isfinite(distances)):
        raise bad_request(f"{DISTANCES[distance]} distances are undefined for some observations (for example rows with the "
                          "same standardized value on every variable). Choose Euclidean or Manhattan distance.")

    # 1. K-Means
    try:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X_scaled)
        kmeans_labels = kmeans.labels_.tolist()
        kmeans_centroids_scaled = kmeans.cluster_centers_.tolist()
        kmeans_centroids = scaler.inverse_transform(kmeans.cluster_centers_).tolist()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"K-Means execution failed: {str(e)}")

    # 2. Hierarchical
    try:
        linkage_matrix = hierarchy.linkage(distances, method=linkage)
        if cut_height is not None:
            hierarchical_labels = hierarchy.fcluster(linkage_matrix, t=cut_height, criterion='distance').tolist()
        else:
            hierarchical_labels = hierarchy.fcluster(linkage_matrix, t=k, criterion='maxclust').tolist()
        min_label = min(hierarchical_labels)
        hierarchical_labels = [label - min_label for label in hierarchical_labels]
        n_clusters = len(set(hierarchical_labels))
        cut_used = float(cut_height) if cut_height is not None else cluster_charts.cut_between(linkage_matrix, n_clusters)

        # Scipy dendrogram coordinates for the interactive chart, coloured below the cut
        dend_dict = hierarchy.dendrogram(linkage_matrix, no_plot=True, color_threshold=cut_used)
        dendrogram_data = {
            "icoord": [list(x) for x in dend_dict["icoord"]],
            "dcoord": [list(y) for y in dend_dict["dcoord"]],
            "ivl": dend_dict["ivl"],
            "color_list": dend_dict["color_list"],
            "leaves": dend_dict["leaves"]
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Hierarchical clustering execution failed: {str(e)}")

    # 3. PCA for Biplot
    try:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        pc1_coords = X_pca[:, 0].tolist()
        pc2_coords = X_pca[:, 1].tolist()
        loadings = pca.components_.T
        loadings_payload = [{"variable": col, "pc1": float(loadings[idx, 0]), "pc2": float(loadings[idx, 1])}
                            for idx, col in enumerate(selected_cols)]
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"PCA projection failed: {str(e)}")

    # 4. Summaries
    df_clean['kmeans_cluster'] = kmeans_labels
    kmeans_summaries = []
    for c_id in range(k):
        c_mean = df_clean[df_clean['kmeans_cluster'] == c_id][selected_cols].mean()
        kmeans_summaries.append({
            "cluster_id": c_id,
            "means": c_mean.replace({np.nan: None}).to_dict(),
            "count": int(sum(df_clean['kmeans_cluster'] == c_id))
        })

    df_clean['hierarchical_cluster'] = hierarchical_labels
    hierarchical_summaries = []
    for c_id in sorted(set(hierarchical_labels)):
        c_mean = df_clean[df_clean['hierarchical_cluster'] == c_id][selected_cols].mean()
        hierarchical_summaries.append({
            "cluster_id": c_id,
            "means": c_mean.replace({np.nan: None}).to_dict(),
            "count": int(sum(df_clean['hierarchical_cluster'] == c_id))
        })

    clean_records = []
    for i, idx in enumerate(df_clean.index):
        row_dict = df_clean.loc[idx].replace({np.nan: None}).to_dict()
        row_dict['original_index'] = int(idx)
        row_dict['excel_row'] = int(idx) + 2
        row_dict['kmeans_cluster'] = kmeans_labels[i]
        row_dict['hierarchical_cluster'] = hierarchical_labels[i]
        clean_records.append(row_dict)

    def silhouette_values(labels, metric="euclidean"):
        """Silhouette width of every observation (needs 2 to n - 1 clusters)."""
        if 2 <= len(set(labels)) <= len(labels) - 1:
            return [float(v) for v in silhouette_samples(X_scaled, labels, metric=metric)]
        return None

    def scaled_profiles(labels):
        """Cluster means of the standardized variables (0 = overall mean, 1 = one standard deviation)."""
        frame = pd.DataFrame(X_scaled, columns=selected_cols)
        frame["cluster"] = labels
        means = frame.groupby("cluster")[selected_cols].mean()
        return {int(c): {col: float(means.loc[c, col]) for col in selected_cols} for c in means.index}

    # 5. Linkage methods compared by how well their trees preserve the distances
    linkage_comparison = []
    for name, label in LINKAGES.items():
        if name in EUCLIDEAN_ONLY and distance != "euclidean":
            continue
        tree = hierarchy.linkage(distances, method=name)
        linkage_comparison.append({"linkage": name, "label": label,
                                   "cophenetic_correlation": float(hierarchy.cophenet(tree, distances)[0])})

    # 6. Choosing k: silhouette widths, K-means within-cluster SS and Calinski-Harabasz index
    sample = {"sample_size": SILHOUETTE_SAMPLE, "random_state": 0} if n > SILHOUETTE_SAMPLE else {}

    def average_silhouette(labels, metric="euclidean"):
        return float(silhouette_score(X_scaled, labels, metric=metric, **sample)) if 2 <= len(set(labels)) <= n - 1 else None

    def calinski(labels):
        return float(calinski_harabasz_score(X_scaled, labels)) if 2 <= len(set(labels)) <= n - 1 else None

    k_rows = []
    for candidate in range(2, min(MAX_K_SEARCH, n - 1) + 1):
        tree_labels = hierarchy.fcluster(linkage_matrix, t=candidate, criterion="maxclust")
        search = KMeans(n_clusters=candidate, random_state=42, n_init="auto").fit(X_scaled)
        k_rows.append({
            "k": candidate,
            "hierarchical_silhouette": average_silhouette(tree_labels, distance),
            "kmeans_silhouette": average_silhouette(search.labels_),
            "kmeans_wss": float(search.inertia_),
            "hierarchical_calinski": calinski(tree_labels),
            "kmeans_calinski": calinski(search.labels_),
        })

    def best_k(key):
        scored = [row for row in k_rows if row[key] is not None]
        return max(scored, key=lambda row: row[key])["k"] if scored else None

    suggested = {"hierarchical": best_k("hierarchical_silhouette"), "kmeans": best_k("kmeans_silhouette")}

    # 7. Tree charts
    leaf_labels = (chart_gallery.as_category(df.loc[df_clean.index, label_col].fillna("")).tolist() if label_col
                   else [str(int(idx) + 2) for idx in df_clean.index])
    quality = cluster_quality(X_scaled, linkage_matrix, kmeans_labels, hierarchical_labels, kmeans.inertia_,
                              distances=distances, metric=distance)
    method_label = f"{LINKAGES[linkage]}, {DISTANCES[distance]} distance"
    notes = []
    large = n > MAX_TREE_CHART_ROWS
    if large:
        notes.append(f"The circular dendrogram and heatmap are drawn for up to {MAX_TREE_CHART_ROWS} observations.")
    charts = {
        "dendrogram": cluster_charts.dendrogram_chart(linkage_matrix, leaf_labels, hierarchical_labels, cut_used,
                                                      method_label, quality.get("cophenetic_correlation"), palette),
        "circular": None if large else cluster_charts.circular_dendrogram_chart(
            linkage_matrix, leaf_labels, hierarchical_labels, cut_used, method_label, palette),
        "heatmap": None if large else cluster_charts.heatmap_dendrogram_chart(
            X_scaled, selected_cols, leaf_labels, linkage_matrix, hierarchical_labels, palette),
        "k_selection": cluster_charts.k_selection_chart(k_rows, suggested, palette) if k_rows else None,
    }

    return {
        "variables": selected_cols,
        "kmeans": {
            "labels": kmeans_labels,
            "centroids": kmeans_centroids,
            "centroids_scaled": kmeans_centroids_scaled,
            "summaries": kmeans_summaries
        },
        "hierarchical": {
            "labels": hierarchical_labels,
            "dendrogram": dendrogram_data,
            "summaries": hierarchical_summaries,
            "cut_height_used": cut_used,
            "linkage": linkage,
            "linkage_label": LINKAGES[linkage],
            "distance": distance,
            "distance_label": DISTANCES[distance],
        },
        "pca": {
            "pc1": pc1_coords,
            "pc2": pc2_coords,
            "loadings": loadings_payload,
            "explained_variance": pca.explained_variance_ratio_.tolist()
        },
        "records": clean_records,
        "quality": quality,
        "silhouette": {
            "kmeans": silhouette_values(kmeans_labels),
            "hierarchical": silhouette_values(hierarchical_labels, distance),
        },
        "profiles_scaled": {
            "kmeans": scaled_profiles(kmeans_labels),
            "hierarchical": scaled_profiles(hierarchical_labels),
        },
        "linkage_comparison": linkage_comparison,
        "k_selection": {"rows": k_rows, "suggested": suggested},
        "leaf_labels": leaf_labels,
        "charts": charts,
        "notes": notes,
    }
