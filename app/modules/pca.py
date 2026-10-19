"""Principal component analysis."""

from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status, Form

from ..auth import get_current_user
from ..figure_style import PALETTES
from ..models import User
from ..measures import pca_decomposition

from .common import read_uploaded_table
from . import chart_gallery, pca_charts

router = APIRouter()

# Older PCA palette names: point colour and arrow colour when the biplot has no groups
LEGACY_COLORS = {
    "Oranges": ("#4F46E5", "#F97316"),
    "Blues": ("#0284C7", "#10B981"),
    "Greens": ("#10B981", "#F97316"),
    "coolwarm": ("#4F46E5", "#EF4444"),
    "Purples": ("#7C3AED", "#EC4899"),
    "magma": ("#3B0F70", "#FE9F6D"),
}
MAX_HEATMAP_COMPONENTS = 10


def bad_request(message):
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)


def biplot_colors(palette):
    if palette in LEGACY_COLORS:
        return LEGACY_COLORS[palette]
    if palette in PALETTES:
        return tuple(PALETTES[palette]["colors"][:2])
    return LEGACY_COLORS["Oranges"]


@router.post("/pca")
def analyze_pca(
    file: UploadFile = File(...),
    columns_str: str = Form(...),  # comma-separated list of columns
    scale: bool = Form(True),      # standardize data
    hue_var: Optional[str] = Form(None), # grouping variable for biplot colors
    palette: Optional[str] = Form("Oranges"), # Color palette choice
    x_component: int = Form(1),  # biplot axes (1-based)
    y_component: int = Form(2),
    biplot_type: str = Form("distance"),  # distance or correlation scaling
    ellipse: str = Form("confidence"),  # group outline: none, confidence, data or hull
    label_col: Optional[str] = Form(None),  # column used to label observations
    rotate_components: Optional[int] = Form(None),  # components for varimax (default: parallel analysis / Kaiser)
    current_user: User = Depends(get_current_user)
):
    if biplot_type not in pca_charts.BIPLOT_TYPES:
        raise bad_request("Biplot type must be 'distance' or 'correlation'.")
    if ellipse not in pca_charts.ELLIPSES:
        raise bad_request("Group outline must be none, confidence, data or hull.")
    df = read_uploaded_table(file)

    pca_cols = list(dict.fromkeys([c.strip() for c in columns_str.split(",") if c.strip()]))
    if len(pca_cols) < 2:
        raise bad_request("PCA analysis requires at least 2 variables.")
    for col in pca_cols:
        if col not in df.columns:
            raise bad_request(f"Column '{col}' not found in dataset.")
    hue_var = hue_var if hue_var and hue_var in df.columns else None
    label_col = label_col or None
    if label_col and label_col not in df.columns:
        raise bad_request(f"Label column '{label_col}' not found in dataset.")

    cols_to_use = list(pca_cols)
    if hue_var and hue_var not in cols_to_use:
        cols_to_use.append(hue_var)
    df_clean = df[cols_to_use].dropna()
    if len(df_clean) < 3:
        raise bad_request("PCA requires at least 3 valid observations.")

    df_pca_data = df_clean[pca_cols]
    try:
        decomposition = pca_decomposition(df_pca_data.to_numpy(dtype=float), scale=scale)
    except ValueError as e:
        raise bad_request(f"PCA variables must be numeric and not constant: {e}")

    X_projected = decomposition["scores"]
    eigenvectors = decomposition["eigenvectors"]
    components = eigenvectors.T  # rows = components, as in sklearn's components_
    eigenvalues = [float(v) for v in decomposition["eigenvalues"]]
    explained_variance_ratio = [float(v) for v in decomposition["explained_variance_ratio"]]
    cumulative_variance_ratio = [float(v) for v in np.cumsum(explained_variance_ratio)]
    n_components = len(eigenvalues)
    correlation_loading_matrix = decomposition["correlation_loadings"]

    if not (1 <= x_component <= n_components and 1 <= y_component <= n_components) or x_component == y_component:
        raise bad_request(f"Choose two different components between PC1 and PC{n_components} for the biplot.")
    plane = (x_component - 1, y_component - 1)

    loadings_dict = {col: [float(v) for v in components[:, i]] for i, col in enumerate(pca_cols)}
    correlation_loadings = {col: [float(v) for v in correlation_loading_matrix[i]] for i, col in enumerate(pca_cols)}
    pc_names = [f"PC{i+1}" for i in range(n_components)]
    sample_scores = [[float(val) for val in row] for row in X_projected]

    groups = chart_gallery.as_category(df_clean[hue_var]).tolist() if hue_var else None
    labels = chart_gallery.as_category(df.loc[df_clean.index, label_col].fillna("")).tolist() if label_col else None
    img_b64, arrow_scale = pca_charts.biplot_chart(
        X_projected, eigenvectors, eigenvalues, correlation_loading_matrix, pca_cols, explained_variance_ratio, plane,
        biplot_type, groups, hue_var, labels, ellipse, palette, biplot_colors(palette))

    # Parallel analysis applies to correlation-matrix PCA (standardized variables)
    parallel = None
    thresholds = None
    if scale:
        thresholds, iterations = pca_charts.parallel_analysis(len(df_pca_data), len(pca_cols))
        thresholds = thresholds[:n_components]
        parallel = {
            "thresholds": [float(v) for v in thresholds],
            "retained": pca_charts.retained_components(eigenvalues, thresholds),
            "iterations": int(iterations),
            "percentile": 95,
        }
    kaiser = int(sum(v >= 1.0 for v in eigenvalues)) if scale else None

    # Varimax rotation of the correlation loadings of the retained components
    if rotate_components is not None:
        if not 2 <= rotate_components <= n_components:
            raise bad_request(f"Varimax rotation needs between 2 and {n_components} components.")
        rotated_count = rotate_components
    else:
        rotated_count = min(max(parallel["retained"] if parallel else 2, 2), n_components)
    rotated, _ = pca_charts.varimax(correlation_loading_matrix[:, :rotated_count])
    rotated, rotated_variance = pca_charts.ordered_rotation(rotated)
    rotated_names = [f"RC{i + 1}" for i in range(rotated_count)]
    varimax = {
        "components": rotated_count,
        "names": rotated_names,
        "loadings": {col: [float(v) for v in rotated[i]] for i, col in enumerate(pca_cols)},
        "variance": [float(v) for v in rotated_variance],
        "percent": [float(100 * v / len(pca_cols)) for v in rotated_variance],
        "communalities": {col: float((rotated[i] ** 2).sum()) for i, col in enumerate(pca_cols)},
    }

    cos2, observation_contributions = pca_charts.observation_quality(X_projected, eigenvalues)
    a, b = plane
    heatmap_count = min(max(rotated_count, 2), n_components, MAX_HEATMAP_COMPONENTS)
    contributions = pca_charts.variable_contributions(components)
    charts = {
        "scree": pca_charts.scree_chart(eigenvalues, explained_variance_ratio, thresholds, scale, palette),
        "correlation_circle": pca_charts.correlation_circle_chart(correlation_loading_matrix, pca_cols,
                                                                  explained_variance_ratio, palette, plane),
        "contributions": pca_charts.contributions_chart(components, pca_cols, explained_variance_ratio, palette, plane),
        "loadings_heatmap": pca_charts.loadings_heatmap_chart(correlation_loading_matrix, pca_cols,
                                                              explained_variance_ratio, heatmap_count, palette),
        "varimax_heatmap": pca_charts.loadings_heatmap_chart(
            rotated, pca_cols, explained_variance_ratio, rotated_count, palette, title="Varimax-rotated loadings",
            labels=[f"{name}\n{share:.1f}%" for name, share in zip(rotated_names, varimax["percent"])],
            note="Varimax rotation (Kaiser normalization) of the correlation loadings, so each variable loads mainly on "
                 "one component. Bold: |r| ≥ 0.5."),
    }

    return {
        "parallel_analysis": parallel,
        "contributions": {pc_names[k]: {col: float(contributions[k, i]) for i, col in enumerate(pca_cols)}
                          for k in range(n_components)},
        "charts": charts,
        "eigenvalues": eigenvalues,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_variance_ratio": cumulative_variance_ratio,
        "loadings": loadings_dict,
        "correlation_loadings": correlation_loadings,
        "kaiser_retained": kaiser,
        "scaled": bool(scale),
        "pc_names": pc_names,
        "variable_names": pca_cols,
        "sample_scores": sample_scores,
        "biplot": {"x_component": x_component, "y_component": y_component, "type": biplot_type, "ellipse": ellipse,
                   "arrow_scale": arrow_scale, "label_column": label_col},
        "varimax": varimax,
        "observation_quality": {
            "components": [x_component, y_component],
            "cos2_plane": [float(v) for v in cos2[:, a] + cos2[:, b]],
            "contribution_x": [float(v) for v in observation_contributions[:, a]],
            "contribution_y": [float(v) for v in observation_contributions[:, b]],
            "labels": labels,
        },
        "plot": img_b64
    }
