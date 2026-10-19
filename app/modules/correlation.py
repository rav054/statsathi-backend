"""Pearson / Spearman correlation matrix with heatmap, scatter-matrix and mixed correlogram charts."""

import io
import base64
from typing import Optional

import pandas as pd
import numpy as np
import scipy.stats as stats
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status, Form
import matplotlib

matplotlib.use("Agg")  # headless rendering for the web server
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgba
from matplotlib.patches import Circle, Rectangle
from matplotlib.ticker import MaxNLocator
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from statsmodels.nonparametric.smoothers_lowess import lowess

from ..auth import get_current_user
from ..figure_style import COLORS, colormap_name, palette_colors
from ..models import User
from ..measures import correlation_tests
from .common import read_uploaded_table

router = APIRouter()

PLOT_TYPES = ("full", "lower", "upper", "pairs", "mixed", "target", "network")
ORDERS = ("original", "clustered")
MAX_PAIRS_VARIABLES = 12
MAX_NETWORK_VARIABLES = 40
COEFFICIENT_NORM = Normalize(vmin=-1, vmax=1)


def significance_stars(p):
    if p is None or np.isnan(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def format_coefficient(value):
    """Two decimals, without a misleading "-0.00" for tiny negative values."""
    return f"{0.0 if abs(value) < 0.005 else value:.2f}"


def fisher_interval(r, n, level=0.95):
    """Confidence interval of a correlation coefficient by Fisher's z transformation (NaN when n ≤ 3)."""
    if n <= 3 or np.isnan(r):
        return float("nan"), float("nan")
    z = np.arctanh(np.clip(r, -0.9999999, 0.9999999))
    half = stats.norm.ppf((1 + level) / 2) / np.sqrt(n - 3)
    return float(np.tanh(z - half)), float(np.tanh(z + half))


def clustered_order(r):
    """Leaf order of average-linkage clustering on the distance 1 − |r|, so related variables sit together."""
    distance = 1 - np.abs(np.nan_to_num(np.asarray(r, dtype=float), nan=0.0))
    distance = np.clip((distance + distance.T) / 2, 0, None)
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    tree = hierarchy.optimal_leaf_ordering(hierarchy.linkage(condensed, method="average"), condensed)
    return hierarchy.leaves_list(tree)


def _legible_on_white(rgba):
    """Very light colormap colours are unreadable as text on white; use the text colour instead."""
    red, green, blue = rgba[:3]
    return COLORS["text"] if 0.299 * red + 0.587 * green + 0.114 * blue > 0.75 else rgba


def draw_heatmap(corr_matrix, plot_type, method_label, cmap_name):
    mask = None
    if plot_type == "lower":
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    elif plot_type == "upper":
        mask = np.tril(np.ones_like(corr_matrix, dtype=bool))

    num_cols = len(corr_matrix.columns)
    fig_size = max(6, min(14, num_cols * 1.2))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        vmin=-1,  # fixed scale so colours mean the same coefficient in every figure
        vmax=1,
        cmap=cmap_name,
        fmt=".2f",
        square=True,
        linewidths=0.75,
        linecolor="#FFFFFF",
        cbar_kws={"shrink": 0.8, "label": f"{method_label} Correlation Coefficient"},
        ax=ax,
        annot_kws={"size": max(7, min(12, 120 // num_cols))}
    )
    ax.grid(False)  # the default style draws grid lines; not over heatmap cells
    title_suffix = " (Lower Triangle)" if plot_type == "lower" else (" (Upper Triangle)" if plot_type == "upper" else "")
    ax.set_title(f"{method_label} Correlation Matrix Heatmap{title_suffix}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    return fig


def draw_scatter_matrix(frame, r, p, n, method, method_label, cmap):
    """Pairs plot: scatter plots below the diagonal, distributions on it, coefficients above it."""
    columns = list(frame.columns)
    k = len(columns)
    side = min(14.0, max(6.5, 1.9 * k))
    fig, axes = plt.subplots(k, k, figsize=(side, side), squeeze=False)
    font = plt.rcParams["font.size"]
    point_color, line_color = palette_colors(None, 2)
    off_diagonal = ~np.eye(k, dtype=bool)
    typical_n = int(n[off_diagonal].max()) if k > 1 else 0

    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            ax.grid(False)
            ax.tick_params(labelsize=font * 0.75, labelbottom=(i == k - 1), labelleft=(j == 0 and i > 0))
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.yaxis.set_major_locator(MaxNLocator(3))

            if i == j:
                values = frame[columns[i]].dropna().to_numpy(dtype=float)
                if len(values) > 0:
                    ax.hist(values, bins="auto", density=True, color=point_color, alpha=0.3, edgecolor="white", linewidth=0.4)
                if len(values) > 2 and values.std() > 0:
                    grid = np.linspace(values.min(), values.max(), 200)
                    ax.plot(grid, stats.gaussian_kde(values)(grid), color=point_color, linewidth=1.3)
                ax.set_ylim(0, ax.get_ylim()[1] * 1.3)  # headroom for the variable name
                ax.set_yticks([])
                ax.spines["left"].set_visible(False)
                ax.text(0.5, 0.97, columns[i], transform=ax.transAxes, ha="center", va="top",
                        fontsize=font * 0.9, fontweight="bold")
            elif i > j:
                pair = frame[[columns[j], columns[i]]].dropna()
                x = pair.iloc[:, 0].to_numpy(dtype=float)
                y = pair.iloc[:, 1].to_numpy(dtype=float)
                ax.scatter(x, y, s=max(4.0, 40.0 / k), color=point_color, alpha=0.6, edgecolors="none")
                if len(x) >= 3 and x.std() > 0 and y.std() > 0:
                    if method == "spearman":
                        # Rank correlation measures monotonic association, so show a LOWESS trend
                        smooth = lowess(y, x, frac=0.75, return_sorted=True)
                        ax.plot(smooth[:, 0], smooth[:, 1], color=line_color, linewidth=1.2)
                    else:
                        slope, intercept = np.polyfit(x, y, 1)
                        ends = np.array([x.min(), x.max()])
                        ax.plot(ends, intercept + slope * ends, color=line_color, linewidth=1.2)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                value = r[i, j]
                if np.isnan(value):
                    ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center", va="center",
                            fontsize=font * 0.9, color=COLORS["muted"])
                else:
                    ax.set_facecolor(to_rgba(cmap(COEFFICIENT_NORM(value)), 0.35))
                    ax.text(0.5, 0.55, f"{format_coefficient(value)}{significance_stars(p[i, j])}", transform=ax.transAxes,
                            ha="center", va="center", fontsize=font * (0.9 + 1.1 * abs(value)),
                            fontweight="bold", color=COLORS["text"])
                    if n[i, j] < typical_n:
                        ax.text(0.5, 0.2, f"n = {n[i, j]}", transform=ax.transAxes, ha="center", va="center",
                                fontsize=font * 0.7, color=COLORS["muted"])

            if i == k - 1:
                ax.set_xlabel(columns[j], fontsize=font * 0.85)
            if j == 0 and i > 0:
                ax.set_ylabel(columns[i], fontsize=font * 0.85)

    symbol = "ρ" if method == "spearman" else "r"
    trend = "LOWESS trend" if method == "spearman" else "least-squares line"
    fig.suptitle(f"{method_label} correlation matrix", x=0.01, ha="left", fontweight="bold",
                 fontsize=plt.rcParams["axes.titlesize"])
    fig.text(0.01, 0.005,
             f"Lower: scatter plots with {trend}. Diagonal: distributions (histogram and density). "
             f"Upper: {symbol} with * p < 0.05, ** p < 0.01, *** p < 0.001.",
             fontsize=font * 0.75, color=COLORS["muted"], ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    return fig


def draw_mixed_correlogram(r, p, columns, method_label, cmap):
    """Correlogram: circles (area proportional to |coefficient|) above the diagonal, numbers below it."""
    k = len(columns)
    side = min(12.0, max(5.0, 0.7 * k + 2.5))
    fig, ax = plt.subplots(figsize=(side + 1.3, side))
    font = plt.rcParams["font.size"]
    cell_font = max(6.0, min(font, 90.0 / k))

    ax.set_xlim(-0.5, k - 0.5)
    ax.set_ylim(k - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for t in range(k + 1):
        ax.plot([-0.5, k - 0.5], [t - 0.5, t - 0.5], color=COLORS["grid"], linewidth=0.6, zorder=0)
        ax.plot([t - 0.5, t - 0.5], [-0.5, k - 0.5], color=COLORS["grid"], linewidth=0.6, zorder=0)

    for i in range(k):
        for j in range(k):
            if i == j:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#F3F4F6", edgecolor="none", zorder=0.5))
                continue
            value = r[i, j]
            if np.isnan(value):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=cell_font * 0.8, color=COLORS["muted"])
                continue
            color = cmap(COEFFICIENT_NORM(value))
            if j > i:
                ax.add_patch(Circle((j, i), radius=0.46 * np.sqrt(abs(value)), facecolor=color,
                                    edgecolor="white", linewidth=0.5, zorder=2))
                if not p[i, j] < 0.05:
                    ax.text(j, i, "×", ha="center", va="center", fontsize=cell_font * 1.2, color=COLORS["muted"], zorder=3)
            else:
                ax.text(j, i, format_coefficient(value), ha="center", va="center", fontsize=cell_font,
                        fontweight="bold", color=_legible_on_white(color))

    ax.set_xticks(range(k))
    ax.set_xticklabels(columns, rotation=45, ha="left", rotation_mode="anchor")
    ax.xaxis.tick_top()
    ax.set_yticks(range(k))
    ax.set_yticklabels(columns)
    ax.tick_params(length=0)

    colorbar = fig.colorbar(ScalarMappable(norm=COEFFICIENT_NORM, cmap=cmap), ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label(f"{method_label} coefficient")
    colorbar.outline.set_visible(False)

    fig.suptitle(f"{method_label} correlogram", x=0.01, ha="left", fontweight="bold",
                 fontsize=plt.rcParams["axes.titlesize"])
    fig.text(0.01, 0.01, "Upper: circle size and colour show the coefficient; × = not significant (p ≥ 0.05). Lower: coefficients.",
             fontsize=font * 0.75, color=COLORS["muted"], ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return fig


def draw_target_chart(columns, r, p, n, target, method, method_label, cmap):
    """Correlation of every other variable with one target, with Fisher-z 95% confidence intervals."""
    t = columns.index(target)
    others = sorted((i for i in range(len(columns)) if i != t and not np.isnan(r[t, i])), key=lambda i: r[t, i])
    font = plt.rcParams["font.size"]
    symbol = "ρ" if method == "spearman" else "r"
    fig, ax = plt.subplots(figsize=(7, max(3.6, 0.4 * len(others) + 1.8)))
    for position, i in enumerate(others):
        value = r[t, i]
        low, high = fisher_interval(value, n[t, i])
        color = cmap(COEFFICIENT_NORM(value)) if p[t, i] < 0.05 else "#9CA3AF"
        if not np.isnan(low):
            ax.hlines(position, low, high, color=color, linewidth=2.2, zorder=2)
        ax.scatter(value, position, s=60, color=color, edgecolors="white", linewidths=0.8, zorder=3)
        edge = high if not np.isnan(high) else value
        ax.text(edge if value >= 0 else (low if not np.isnan(low) else value), position,
                f"  {format_coefficient(value)}{significance_stars(p[t, i])}  ", va="center",
                ha="left" if value >= 0 else "right", fontsize=font * 0.85, color=COLORS["text"])
    ax.axvline(0, color=COLORS["axis"], linewidth=1.0, zorder=1)
    ax.set_yticks(range(len(others)), labels=[columns[i] for i in others])
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.7, len(others) - 0.3)
    ax.grid(False)
    ax.grid(True, axis="x")
    ax.set_title(f"{method_label} correlation of each variable with {target}")
    ax.set_xlabel(f"{symbol} with 95% confidence interval")
    approximate = " (approximate for Spearman)" if method == "spearman" else ""
    fig.text(0.01, 0.005, f"Bars: 95% confidence intervals by Fisher's z{approximate}. Grey: not significant (p ≥ 0.05). "
                          "* p < 0.05, ** p < 0.01, *** p < 0.001.", fontsize=font * 0.75, color=COLORS["muted"],
             ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def draw_network(columns, r, p, threshold, method, method_label, cmap):
    """Variables on a circle joined by lines for correlations with |coefficient| ≥ threshold and p < 0.05."""
    k = len(columns)
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False) + np.pi / 2
    xy = np.column_stack((np.cos(angles), np.sin(angles)))
    font = plt.rcParams["font.size"]
    symbol = "ρ" if method == "spearman" else "r"
    edges = [(i, j) for i in range(k) for j in range(i + 1, k)
             if not np.isnan(r[i, j]) and abs(r[i, j]) >= threshold and p[i, j] < 0.05]
    degree = np.zeros(k)

    fig, ax = plt.subplots(figsize=(7, 6.4))
    for i, j in sorted(edges, key=lambda e: abs(r[e])):  # strongest drawn last, on top
        value = r[i, j]
        width = 0.8 + 5.0 * (abs(value) - threshold) / max(1e-9, 1 - threshold)
        ax.plot(xy[[i, j], 0], xy[[i, j], 1], color=cmap(COEFFICIENT_NORM(value)), linewidth=width,
                alpha=0.85, solid_capstyle="round", zorder=1)
        degree[i] += 1
        degree[j] += 1
    ax.scatter(xy[:, 0], xy[:, 1], s=160 + 70 * degree, color="white", edgecolors=COLORS["text"], linewidths=1.2, zorder=3)
    for (x, y), name in zip(xy, columns):
        ax.text(1.14 * x, 1.14 * y, name, fontsize=font * 0.9, color=COLORS["text"],
                ha="left" if x > 0.15 else "right" if x < -0.15 else "center",
                va="bottom" if y > 0.15 else "top" if y < -0.15 else "center")
    if not edges:
        ax.text(0, 0, "No correlation passes the threshold", ha="center", va="center", fontsize=font, color=COLORS["muted"])
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    colorbar = fig.colorbar(ScalarMappable(norm=COEFFICIENT_NORM, cmap=cmap), ax=ax, fraction=0.04, pad=0.02)
    colorbar.set_label(f"{method_label} coefficient")
    colorbar.outline.set_visible(False)
    ax.set_title(f"{method_label} correlation network", loc="left")
    fig.text(0.01, 0.005, f"Lines: correlations with |{symbol}| ≥ {threshold:.2f} and p < 0.05 ({len(edges)} of "
                          f"{k * (k - 1) // 2} pairs); thicker lines are stronger. Node size: number of connections.",
             fontsize=font * 0.75, color=COLORS["muted"], ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


@router.post("/correlation")
def analyze_correlation(
    file: UploadFile = File(...),
    palette: str = Form("RdBu_r"),
    plot_type: str = Form("full"),  # "full", "lower", "upper" heatmaps; "pairs" scatter matrix; "mixed" correlogram
    method: str = Form("pearson"),  # "pearson" or "spearman"
    dpi: int = Form(150),  # chart resolution: 150, 300 or 600
    columns_str: Optional[str] = Form(None),  # comma-separated numeric columns; default: all numeric columns
    order: str = Form("original"),  # "original" column order or "clustered" (related variables next to each other)
    target: Optional[str] = Form(None),  # target variable for plot_type "target"
    threshold: float = Form(0.3),  # minimum |coefficient| drawn in plot_type "network"
    current_user: User = Depends(get_current_user)
):
    df = read_uploaded_table(file)

    if columns_str:
        requested = list(dict.fromkeys(c.strip() for c in columns_str.split(",") if c.strip()))
        missing = [c for c in requested if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Column(s) not found in dataset: {', '.join(missing)}."
            )
        non_numeric = [c for c in requested if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Correlation needs numeric columns; these are not numeric: {', '.join(non_numeric)}."
            )
        numeric_df = df[requested]
    else:
        numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded dataset does not contain any numeric columns. Please upload a dataset with numeric variables."
        )

    if len(numeric_df.columns) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Correlation analysis requires at least 2 numeric columns."
        )

    method = (method or "pearson").strip().lower()
    if method not in ("pearson", "spearman"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Correlation method must be 'pearson' or 'spearman'."
        )
    plot_type = (plot_type or "full").strip().lower()
    if plot_type not in PLOT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Chart type must be one of: {', '.join(PLOT_TYPES)}."
        )
    if plot_type == "pairs" and len(numeric_df.columns) > MAX_PAIRS_VARIABLES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(f"The scatter matrix shows at most {MAX_PAIRS_VARIABLES} variables (this selection has "
                    f"{len(numeric_df.columns)}). Select fewer variables, or use a heatmap or correlogram.")
        )

    order = (order or "original").strip().lower()
    if order not in ORDERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Variable order must be one of: {', '.join(ORDERS)}."
        )
    if plot_type == "target" and (not target or target not in numeric_df.columns):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Choose the target variable: one of the numeric columns included in the analysis."
        )
    if plot_type == "network":
        if not 0 < threshold < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The network threshold must be between 0 and 1 (for example 0.3)."
            )
        if len(numeric_df.columns) > MAX_NETWORK_VARIABLES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"The correlation network shows at most {MAX_NETWORK_VARIABLES} variables; select fewer."
            )

    method_label = "Pearson" if method == "pearson" else "Spearman"
    r_values, p_values, n_values = correlation_tests(numeric_df, method)
    columns = list(numeric_df.columns)
    if order == "clustered" and len(columns) > 2:
        index = clustered_order(r_values)
        columns = [columns[i] for i in index]
        numeric_df = numeric_df[columns]
        r_values = r_values[np.ix_(index, index)]
        p_values = p_values[np.ix_(index, index)]
        n_values = n_values[np.ix_(index, index)]
    corr_matrix = pd.DataFrame(r_values, index=columns, columns=columns)
    cmap_name = colormap_name(palette)

    fig = None
    try:
        if plot_type == "target":
            fig = draw_target_chart(columns, r_values, p_values, n_values, target, method, method_label,
                                    matplotlib.colormaps[cmap_name])
        elif plot_type == "network":
            fig = draw_network(columns, r_values, p_values, threshold, method, method_label,
                               matplotlib.colormaps[cmap_name])
        elif plot_type == "pairs":
            fig = draw_scatter_matrix(numeric_df, r_values, p_values, n_values, method, method_label,
                                      matplotlib.colormaps[cmap_name])
        elif plot_type == "mixed":
            fig = draw_mixed_correlogram(r_values, p_values, list(numeric_df.columns), method_label,
                                         matplotlib.colormaps[cmap_name])
        else:
            fig = draw_heatmap(corr_matrix, plot_type, method_label, cmap_name)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=max(72, min(int(dpi), 600)))
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating the correlation chart: {str(e)}"
        )
    finally:
        if fig is not None:
            plt.close(fig)

    matrix_data = {
        "columns": corr_matrix.columns.tolist(),
        "index": corr_matrix.index.tolist(),
        "values": corr_matrix.replace({np.nan: None}).values.tolist()
    }

    return {
        "matrix": matrix_data,
        "method": method,
        "plot_type": plot_type,
        "order": order,
        "p_values": pd.DataFrame(p_values).replace({np.nan: None}).values.tolist(),
        "n": n_values.tolist(),
        "plot": img_b64
    }
