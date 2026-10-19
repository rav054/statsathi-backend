"""Non-parametric tests: Mann-Whitney, Wilcoxon, Kruskal-Wallis, Friedman, chi-square."""

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

from ..auth import get_current_user
from ..figure_style import palette_colors
from ..models import User
from ..measures import chi_square_independence, rank_biserial_independent, rank_biserial_paired

from .common import read_uploaded_table

router = APIRouter()

@router.post("/nonparametric")
def analyze_nonparametric(
    file: UploadFile = File(...),
    test_type: str = Form(...),  # "mann_whitney", "wilcoxon", "kruskal_wallis", "friedman", "chi_square"
    col1: Optional[str] = Form(None),
    col2: Optional[str] = Form(None),
    cols_str: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user)
):
    df = read_uploaded_table(file)

    # Parse columns
    if cols_str:
        cols = [c.strip() for c in cols_str.split(",") if c.strip()]
    else:
        cols = []
        if col1:
            cols.append(col1)
        if col2:
            cols.append(col2)

    if not cols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one variable must be specified for analysis."
        )

    for c in cols:
        if c not in df.columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Column '{c}' not found in dataset."
            )

    result_stats = {}
    normality_results = {}
    plot_buf = io.BytesIO()

    try:
        plt.clf()
        plt.close('all')
        fig, ax = plt.subplots(figsize=(7, 5))

        # Dynamic Routing for Independent tests
        actual_test_type = test_type
        if test_type in ["mann_whitney", "kruskal_wallis"]:
            if len(cols) == 2:
                actual_test_type = "mann_whitney"
            elif len(cols) >= 3:
                actual_test_type = "kruskal_wallis"
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Independent groups comparison requires at least 2 variables."
                )

        if actual_test_type == "mann_whitney":
            c1, c2 = cols[0], cols[1]
            v1 = df[c1].dropna()
            v2 = df[c2].dropna()
            if len(v1) < 2 or len(v2) < 2:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Both columns must contain at least 2 data values."
                )

            # Shapiro-Wilk Normality checks (Justification reference)
            for col_name, v_data in [(c1, v1), (c2, v2)]:
                if len(v_data) >= 3:
                    w_stat, w_p = stats.shapiro(v_data)
                    normality_results[col_name] = {"stat": float(w_stat), "p_value": float(w_p), "normal": bool(w_p > 0.05)}
                else:
                    normality_results[col_name] = {"error": "Sample size too small for normality test (n < 3)"}

            u_stat, p_val = stats.mannwhitneyu(v1, v2, alternative='two-sided')
            
            # Median, Q1, Q3, IQR
            median1 = float(v1.median())
            q25_1 = float(np.percentile(v1, 25))
            q75_1 = float(np.percentile(v1, 75))
            iqr_1 = q75_1 - q25_1

            median2 = float(v2.median())
            q25_2 = float(np.percentile(v2, 25))
            q75_2 = float(np.percentile(v2, 75))
            iqr_2 = q75_2 - q25_2

            # Rank-biserial correlation: P(group1 > group2) - P(group1 < group2)
            n1 = len(v1)
            n2 = len(v2)
            effect_size = rank_biserial_independent(u_stat, n1, n2)

            result_stats = {
                "test_name": "Mann-Whitney U Test",
                "statistic": float(u_stat),
                "p_value": float(p_val),
                "median_group1": median1,
                "q25_group1": q25_1,
                "q75_group1": q75_1,
                "iqr_group1": iqr_1,
                "median_group2": median2,
                "q25_group2": q25_2,
                "q75_group2": q75_2,
                "iqr_group2": iqr_2,
                "effect_size": float(effect_size),
                "effect_size_name": "Rank-biserial correlation",
                "n_group1": n1,
                "n_group2": n2,
                "significant": bool(p_val < 0.05)
            }

            # Plot Violin
            plot_df = pd.DataFrame({
                "Value": pd.concat([v1, v2]),
                "Variable": [c1] * len(v1) + [c2] * len(v2)
            })
            sns.violinplot(data=plot_df, x="Variable", y="Value", palette=palette_colors(None, 2), ax=ax, inner="quartile", width=0.5)
            sns.stripplot(data=plot_df, x="Variable", y="Value", color="#1E293B", size=4, jitter=0.1, alpha=0.4, ax=ax)
            ax.set_title(f"Mann-Whitney Violin Comparison: {c1} vs {c2}", fontsize=12, fontweight="bold", pad=15)
            ax.set_xlabel("Variable", fontsize=10)
            ax.set_ylabel("Value", fontsize=10)

        elif actual_test_type == "kruskal_wallis":
            groups_data = [df[c].dropna() for c in cols]
            for i, c in enumerate(cols):
                if len(groups_data[i]) < 2:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Column '{c}' must contain at least 2 data values."
                    )

            # Shapiro-Wilk for reference
            for c, values in zip(cols, groups_data):
                if len(values) >= 3:
                    w_stat, w_p = stats.shapiro(values)
                    normality_results[c] = {"stat": float(w_stat), "p_value": float(w_p), "normal": bool(w_p > 0.05)}
                else:
                    normality_results[c] = {"error": "Sample size too small for normality test (n < 3)"}

            h_stat, p_val = stats.kruskal(*groups_data)

            # Descriptives for all groups dynamically
            medians = {c: float(groups_data[i].median()) for i, c in enumerate(cols)}
            q25 = {c: float(np.percentile(groups_data[i], 25)) for i, c in enumerate(cols)}
            q75 = {c: float(np.percentile(groups_data[i], 75)) for i, c in enumerate(cols)}
            iqr = {c: (q75[c] - q25[c]) for c in cols}
            sizes = {c: len(groups_data[i]) for i, c in enumerate(cols)}

            # Effect size: Eta-squared H
            k_groups = len(cols)
            total_n = sum(sizes.values())
            eta_sq_h = (h_stat - k_groups + 1) / (total_n - k_groups) if total_n > k_groups else 0.0
            eta_sq_h = max(0.0, eta_sq_h)

            result_stats = {
                "test_name": "Kruskal-Wallis H-Test",
                "statistic": float(h_stat),
                "p_value": float(p_val),
                "medians": medians,
                "q25": q25,
                "q75": q75,
                "iqr": iqr,
                "sizes": sizes,
                "effect_size": float(eta_sq_h),
                "effect_size_name": "Eta-squared (H)",
                "significant": bool(p_val < 0.05)
            }

            # Plot Boxplot
            plot_df = pd.melt(df[cols].dropna(), value_vars=cols, var_name="Variable", value_name="Value")
            sns.boxplot(data=plot_df, x="Variable", y="Value", palette=palette_colors(None, plot_df["Variable"].nunique()), ax=ax, width=0.45)
            sns.stripplot(data=plot_df, x="Variable", y="Value", color="#1E293B", size=4, jitter=0.1, alpha=0.5, ax=ax)
            ax.set_title(f"Kruskal-Wallis Comparison: {', '.join(cols)}", fontsize=12, fontweight="bold", pad=15)
            ax.set_xlabel("Variable", fontsize=10)
            ax.set_ylabel("Value", fontsize=10)

        elif actual_test_type == "wilcoxon":
            if len(cols) != 2:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Wilcoxon Signed-Rank test requires exactly 2 columns."
                )
            c1, c2 = cols[0], cols[1]
            paired_df = df[[c1, c2]].dropna()
            if len(paired_df) < 2:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Not enough paired observations after dropping missing values."
                )

            v1_p = paired_df[c1]
            v2_p = paired_df[c2]
            diff = v1_p - v2_p

            if diff.abs().sum() == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Wilcoxon test requires at least one non-zero difference between variables."
                )

            # Normality check ONLY on differences array
            if len(diff) >= 3:
                w_stat, w_p = stats.shapiro(diff)
                normality_results["differences"] = {"stat": float(w_stat), "p_value": float(w_p), "normal": bool(w_p > 0.05)}
            else:
                normality_results["differences"] = {"error": "Sample size too small for normality test (n < 3)"}

            stat, p_val = stats.wilcoxon(v1_p, v2_p, alternative='two-sided')

            # Calculate descriptives
            median1 = float(v1_p.median())
            q25_1 = float(np.percentile(v1_p, 25))
            q75_1 = float(np.percentile(v1_p, 75))
            iqr_1 = q75_1 - q25_1

            median2 = float(v2_p.median())
            q25_2 = float(np.percentile(v2_p, 25))
            q75_2 = float(np.percentile(v2_p, 75))
            iqr_2 = q75_2 - q25_2

            median_diff = float(diff.median())
            q25_diff = float(np.percentile(diff, 25))
            q75_diff = float(np.percentile(diff, 75))
            iqr_diff = q75_diff - q25_diff

            # Matched-pairs rank-biserial correlation; zero differences are excluded, as in the test
            n_pairs = len(diff)
            paired_effect = rank_biserial_paired(diff.values)
            effect_size = paired_effect["rank_biserial"]

            result_stats = {
                "test_name": "Wilcoxon Signed-Rank Test",
                "statistic": float(stat),
                "p_value": float(p_val),
                "median_group1": median1,
                "q25_group1": q25_1,
                "q75_group1": q75_1,
                "iqr_group1": iqr_1,
                "median_group2": median2,
                "q25_group2": q25_2,
                "q75_group2": q75_2,
                "iqr_group2": iqr_2,
                "median_difference": median_diff,
                "q25_difference": q25_diff,
                "q75_difference": q75_diff,
                "iqr_difference": iqr_diff,
                "effect_size": effect_size,
                "effect_size_name": "Rank-biserial correlation (matched pairs)",
                "n": n_pairs,
                "n_nonzero": paired_effect["n_nonzero"],
                "w_plus": paired_effect["w_plus"],
                "w_minus": paired_effect["w_minus"],
                "significant": bool(p_val < 0.05)
            }

            # Plot differences boxplot
            sns.boxplot(y=diff, color="#F97316", width=0.3, ax=ax)
            sns.stripplot(y=diff, color="#1E293B", size=4, jitter=0.05, alpha=0.6, ax=ax)
            ax.axhline(0, color="#64748B", linestyle="dashed", linewidth=1.5, label="No Difference (0)")
            ax.set_title(f"Wilcoxon Paired Differences Boxplot ({c1} - {c2})", fontsize=12, fontweight="bold", pad=15)
            ax.set_ylabel("Difference Value", fontsize=10)
            ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0")

        elif actual_test_type == "friedman":
            if len(cols) < 3:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Friedman Test requires 3 or more paired columns."
                )

            paired_df = df[cols].dropna()
            if len(paired_df) < 2:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Not enough paired observations after dropping missing values."
                )

            n_samples = len(paired_df)
            k_groups = len(cols)

            # Shapiro-Wilk for repeated measures reference
            for c in cols:
                vals = paired_df[c]
                if len(vals) >= 3:
                    w_stat, w_p = stats.shapiro(vals)
                    normality_results[c] = {"stat": float(w_stat), "p_value": float(w_p), "normal": bool(w_p > 0.05)}
                else:
                    normality_results[c] = {"error": "Sample size too small for normality test (n < 3)"}

            stat, p_val = stats.friedmanchisquare(*[paired_df[c] for c in cols])

            # Kendall's W = chi2 / (N * (k - 1))
            kendalls_w = stat / (n_samples * (k_groups - 1)) if n_samples * (k_groups - 1) > 0 else 0.0

            # Calculate medians, Q1, Q3, IQR
            medians = {c: float(paired_df[c].median()) for c in cols}
            q25 = {c: float(np.percentile(paired_df[c], 25)) for c in cols}
            q75 = {c: float(np.percentile(paired_df[c], 75)) for c in cols}
            iqr = {c: (q75[c] - q25[c]) for c in cols}
            sizes = {c: len(paired_df) for c in cols}

            result_stats = {
                "test_name": "Friedman Test (Repeated Measures)",
                "statistic": float(stat),
                "p_value": float(p_val),
                "medians": medians,
                "q25": q25,
                "q75": q75,
                "iqr": iqr,
                "sizes": sizes,
                "effect_size": float(kendalls_w),
                "effect_size_name": "Kendall's W",
                "n": n_samples,
                "k": k_groups,
                "significant": bool(p_val < 0.05)
            }

            # Plot Repeated Measures Boxplot
            plot_df = pd.melt(paired_df, value_vars=cols, var_name="Variable", value_name="Value")
            sns.boxplot(data=plot_df, x="Variable", y="Value", palette=palette_colors(None, plot_df["Variable"].nunique()), ax=ax, width=0.45)
            sns.stripplot(data=plot_df, x="Variable", y="Value", color="#1E293B", size=4, jitter=0.1, alpha=0.5, ax=ax)
            ax.set_title(f"Friedman Repeated Measures: {', '.join(cols)}", fontsize=12, fontweight="bold", pad=15)
            ax.set_xlabel("Variable", fontsize=10)
            ax.set_ylabel("Value", fontsize=10)

        elif actual_test_type == "chi_square":
            if len(cols) != 2:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Chi-Square test of independence requires exactly 2 categorical columns."
                )
            c1, c2 = cols[0], cols[1]
            cleaned_df = df[[c1, c2]].dropna()
            if len(cleaned_df) < 5:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Not enough observations after dropping missing values."
                )

            contingency = pd.crosstab(cleaned_df[c1], cleaned_df[c2])
            if min(contingency.shape) < 2:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="The chi-square test needs at least 2 categories in each variable."
                )
            chi = chi_square_independence(contingency.values)
            chi_notes = []
            if chi["pct_expected_below_5"] > 20:
                chi_notes.append(
                    f"{chi['pct_expected_below_5']:.0f}% of expected counts are below 5 (smallest {chi['min_expected']:.2f}), "
                    "so the chi-square approximation may be unreliable."
                    + (" Report Fisher's exact test for this 2x2 table." if "fisher_exact_p" in chi else " Consider merging sparse categories.")
                )

            contingency_data = {
                "index": [str(x) for x in contingency.index.tolist()],
                "columns": [str(x) for x in contingency.columns.tolist()],
                "values": contingency.values.tolist(),
                "expected": np.round(chi["expected"], 4).tolist()
            }

            result_stats = {
                "test_name": "Chi-Square Test of Independence",
                "statistic": chi["statistic"],
                "p_value": chi["p_value"],
                "degrees_of_freedom": chi["degrees_of_freedom"],
                "effect_size": chi["cramers_v"],
                "effect_size_name": "Cramér's V",
                "n": int(contingency.values.sum()),
                "contingency_table": contingency_data,
                "min_expected": chi["min_expected"],
                "pct_expected_below_5": chi["pct_expected_below_5"],
                "yates_corrected": chi.get("yates"),
                "fisher_exact_p": chi.get("fisher_exact_p"),
                "notes": chi_notes,
                "significant": bool(chi["p_value"] < 0.05)
            }
            # Plot Frequency
            sns.countplot(data=cleaned_df, x=c1, hue=c2, palette=palette_colors(None, cleaned_df[c2].nunique()), ax=ax)
            ax.set_title(f"Chi-Square Frequency: {c1} by {c2}", fontsize=12, fontweight="bold", pad=15)
            ax.set_xlabel(c1, fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            
            # Prevent overlapping of x-axis tick labels by rotating and reducing font size
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
            
            ax.legend(title=c2, frameon=True, facecolor="white", edgecolor="#E2E8F0")

        plt.tight_layout()
        plt.savefig(plot_buf, format="png", dpi=150)
        plot_buf.seek(0)
        plt.close(fig)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run non-parametric analysis: {str(e)}"
        )

    img_b64 = base64.b64encode(plot_buf.getvalue()).decode('utf-8')

    return {
        "normality": normality_results,
        "statistics": result_stats,
        "plot": img_b64
    }
