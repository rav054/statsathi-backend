"""Parametric tests: t-tests and z-tests."""

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
from ..measures import two_sample_t

from .common import read_uploaded_table

router = APIRouter()

@router.post("/parametric")
def analyze_parametric(
    file: UploadFile = File(...),
    test_type: str = Form(...),  # "independent_t", "paired_t", "one_sample_t", "z_test"
    col1: str = Form(...),
    col2: Optional[str] = Form(None),
    mu: Optional[float] = Form(0.0),
    hypothesized_mean: Optional[float] = Form(None),
    current_user: User = Depends(get_current_user)
):
    df = read_uploaded_table(file)

    # Use hypothesized_mean as popmean if passed, else fallback to mu
    h_mean = hypothesized_mean if hypothesized_mean is not None else (mu if mu is not None else 0.0)

    # Validate col1
    if col1 not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Column '{col1}' not found in dataset."
        )

    v1 = df[col1].dropna()
    if len(v1) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Column '{col1}' must contain at least 2 data values."
        )

    # Variables for results
    shapiro_results = {}
    levene_results = None
    test_used = ""
    result_stats = {}
    plot_buf = io.BytesIO()

    try:
        plt.clf()
        plt.close('all')
        fig, ax = plt.subplots(figsize=(7, 5))

        if test_type == "independent_t":
            if not col2 or col2 not in df.columns:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Independent t-test requires a second comparison column variable."
                )
            v2 = df[col2].dropna()
            if len(v2) < 2:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Column '{col2}' must contain at least 2 data values."
                )

            # Shapiro-Wilk Normality test on both groups
            for col_name, v_data in [(col1, v1), (col2, v2)]:
                if len(v_data) >= 3:
                    w_stat, w_p = stats.shapiro(v_data)
                    shapiro_results[col_name] = {"stat": float(w_stat), "p_value": float(w_p), "normal": bool(w_p > 0.05)}
                else:
                    shapiro_results[col_name] = {"error": "Sample size too small for normality test (n < 3)"}

            # Levene's test for equality of variances
            lev_stat, lev_p = stats.levene(v1, v2)
            equal_var = bool(lev_p >= 0.05)
            levene_results = {"stat": float(lev_stat), "p_value": float(lev_p), "equal_var": equal_var}

            # Student's or Welch's t-test chosen by Levene's test (the results screen says which);
            # both are returned so either can be reported.
            summary = two_sample_t(v1.values, v2.values)
            chosen = summary["student"] if equal_var else summary["welch"]
            test_used = "Student's t-Test" if equal_var else "Welch's t-Test"
            p_val = chosen["p_value"]

            result_stats = {
                "test_name": test_used,
                "t_statistic": chosen["t_statistic"],
                "p_value": p_val,
                "degrees_of_freedom": chosen["degrees_of_freedom"],
                "mean_group1": float(v1.mean()),
                "mean_group2": float(v2.mean()),
                "std_group1": float(v1.std(ddof=1)),
                "std_group2": float(v2.std(ddof=1)),
                "n_group1": len(v1),
                "n_group2": len(v2),
                "mean_difference": summary["mean_difference"],
                "se_difference": chosen["se_difference"],
                "ci_difference": chosen["ci_difference"],
                "cohens_d": summary["cohens_d"],
                "hedges_g": summary["hedges_g"],
                "student": summary["student"],
                "welch": summary["welch"],
                "significant": bool(p_val is not None and p_val < 0.05)
            }
            # Group Comparison Boxplot
            plot_df = pd.DataFrame({
                "Value": pd.concat([v1, v2]),
                "Variable": [col1] * len(v1) + [col2] * len(v2)
            })
            sns.boxplot(data=plot_df, x="Variable", y="Value", palette=palette_colors(None, 2), ax=ax, width=0.45)
            sns.stripplot(data=plot_df, x="Variable", y="Value", color="#1E293B", size=4, jitter=0.1, alpha=0.5, ax=ax)
            ax.set_title(f"{test_used}: {col1} vs {col2}", fontsize=12, fontweight="bold", pad=15)
            ax.set_xlabel("Variable", fontsize=10)
            ax.set_ylabel("Values", fontsize=10)

        elif test_type == "paired_t":
            if not col2 or col2 not in df.columns:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Paired t-test requires a second comparison column variable."
                )
            paired_df = df[[col1, col2]].dropna()
            if len(paired_df) < 2:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Not enough paired observations after dropping missing values."
                )
            
            v1_p = paired_df[col1]
            v2_p = paired_df[col2]

            # Calculate paired differences
            diff = v1_p - v2_p

            # Normality check ONLY on differences
            if len(diff) >= 3:
                w_stat, w_p = stats.shapiro(diff)
                shapiro_results = {"differences": {"stat": float(w_stat), "p_value": float(w_p), "normal": bool(w_p > 0.05)}}
            else:
                shapiro_results = {"differences": {"error": "Sample size too small for normality test (n < 3)"}}

            t_stat, p_val = stats.ttest_rel(v1_p, v2_p)
            test_used = "Paired t-Test"
            df_val = float(len(diff) - 1)

            mean_diff = float(diff.mean())
            std_diff = float(diff.std(ddof=1))
            se_diff = std_diff / np.sqrt(len(diff))
            
            # Confidence Interval on Mean Difference
            margin_of_error = stats.t.ppf(0.975, int(df_val)) * se_diff
            ci_diff = [mean_diff - margin_of_error, mean_diff + margin_of_error]

            result_stats = {
                "test_name": test_used,
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "degrees_of_freedom": df_val,
                "mean_group1": float(v1_p.mean()),
                "mean_group2": float(v2_p.mean()),
                "std_group1": float(v1_p.std(ddof=1)),
                "std_group2": float(v2_p.std(ddof=1)),
                "mean_difference": mean_diff,
                "std_difference": std_diff,
                "cohens_d": float(mean_diff / std_diff) if std_diff > 0 else None,  # Cohen's d_z
                "ci_difference": ci_diff,
                "n": len(diff),
                "significant": bool(p_val < 0.05)
            }

            # Differences Histogram
            sns.histplot(diff, kde=True, color="#4F46E5", ax=ax, alpha=0.6)
            ax.axvline(mean_diff, color="#F97316", linestyle="solid", linewidth=2.5, label=f"Mean Diff ({mean_diff:.2f})")
            ax.axvline(0, color="#64748B", linestyle="dashed", linewidth=1.5, label="No Diff (0)")
            ax.set_title(f"Distribution of Paired Differences ({col1} - {col2})", fontsize=12, fontweight="bold", pad=15)
            ax.set_xlabel("Differences", fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0")

        elif test_type == "one_sample_t":
            # Normality check on v1
            if len(v1) >= 3:
                w_stat, w_p = stats.shapiro(v1)
                shapiro_results[col1] = {"stat": float(w_stat), "p_value": float(w_p), "normal": bool(w_p > 0.05)}
            else:
                shapiro_results[col1] = {"error": "Sample size too small for normality test (n < 3)"}

            t_stat, p_val = stats.ttest_1samp(v1, popmean=h_mean)
            test_used = "One-Sample t-Test"
            df_val = float(len(v1) - 1)

            mean_v1 = float(v1.mean())
            std_v1 = float(v1.std(ddof=1))
            se = std_v1 / np.sqrt(len(v1))

            # Confidence Interval on Mean
            margin_of_error = stats.t.ppf(0.975, int(df_val)) * se
            ci = [mean_v1 - margin_of_error, mean_v1 + margin_of_error]

            # Cohen's d
            cohens_d = (mean_v1 - h_mean) / std_v1 if std_v1 > 0 else 0.0

            result_stats = {
                "test_name": test_used,
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "degrees_of_freedom": df_val,
                "sample_mean": mean_v1,
                "hypothesized_mean": h_mean,
                "sample_std": std_v1,
                "cohens_d": float(cohens_d),
                "ci": ci,
                "n": len(v1),
                "significant": bool(p_val < 0.05)
            }

            # Distribution Plot
            sns.histplot(v1, kde=True, color="#4F46E5", ax=ax, alpha=0.6)
            ax.axvline(mean_v1, color="#4F46E5", linestyle="solid", linewidth=2.5, label=f"Sample Mean ({mean_v1:.2f})")
            ax.axvline(h_mean, color="#F97316", linestyle="dashed", linewidth=2.5, label=f"Hypothesized Mean ({h_mean:.2f})")
            ax.set_title(f"Distribution of {col1} vs Hypothesized Mean", fontsize=12, fontweight="bold", pad=15)
            ax.set_xlabel(col1, fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0")

        elif test_type == "z_test":
            test_used = "Z-Test"
            
            # Shapiro-Wilk on v1
            if len(v1) >= 3:
                w_stat, w_p = stats.shapiro(v1)
                shapiro_results[col1] = {"stat": float(w_stat), "p_value": float(w_p), "normal": bool(w_p > 0.05)}
            else:
                shapiro_results[col1] = {"error": "Sample size too small for normality test (n < 3)"}

            if col2 and col2 in df.columns:
                v2 = df[col2].dropna()
                if len(v2) < 2:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Column '{col2}' must contain at least 2 data values."
                    )
                
                # Shapiro-Wilk on v2
                if len(v2) >= 3:
                    w_stat2, w_p2 = stats.shapiro(v2)
                    shapiro_results[col2] = {"stat": float(w_stat2), "p_value": float(w_p2), "normal": bool(w_p2 > 0.05)}
                else:
                    shapiro_results[col2] = {"error": "Sample size too small for normality test (n < 3)"}

                mean1 = float(v1.mean())
                std1 = float(v1.std(ddof=1))
                n1 = len(v1)
                mean2 = float(v2.mean())
                std2 = float(v2.std(ddof=1))
                n2 = len(v2)

                # Z-statistic calculation
                pooled_se = np.sqrt((std1**2 / n1) + (std2**2 / n2))
                z_stat = (mean1 - mean2) / pooled_se if pooled_se > 0 else 0.0
                p_val = stats.norm.sf(abs(z_stat)) * 2.0

                # 95% CI of the Difference
                margin_of_error = stats.norm.ppf(0.975) * pooled_se
                ci_diff = [mean1 - mean2 - margin_of_error, mean1 - mean2 + margin_of_error]

                result_stats = {
                    "test_name": "Two-Sample Z-Test",
                    "z_statistic": float(z_stat),
                    "p_value": float(p_val),
                    "mean_group1": mean1,
                    "mean_group2": mean2,
                    "std_group1": std1,
                    "std_group2": std2,
                    "n_group1": n1,
                    "n_group2": n2,
                    "ci_difference": ci_diff,
                    "significant": bool(p_val < 0.05)
                }

                plot_df = pd.DataFrame({
                    "Value": pd.concat([v1, v2]),
                    "Variable": [col1] * n1 + [col2] * n2
                })
                sns.boxplot(data=plot_df, x="Variable", y="Value", palette=palette_colors(None, 2), ax=ax, width=0.45)
                sns.stripplot(data=plot_df, x="Variable", y="Value", color="#1E293B", size=4, jitter=0.1, alpha=0.5, ax=ax)
                ax.set_title(f"Z-Test Comparison: {col1} vs {col2}", fontsize=12, fontweight="bold", pad=15)
            else:
                # One-sample Z-test
                mean1 = float(v1.mean())
                std1 = float(v1.std(ddof=1))
                n1 = len(v1)
                se = std1 / np.sqrt(n1)
                z_stat = (mean1 - h_mean) / se if se > 0 else 0.0
                p_val = stats.norm.sf(abs(z_stat)) * 2.0

                # 95% CI of the Mean Difference
                margin_of_error = stats.norm.ppf(0.975) * se
                ci_diff = [mean1 - margin_of_error, mean1 + margin_of_error]

                result_stats = {
                    "test_name": "One-Sample Z-Test",
                    "z_statistic": float(z_stat),
                    "p_value": float(p_val),
                    "sample_mean": mean1,
                    "hypothesized_mean": h_mean,
                    "sample_std": std1,
                    "n": n1,
                    "ci_difference": ci_diff,
                    "significant": bool(p_val < 0.05)
                }

                sns.histplot(v1, kde=True, color="#4F46E5", ax=ax, alpha=0.6)
                ax.axvline(mean1, color="#4F46E5", linestyle="solid", linewidth=2.5, label=f"Sample Mean ({mean1:.2f})")
                ax.axvline(h_mean, color="#F97316", linestyle="dashed", linewidth=2.5, label=f"Hypothesized Mean ({h_mean:.2f})")
                ax.set_title(f"Distribution vs Hypothesized Mean (Z-Test)", fontsize=12, fontweight="bold", pad=15)
                ax.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0")

        plt.tight_layout()
        plt.savefig(plot_buf, format="png", dpi=150)
        plot_buf.seek(0)
        plt.close(fig)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate parametric comparison chart: {str(e)}"
        )

    img_b64 = base64.b64encode(plot_buf.getvalue()).decode('utf-8')

    return {
        "shapiro_results": shapiro_results,
        "levene_results": levene_results,
        "test_used": test_used,
        "statistics": result_stats,
        "plot": img_b64
    }
