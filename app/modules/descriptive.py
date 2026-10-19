"""Descriptive statistics and distribution diagnostics."""

from typing import Optional

import pandas as pd
import numpy as np
import scipy.stats as stats
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status, Form

from ..auth import get_current_user
from ..models import User
from ..measures import distribution_extras

from .common import read_uploaded_table

router = APIRouter()

@router.post("/descriptive")
def analyze_descriptive(
    file: UploadFile = File(...),
    columns_str: str = Form(...),
    group_var: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user)
):
    df = read_uploaded_table(file)

    # Parse columns
    selected_cols = [c.strip() for c in columns_str.split(",") if c.strip()]
    if not selected_cols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please select at least one variable."
        )

    # Check columns exist
    for col in selected_cols:
        if col not in df.columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Column '{col}' not found in dataset."
            )

    if group_var and group_var not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Grouping column '{group_var}' not found in dataset."
        )

    # Helper function to compute descriptive statistics
    def calculate_stats(s_orig):
        # Data Quality
        total_count = len(s_orig)
        missing_count = int(s_orig.isna().sum())
        missing_pct = float((missing_count / total_count) * 100) if total_count > 0 else 0.0

        # Try to convert values to numeric
        s_numeric = pd.to_numeric(s_orig, errors='coerce')

        # Non-null values
        s_clean = s_numeric.dropna()
        n = len(s_clean)

        if n == 0:
            return {
                "n": 0,
                "missing": {"count": missing_count, "percentage": missing_pct},
                "outliers": [],
                "error": "No non-null observations available."
            }

        # Calculate metrics
        try:
            mean_val = float(s_clean.mean())
            median_val = float(s_clean.median())
            
            # Mode calculation
            mode_res = s_clean.mode()
            # No mode when no value repeats (continuous data)
            mode_val = float(mode_res.iloc[0]) if not mode_res.empty and s_clean.value_counts().iloc[0] > 1 else None

            sd_val = float(s_clean.std(ddof=1)) if n > 1 else 0.0
            se_val = float(s_clean.sem()) if n > 0 else 0.0
            var_val = float(s_clean.var(ddof=1)) if n > 1 else 0.0
            
            min_val = float(s_clean.min())
            max_val = float(s_clean.max())
            range_val = float(max_val - min_val)

            # IQR
            q75, q25 = np.percentile(s_clean, [75, 25])
            iqr_val = float(q75 - q25)

            # Outliers
            lower_bound = q25 - 1.5 * iqr_val
            upper_bound = q75 + 1.5 * iqr_val
            outliers_mask = (s_clean < lower_bound) | (s_clean > upper_bound)
            outlier_indices = s_clean[outliers_mask].index.tolist()
            outlier_values = s_clean[outliers_mask].tolist()
            outliers_list = [{"row": int(idx + 2), "val": float(val)} for idx, val in zip(outlier_indices, outlier_values)]

            # Skewness, Kurtosis
            skew_val = float(s_clean.skew()) if n > 2 else 0.0
            kurt_val = float(s_clean.kurtosis()) if n > 3 else 0.0

            # C.V. %
            # C.V. is undefined when the mean is (numerically) zero
            cv_val = float((sd_val / mean_val) * 100) if abs(mean_val) > 1e-9 * float(s_clean.abs().max()) else None

            # Shapiro-Wilk
            shapiro_res = None
            if n >= 3:
                w_stat, w_p = stats.shapiro(s_clean)
                shapiro_res = {
                    "stat": float(w_stat),
                    "p_value": float(w_p),
                    "normal": bool(w_p >= 0.05)
                }
            else:
                shapiro_res = {
                    "error": "Sample size too small for normality test (n < 3)"
                }

            # Q-Q coordinates
            qq_res = None
            if n >= 1:
                sorted_data = np.sort(s_clean)
                positions = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
                theoretical_quantiles = stats.norm.ppf(positions)
                
                # Fit line
                if n > 1:
                    slope, intercept, r_val, p_val, std_err = stats.linregress(theoretical_quantiles, sorted_data)
                else:
                    slope, intercept = 1.0, 0.0

                qq_res = {
                    "theoretical": theoretical_quantiles.tolist(),
                    "ordered": sorted_data.tolist(),
                    "slope": float(slope),
                    "intercept": float(intercept)
                }

            return {
                "n": n,
                "mean": mean_val,
                "median": median_val,
                "mode": mode_val,
                "sd": sd_val,
                "se": se_val,
                "var": var_val,
                "min": min_val,
                "max": max_val,
                "range": range_val,
                "iqr": iqr_val,
                "skewness": skew_val,
                "kurtosis": kurt_val,
                "cv": cv_val,
                **distribution_extras(s_clean.values),
                "shapiro": shapiro_res,
                "qq": qq_res,
                "missing": {"count": missing_count, "percentage": missing_pct},
                "outliers": outliers_list,
                "raw_data": s_clean.tolist()
            }
        except Exception as err:
            return {
                "error": f"Failed to compute statistics: {str(err)}"
            }

    # Populate results
    variables_payload = {}
    for col in selected_cols:
        overall_stats = calculate_stats(df[col])
        
        group_stats = {}
        if group_var:
            grouped = df.groupby(group_var)
            for g_name, g_df in grouped:
                group_stats[str(g_name)] = calculate_stats(g_df[col])

        variables_payload[col] = {
            "overall": overall_stats,
            "groups": group_stats
        }

    return {
        "group_var": group_var,
        "variables": variables_payload
    }
