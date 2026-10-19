"""ANOVA for designed experiments (CRD, RBD, factorial, split-plot, Latin square) with post-hoc tests."""

import io
import base64
import itertools
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
from ..figure_style import COLORS, palette_colors, pretty_label
from .mean_chart import mean_separation_png
from ..models import User
from ..posthoc import (
    METHOD_LABELS,
    PosthocError,
    compact_letter_display,
    pairwise_comparisons,
    pooled_pair_error,
    splitplot_pair_error,
    welch_pair_error,
)

from .common import read_uploaded_table

router = APIRouter()

def fit_rss(X, Y):
    beta, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
    if len(residuals) > 0:
        return float(residuals[0])
    else:
        pred = X.dot(beta)
        return float(np.sum((Y - pred)**2))


def create_dummy_vars(df, column):
    """Sum-to-zero (effect) coding: level j -> +1, first level -> -1, other levels -> 0.

    Dropping an effect's columns from the full model then yields Type III sums of
    squares (equal to Type I/II for balanced data). Plain 0/1 dummy coding does not,
    and gave wrong main-effect SS whenever the model contained interactions.
    """
    categories = sorted(df[column].dropna().unique().tolist())
    if len(categories) < 2:
        raise ValueError(f"Factor '{column}' must have at least 2 categories.")

    reference = (df[column] == categories[0]).astype(float).values
    dummy_cols = []
    for level in categories[1:]:
        dummy_cols.append((df[column] == level).astype(float).values - reference)

    return dummy_cols, categories


def _missing_combinations(df, factors, counts):
    levels = [sorted(df[f].unique().tolist(), key=str) for f in factors]
    present = set(counts.index) if len(factors) > 1 else {(key,) for key in counts.index}
    return [combo for combo in itertools.product(*levels) if combo not in present]


def require_complete_design(df, factors, design_name):
    """Block designs: every combination of `factors` must occur exactly once."""
    counts = df.groupby(factors).size()
    repeated = counts[counts > 1]
    if len(repeated) > 0:
        example = "; ".join(" / ".join(map(str, k)) if isinstance(k, tuple) else str(k) for k in repeated.index[:3])
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{design_name} requires exactly 1 observation per {' x '.join(factors)} combination. Repeated: {example}."
        )
    missing = _missing_combinations(df, factors, counts)
    if missing:
        shown = "; ".join(" / ".join(map(str, m)) for m in missing[:5])
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"{design_name} requires every {' x '.join(factors)} combination, but {len(missing)} "
                f"{'is' if len(missing) == 1 else 'are'} missing (e.g. {shown}). Missing-plot estimation is not "
                "supported yet: fill in the missing values or remove the incomplete block."
            )
        )


def require_all_cells(df, factors, design_name):
    """Factorial CRD: every factor-level combination needs at least one observation."""
    missing = _missing_combinations(df, factors, df.groupby(factors).size())
    if missing:
        shown = "; ".join(" / ".join(map(str, m)) for m in missing[:5])
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{design_name} needs at least one observation for every combination of {', '.join(factors)}. Empty: {shown}."
        )


def build_twoway_design_matrix(df, factorA, factorB):
    dummiesA, catsA = create_dummy_vars(df, factorA)
    dummiesB, catsB = create_dummy_vars(df, factorB)
    
    n_samples = len(df)
    cols = [np.ones(n_samples)]
    cols.extend(dummiesA)
    cols.extend(dummiesB)
    
    for dA in dummiesA:
        for dB in dummiesB:
            cols.append(dA * dB)
            
    X = np.column_stack(cols)
    
    idx_intercept = [0]
    idx_A = list(range(1, 1 + len(dummiesA)))
    idx_B = list(range(1 + len(dummiesA), 1 + len(dummiesA) + len(dummiesB)))
    idx_AB = list(range(1 + len(dummiesA) + len(dummiesB), len(cols)))
    
    return X, idx_intercept, idx_A, idx_B, idx_AB


def build_multifactor_design_matrix(df, factors):
    n_samples = len(df)
    cols = [np.ones(n_samples)]
    
    dummies_dict = {}
    cats_dict = {}
    for f in factors:
        dummies, cats = create_dummy_vars(df, f)
        dummies_dict[f] = dummies
        cats_dict[f] = cats

    effects_indices = {}
    current_idx = 1
    
    for r in range(1, len(factors) + 1):
        for combo in itertools.combinations(factors, r):
            effect_name = ":".join(combo)
            prod_list = [np.ones(n_samples)]
            for f in combo:
                next_prod_list = []
                for p in prod_list:
                    for d in dummies_dict[f]:
                        next_prod_list.append(p * d)
                prod_list = next_prod_list
            
            start_idx = current_idx
            cols.extend(prod_list)
            current_idx += len(prod_list)
            effects_indices[effect_name] = list(range(start_idx, current_idx))
            
    X = np.column_stack(cols)
    return X, effects_indices


def build_multifactor_rbd_design_matrix(df, factors, rep_var):
    n_samples = len(df)
    cols = [np.ones(n_samples)]
    
    # 1. Add dummy variables for replication/block factor
    rep_dummies, rep_cats = create_dummy_vars(df, rep_var)
    cols.extend(rep_dummies)
    
    effects_indices = {}
    current_idx = 1
    
    # Record indices for replication/block factor
    start_idx = current_idx
    current_idx += len(rep_dummies)
    effects_indices[rep_var] = list(range(start_idx, current_idx))
    
    # 2. Add dummy variables for each factor and their interactions
    dummies_dict = {}
    cats_dict = {}
    for f in factors:
        dummies, cats = create_dummy_vars(df, f)
        dummies_dict[f] = dummies
        cats_dict[f] = cats

    for r in range(1, len(factors) + 1):
        for combo in itertools.combinations(factors, r):
            effect_name = ":".join(combo)
            prod_list = [np.ones(n_samples)]
            for f in combo:
                next_prod_list = []
                for p in prod_list:
                    for d in dummies_dict[f]:
                        next_prod_list.append(p * d)
                prod_list = next_prod_list
            
            start_idx = current_idx
            cols.extend(prod_list)
            current_idx += len(prod_list)
            effects_indices[effect_name] = list(range(start_idx, current_idx))
            
    X = np.column_stack(cols)
    return X, effects_indices


@router.post("/anova")
def analyze_anova(
    file: UploadFile = File(...),
    test_type: str = Form(...),  # "oneway", "rbd_oneway", "twoway", "rbd_twoway", "splitplot", "crd_multifactor", "rbd_multifactor"
    dep_var: str = Form(...),    # numeric DV
    ind_var1: Optional[str] = Form(None),   # categorical IV 1
    ind_var2: Optional[str] = Form(None),  # categorical IV 2
    rep_var: Optional[str] = Form(None),   # categorical replication/block factor
    factors: Optional[str] = Form(None),   # Comma-separated list of factors for multi-factor ANOVA
    posthoc_method: Optional[str] = Form("tukey"),  # "tukey" or "games_howell"
    palette: Optional[str] = Form(None),  # a key from figure_style.json; None uses the default palette
    current_user: User = Depends(get_current_user)
):
    # Sanitize default Form parameters if function is called directly in Python
    if type(ind_var1).__name__ == "Form": ind_var1 = None
    if type(ind_var2).__name__ == "Form": ind_var2 = None
    if type(rep_var).__name__ == "Form": rep_var = None
    if type(factors).__name__ == "Form": factors = None
    if type(posthoc_method).__name__ == "Form": posthoc_method = "tukey"
    if type(palette).__name__ == "Form": palette = None

    posthoc_method = (posthoc_method or "tukey").lower()
    if posthoc_method not in METHOD_LABELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown post-hoc method '{posthoc_method}'. Choose one of: {', '.join(METHOD_LABELS)}."
        )
    if posthoc_method == "games_howell" and test_type != "oneway":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Games-Howell is only available for One-Way CRD. Designs with blocks or several factors use a pooled error term: choose Tukey, LSD, Duncan or Bonferroni."
        )

    df = read_uploaded_table(file, str_columns=True)

    if dep_var not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Column '{dep_var}' not found in dataset."
        )

    factor_list = []
    if test_type in ["crd_multifactor", "rbd_multifactor"]:
        if not factors:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The 'factors' parameter is required for multi-factor ANOVA."
            )
        factor_list = [f.strip() for f in factors.split(",") if f.strip()]
        # Immediately convert factor columns to string/category type
        for col in factor_list:
            if col in df.columns:
                df[col] = df[col].astype(str)
        if len(factor_list) < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one factor must be selected for multi-factor ANOVA."
            )
        for f in factor_list:
            if f not in df.columns:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Column '{f}' not found in dataset."
                )
    else:
        if not ind_var1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Independent Factor 1 (ind_var1) is required."
            )
        if ind_var1 not in df.columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Column '{ind_var1}' not found in dataset."
            )

    if ind_var2 and ind_var2 not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Column '{ind_var2}' not found in dataset."
        )

    if rep_var and rep_var not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Column '{rep_var}' not found in dataset."
        )

    # Convert dependent variable to numeric first, coercing invalid text to NaN
    try:
        df[dep_var] = pd.to_numeric(df[dep_var], errors='coerce')
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Dependent variable '{dep_var}' must be numeric."
        )

    cols_to_use = [dep_var]
    if test_type in ["crd_multifactor", "rbd_multifactor"]:
        cols_to_use.extend(factor_list)
    else:
        cols_to_use.append(ind_var1)
        if ind_var2:
            cols_to_use.append(ind_var2)
    if rep_var:
        cols_to_use.append(rep_var)
        
    df_clean = df[cols_to_use].dropna().copy()
    
    if len(df_clean) < 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset contains empty cells or invalid dimensions. ANOVA requires at least 5 valid observations after removing missing values or non-numeric cells."
        )

    # Check for zero variance in ANOVA dependent variable
    if df_clean[dep_var].var() == 0 or np.isnan(df_clean[dep_var].var()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The dependent variable has zero variance (all values are identical). ANOVA cannot be performed."
        )

    shapiro_results = {}
    levene_results = None
    anova_table = {}
    posthoc_results = []
    cd_results = []
    notes = []
    plot_buf = io.BytesIO()

    try:
        plt.clf()
        plt.close('all')
        fig, ax = plt.subplots(figsize=(7, 5))

        if test_type == "oneway":
            groups = df_clean.groupby(ind_var1)
            group_names = sorted(list(groups.groups.keys()))
            group_lists = [group[dep_var].values for name, group in groups]
            
            if len(group_lists) < 2:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Categorical variable '{ind_var1}' must have at least 2 categories."
                )

            for name, g_data in zip(group_names, group_lists):
                if len(g_data) >= 3:
                    w_stat, w_p = stats.shapiro(g_data)
                    shapiro_results[str(name)] = {"stat": float(w_stat), "p_value": float(w_p), "normal": bool(w_p > 0.05)}
                else:
                    shapiro_results[str(name)] = {"error": "Sample size too small for normality test (n < 3)"}

            if len(group_lists) >= 2:
                lev_stat, lev_p = stats.levene(*group_lists)
                equal_var = bool(lev_p >= 0.05)
                levene_results = {"stat": float(lev_stat), "p_value": float(lev_p), "equal_var": equal_var}
            else:
                equal_var = True

            # The classic (Fisher) ANOVA is always reported: CD, DMRT and letters are built on it.
            f_stat, p_val = stats.f_oneway(*group_lists)
            df_between = len(group_lists) - 1
            df_within = len(df_clean) - len(group_lists)

            grand_mean = df_clean[dep_var].mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in group_lists)
            ss_within = sum(((g - g.mean())**2).sum() for g in group_lists)
            ss_total = ss_between + ss_within

            ms_between = ss_between / df_between if df_between > 0 else 0.0
            ms_within = ss_within / df_within if df_within > 0 else 0.0

            anova_table = {
                "method": "Completely Randomized Design (CRD) - One Factor ANOVA",
                "equal_var": equal_var,
                "df_between": int(df_between),
                "ss_between": float(ss_between),
                "ms_between": float(ms_between),
                "f_statistic": float(f_stat),
                "p_value": float(p_val),
                "df_within": int(df_within),
                "ss_within": float(ss_within),
                "ms_within": float(ms_within),
                "ss_total": float(ss_total),
                "df_total": int(df_between + df_within),
                "significant": bool(p_val < 0.05)
            }

            group_sizes = [len(g) for g in group_lists]
            r = len(df_clean) / len(group_names)
            if len(set(group_sizes)) > 1:
                notes.append(
                    "Groups have unequal replication. The CD row uses the average replication (N/k); "
                    "post-hoc tests and letters use the exact standard error for each pair."
                )
            if ms_within > 0 and r > 0 and df_within > 0:
                se_d = np.sqrt(2.0 * ms_within / r)
                t_5 = stats.t.ppf(0.975, df_within)
                t_1 = stats.t.ppf(0.995, df_within)
                cd_results.append({
                    "parameter": ind_var1,
                    "se_d": float(se_d),
                    "cd_5": float(t_5 * se_d),
                    "cd_1": float(t_1 * se_d)
                })

            if not equal_var:
                # Welch's ANOVA is reported alongside the classic table, never silently substituted
                k = len(group_lists)
                ns = np.array(group_sizes, dtype=float)
                means = np.array([g.mean() for g in group_lists])
                vars_ = np.array([g.var(ddof=1) if len(g) > 1 else 0.0 for g in group_lists])
                if np.all(ns > 1) and np.all(vars_ > 0):
                    ws = ns / vars_
                    sum_ws = ws.sum()
                    w_mean = (ws * means).sum() / sum_ws
                    a_term = sum((1.0 / (ns - 1)) * (1.0 - ws / sum_ws)**2)
                    if a_term > 0:
                        f_welch = (sum(ws * (means - w_mean)**2) / (k - 1)) / (1.0 + (2.0 * (k - 2) / (k**2 - 1)) * a_term)
                        df_den = (k**2 - 1) / (3.0 * a_term)
                        p_welch = stats.f.sf(f_welch, k - 1, df_den)
                        anova_table["welch_anova"] = {
                            "f_statistic": float(f_welch),
                            "df_between": int(k - 1),
                            "df_within": float(df_den),
                            "p_value": float(p_welch),
                            "significant": bool(p_welch < 0.05)
                        }
                        notes.append(
                            f"Levene's test indicates unequal variances (p = {levene_results['p_value']:.4f}). "
                            f"Welch's ANOVA: F({k - 1}, {df_den:.2f}) = {f_welch:.3f}, p = {p_welch:.4g}."
                            + ("" if posthoc_method == "games_howell" else " Games-Howell is the safer post-hoc choice here.")
                        )
            sns.boxplot(data=df_clean, x=ind_var1, y=dep_var, color=palette_colors(palette)[0], ax=ax, width=0.45, linewidth=0.9, fliersize=3)
            sns.stripplot(data=df_clean, x=ind_var1, y=dep_var, color=COLORS["text"], size=4, jitter=0.1, alpha=0.45, ax=ax)
            ax.set_title(f"One-Way ANOVA: {dep_var} by {ind_var1}", fontsize=12, fontweight="bold", pad=15)
            ax.set_xlabel(ind_var1, fontsize=10)
            ax.set_ylabel(dep_var, fontsize=10)

        elif test_type == "rbd_oneway":
            if not rep_var:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Replication/Block variable (rep_var) is required for One-Factor RBD."
                )

            require_complete_design(df_clean, [ind_var1, rep_var], "Randomized Block Design (RBD)")

            treats = sorted(df_clean[ind_var1].unique().tolist())
            reps = sorted(df_clean[rep_var].unique().tolist())
            t = len(treats)
            r = len(reps)
            N = len(df_clean)

            for treat in treats:
                g_data = df_clean[df_clean[ind_var1] == treat][dep_var].values
                if len(g_data) >= 3:
                    w_stat, w_p = stats.shapiro(g_data)
                    shapiro_results[str(treat)] = {"stat": float(w_stat), "p_value": float(w_p), "normal": bool(w_p > 0.05)}
                else:
                    shapiro_results[str(treat)] = {"error": "Sample size too small for normality test (n < 3)"}

            group_lists = [df_clean[df_clean[ind_var1] == treat][dep_var].values for treat in treats]
            lev_stat, lev_p = stats.levene(*group_lists)
            levene_results = {"stat": float(lev_stat), "p_value": float(lev_p), "equal_var": bool(lev_p >= 0.05)}

            grand_mean = df_clean[dep_var].mean()
            ss_total = ((df_clean[dep_var] - grand_mean)**2).sum()
            
            rep_means = df_clean.groupby(rep_var)[dep_var].mean()
            ss_rep = t * sum((mean - grand_mean)**2 for mean in rep_means)
            
            treat_means = df_clean.groupby(ind_var1)[dep_var].mean()
            ss_treat = r * sum((mean - grand_mean)**2 for mean in treat_means)
            
            ss_error = max(0.0, ss_total - ss_rep - ss_treat)

            df_rep = r - 1
            df_treat = t - 1
            df_error = (r - 1) * (t - 1)
            df_total = N - 1

            ms_rep = ss_rep / df_rep if df_rep > 0 else 0.0
            ms_treat = ss_treat / df_treat if df_treat > 0 else 0.0
            ms_error = ss_error / df_error if df_error > 0 else 0.0

            f_treat = ms_treat / ms_error if ms_error > 0 else 0.0
            f_rep = ms_rep / ms_error if ms_error > 0 else 0.0

            p_treat = stats.f.sf(f_treat, df_treat, df_error)
            p_rep = stats.f.sf(f_rep, df_rep, df_error)

            anova_table = {
                "method": "One-Way Randomized Block Design (RBD) ANOVA",
                "df_rep": int(df_rep),
                "ss_rep": float(ss_rep),
                "ms_rep": float(ms_rep),
                "f_rep": float(f_rep),
                "p_rep": float(p_rep),
                "significant_rep": bool(p_rep < 0.05),

                "df_between": int(df_treat),  # treatments
                "ss_between": float(ss_treat),
                "ms_between": float(ms_treat),
                "f_statistic": float(f_treat),
                "p_value": float(p_treat),
                "significant": bool(p_treat < 0.05),

                "df_within": int(df_error),  # error
                "ss_within": float(ss_error),
                "ms_within": float(ms_error),

                "ss_total": float(ss_total),
                "df_total": int(df_total)
            }

            if ms_error > 0 and r > 0 and df_error > 0:
                se_d = np.sqrt(2.0 * ms_error / r)
                t_5 = stats.t.ppf(0.975, df_error)
                t_1 = stats.t.ppf(0.995, df_error)
                cd_results.append({
                    "parameter": ind_var1,
                    "se_d": float(se_d),
                    "cd_5": float(t_5 * se_d),
                    "cd_1": float(t_1 * se_d)
                })

            sns.boxplot(data=df_clean, x=ind_var1, y=dep_var, color=palette_colors(palette)[0], ax=ax, width=0.45, linewidth=0.9, fliersize=3)
            sns.stripplot(data=df_clean, x=ind_var1, y=dep_var, color=COLORS["text"], size=4, jitter=0.1, alpha=0.45, ax=ax)
            ax.set_title(f"One-Factor RBD: {dep_var} by {ind_var1}", fontsize=12, fontweight="bold", pad=15)
            ax.set_xlabel(ind_var1, fontsize=10)
            ax.set_ylabel(dep_var, fontsize=10)

        elif test_type == "lsd":
            if not ind_var2 or not rep_var:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Latin Square Design (LSD) requires Columns (ind_var2) and Rows (rep_var) factors."
                )

            t_levels = sorted(df_clean[ind_var1].unique().tolist())
            c_levels = sorted(df_clean[ind_var2].unique().tolist())
            r_levels = sorted(df_clean[rep_var].unique().tolist())
            
            t = len(t_levels)
            c = len(c_levels)
            r = len(r_levels)
            N = len(df_clean)

            if t != c or t != r:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Latin Square Design (LSD) requires the number of rows, columns, and treatments to be equal. Found: rows={r}, columns={c}, treatments={t}."
                )

            require_complete_design(df_clean, [rep_var, ind_var2], "Latin Square Design (LSD)")
            for line_factor, line_name in [(rep_var, "row"), (ind_var2, "column")]:
                per_line = df_clean.groupby([line_factor, ind_var1]).size()
                if len(per_line) != t * t or not all(per_line == 1):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"This is not a Latin square: every treatment must appear exactly once in each {line_name} ('{line_factor}')."
                    )

            for treat in t_levels:
                g_data = df_clean[df_clean[ind_var1] == treat][dep_var].values
                if len(g_data) >= 3:
                    w_stat, w_p = stats.shapiro(g_data)
                    shapiro_results[str(treat)] = {"stat": float(w_stat), "p_value": float(w_p), "normal": bool(w_p > 0.05)}
                else:
                    shapiro_results[str(treat)] = {"error": "Sample size too small for normality test (n < 3)"}

            group_lists = [df_clean[df_clean[ind_var1] == treat][dep_var].values for treat in t_levels]
            lev_stat, lev_p = stats.levene(*group_lists)
            levene_results = {"stat": float(lev_stat), "p_value": float(lev_p), "equal_var": bool(lev_p >= 0.05)}

            grand_mean = df_clean[dep_var].mean()
            ss_total = ((df_clean[dep_var] - grand_mean)**2).sum()
            
            row_means = df_clean.groupby(rep_var)[dep_var].mean()
            ss_row = t * sum((mean - grand_mean)**2 for mean in row_means)
            
            col_means = df_clean.groupby(ind_var2)[dep_var].mean()
            ss_col = t * sum((mean - grand_mean)**2 for mean in col_means)
            
            treat_means = df_clean.groupby(ind_var1)[dep_var].mean()
            ss_treat = r * sum((mean - grand_mean)**2 for mean in treat_means)
            
            ss_error = max(0.0, ss_total - ss_row - ss_col - ss_treat)

            df_row = t - 1
            df_col = t - 1
            df_treat = t - 1
            df_error = (t - 1) * (t - 2)
            df_total = N - 1

            ms_row = ss_row / df_row if df_row > 0 else 0.0
            ms_col = ss_col / df_col if df_col > 0 else 0.0
            ms_treat = ss_treat / df_treat if df_treat > 0 else 0.0
            ms_error = ss_error / df_error if df_error > 0 else 0.0

            f_treat = ms_treat / ms_error if ms_error > 0 else 0.0
            f_row = ms_row / ms_error if ms_error > 0 else 0.0
            f_col = ms_col / ms_error if ms_error > 0 else 0.0

            p_treat = stats.f.sf(f_treat, df_treat, df_error) if df_error > 0 else 1.0
            p_row = stats.f.sf(f_row, df_row, df_error) if df_error > 0 else 1.0
            p_col = stats.f.sf(f_col, df_col, df_error) if df_error > 0 else 1.0

            anova_table = {
                "method": "Latin Square Design (LSD) ANOVA",
                "df_row": int(df_row),
                "ss_row": float(ss_row),
                "ms_row": float(ms_row),
                "f_row": float(f_row),
                "p_row": float(p_row),
                "significant_row": bool(p_row < 0.05),

                "df_col": int(df_col),
                "ss_col": float(ss_col),
                "ms_col": float(ms_col),
                "f_col": float(f_col),
                "p_col": float(p_col),
                "significant_col": bool(p_col < 0.05),

                "df_between": int(df_treat),  # treatments
                "ss_between": float(ss_treat),
                "ms_between": float(ms_treat),
                "f_statistic": float(f_treat),
                "p_value": float(p_treat),
                "significant": bool(p_treat < 0.05),

                "df_within": int(df_error),  # error
                "ss_within": float(ss_error),
                "ms_within": float(ms_error),

                "ss_total": float(ss_total),
                "df_total": int(df_total)
            }

            if ms_error > 0 and t > 0 and df_error > 0:
                se_d = np.sqrt(2.0 * ms_error / t)
                t_5 = stats.t.ppf(0.975, df_error)
                t_1 = stats.t.ppf(0.995, df_error)
                cd_results.append({
                    "parameter": ind_var1,
                    "se_d": float(se_d),
                    "cd_5": float(t_5 * se_d),
                    "cd_1": float(t_1 * se_d)
                })

            sns.boxplot(data=df_clean, x=ind_var1, y=dep_var, color=palette_colors(palette)[0], ax=ax, width=0.45, linewidth=0.9, fliersize=3)
            sns.stripplot(data=df_clean, x=ind_var1, y=dep_var, color=COLORS["text"], size=4, jitter=0.1, alpha=0.45, ax=ax)
            ax.set_title(f"Latin Square Design (LSD): {dep_var} by {ind_var1}", fontsize=12, fontweight="bold", pad=15)
            ax.set_xlabel(ind_var1, fontsize=10)
            ax.set_ylabel(dep_var, fontsize=10)

        elif test_type == "twoway" and ind_var2:
            require_all_cells(df_clean, [ind_var1, ind_var2], "Two-Way ANOVA")
            X, idx_intercept, idx_A, idx_B, idx_AB = build_twoway_design_matrix(df_clean, ind_var1, ind_var2)
            Y = df_clean[dep_var].values
            
            n_samples = len(df_clean)
            n_coefs = X.shape[1]
            
            if n_samples <= n_coefs:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Insufficent sample size ({n_samples}) for Two-Way ANOVA. Needs more rows than combinations."
                )

            rss_full = fit_rss(X, Y)
            rss_no_AB = fit_rss(X[:, idx_intercept + idx_A + idx_B], Y)
            rss_no_A = fit_rss(X[:, idx_intercept + idx_B + idx_AB], Y)
            rss_no_B = fit_rss(X[:, idx_intercept + idx_A + idx_AB], Y)
            
            ss_A = max(0.0, rss_no_A - rss_full)
            ss_B = max(0.0, rss_no_B - rss_full)
            ss_AB = max(0.0, rss_no_AB - rss_full)
            ss_error = max(0.0, rss_full)
            ss_total = ((Y - Y.mean())**2).sum()
            
            df_A = len(idx_A)
            df_B = len(idx_B)
            df_AB = len(idx_AB)
            df_error = n_samples - n_coefs
            
            ms_A = ss_A / df_A if df_A > 0 else 0.0
            ms_B = ss_B / df_B if df_B > 0 else 0.0
            ms_AB = ss_AB / df_AB if df_AB > 0 else 0.0
            ms_error = ss_error / df_error if df_error > 0 else 0.0
            
            f_A = ms_A / ms_error if ms_error > 0 else 0.0
            p_A = stats.f.sf(f_A, df_A, df_error)
            
            f_B = ms_B / ms_error if ms_error > 0 else 0.0
            p_B = stats.f.sf(f_B, df_B, df_error)
            
            f_AB = ms_AB / ms_error if ms_error > 0 else 0.0
            p_AB = stats.f.sf(f_AB, df_AB, df_error)
            
            anova_table = {
                "method": "Two-Way ANOVA with Interaction (Type III SS)",
                "factorA": {
                    "name": ind_var1,
                    "df": int(df_A),
                    "ss": float(ss_A),
                    "ms": float(ms_A),
                    "f_statistic": float(f_A),
                    "p_value": float(p_A),
                    "significant": bool(p_A < 0.05)
                },
                "factorB": {
                    "name": ind_var2,
                    "df": int(df_B),
                    "ss": float(ss_B),
                    "ms": float(ms_B),
                    "f_statistic": float(f_B),
                    "p_value": float(p_B),
                    "significant": bool(p_B < 0.05)
                },
                "interaction": {
                    "name": f"{ind_var1} x {ind_var2}",
                    "df": int(df_AB),
                    "ss": float(ss_AB),
                    "ms": float(ms_AB),
                    "f_statistic": float(f_AB),
                    "p_value": float(p_AB),
                    "significant": bool(p_AB < 0.05)
                },
                "error": {
                    "df": int(df_error),
                    "ss": float(ss_error),
                    "ms": float(ms_error)
                },
                "total": {
                    "df": int(n_samples - 1),
                    "ss": float(ss_total)
                }
            }

            a_levels = len(df_clean[ind_var1].unique())
            b_levels = len(df_clean[ind_var2].unique())
            r = n_samples / (a_levels * b_levels)
            
            if ms_error > 0 and r > 0 and df_error > 0:
                t_5 = stats.t.ppf(0.975, df_error)
                t_1 = stats.t.ppf(0.995, df_error)
                
                se_A = np.sqrt(2.0 * ms_error / (b_levels * r))
                cd_results.append({
                    "parameter": f"Factor A: {ind_var1}",
                    "se_d": float(se_A),
                    "cd_5": float(t_5 * se_A),
                    "cd_1": float(t_1 * se_A)
                })
                se_B = np.sqrt(2.0 * ms_error / (a_levels * r))
                cd_results.append({
                    "parameter": f"Factor B: {ind_var2}",
                    "se_d": float(se_B),
                    "cd_5": float(t_5 * se_B),
                    "cd_1": float(t_1 * se_B)
                })
                se_AB = np.sqrt(2.0 * ms_error / r)
                cd_results.append({
                    "parameter": f"Interaction: {ind_var1} x {ind_var2}",
                    "se_d": float(se_AB),
                    "cd_5": float(t_5 * se_AB),
                    "cd_1": float(t_1 * se_AB)
                })

            sns.pointplot(data=df_clean, x=ind_var1, y=dep_var, hue=ind_var2, palette=palette_colors(palette, df_clean[ind_var2].nunique()), markers=["o", "s", "^", "D", "v"], linestyles=["-", "--", "-.", ":", "-"], capsize=0.1, ax=ax)
            ax.set_title(f"Interaction: {dep_var} by {ind_var1} & {ind_var2}", fontsize=12, fontweight="bold", pad=15)
            ax.set_xlabel(ind_var1, fontsize=10)
            ax.set_ylabel(dep_var, fontsize=10)
            ax.legend(title=ind_var2, frameon=True, facecolor="white", edgecolor="#E2E8F0")

        elif test_type == "rbd_twoway" and ind_var2:
            if not rep_var:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Replication/Block variable (rep_var) is required for Two-Factor RBD."
                )

            require_complete_design(df_clean, [ind_var1, ind_var2, rep_var], "Two-Factor RBD")

            a_vals = sorted(df_clean[ind_var1].unique().tolist())
            b_vals = sorted(df_clean[ind_var2].unique().tolist())
            r_vals = sorted(df_clean[rep_var].unique().tolist())
            a = len(a_vals)
            b = len(b_vals)
            r = len(r_vals)
            N = len(df_clean)

            grand_mean = df_clean[dep_var].mean()
            ss_total = ((df_clean[dep_var] - grand_mean)**2).sum()

            rep_means = df_clean.groupby(rep_var)[dep_var].mean()
            ss_rep = a * b * sum((m - grand_mean)**2 for m in rep_means)

            a_means = df_clean.groupby(ind_var1)[dep_var].mean()
            ss_A = b * r * sum((m - grand_mean)**2 for m in a_means)

            b_means = df_clean.groupby(ind_var2)[dep_var].mean()
            ss_B = a * r * sum((m - grand_mean)**2 for m in b_means)

            ab_means = df_clean.groupby([ind_var1, ind_var2])[dep_var].mean()
            ss_AB = r * sum((ab_means[ai, bi] - a_means[ai] - b_means[bi] + grand_mean)**2 for ai in a_vals for bi in b_vals)

            ss_error = max(0.0, ss_total - ss_rep - ss_A - ss_B - ss_AB)

            df_rep = r - 1
            df_A = a - 1
            df_B = b - 1
            df_AB = (a - 1) * (b - 1)
            df_error = (r - 1) * (a * b - 1)
            df_total = N - 1

            ms_rep = ss_rep / df_rep if df_rep > 0 else 0.0
            ms_A = ss_A / df_A if df_A > 0 else 0.0
            ms_B = ss_B / df_B if df_B > 0 else 0.0
            ms_AB = ss_AB / df_AB if df_AB > 0 else 0.0
            ms_error = ss_error / df_error if df_error > 0 else 0.0

            f_rep = ms_rep / ms_error if ms_error > 0 else 0.0
            f_A = ms_A / ms_error if ms_error > 0 else 0.0
            f_B = ms_B / ms_error if ms_error > 0 else 0.0
            f_AB = ms_AB / ms_error if ms_error > 0 else 0.0

            p_rep = stats.f.sf(f_rep, df_rep, df_error)
            p_A = stats.f.sf(f_A, df_A, df_error)
            p_B = stats.f.sf(f_B, df_B, df_error)
            p_AB = stats.f.sf(f_AB, df_AB, df_error)

            anova_table = {
                "method": "Two-Way Randomized Block Design (RBD) ANOVA",
                "df_rep": int(df_rep),
                "ss_rep": float(ss_rep),
                "ms_rep": float(ms_rep),
                "f_rep": float(f_rep),
                "p_rep": float(p_rep),
                "significant_rep": bool(p_rep < 0.05),

                "factorA": {
                    "name": ind_var1,
                    "df": int(df_A),
                    "ss": float(ss_A),
                    "ms": float(ms_A),
                    "f_statistic": float(f_A),
                    "p_value": float(p_A),
                    "significant": bool(p_A < 0.05)
                },
                "factorB": {
                    "name": ind_var2,
                    "df": int(df_B),
                    "ss": float(ss_B),
                    "ms": float(ms_B),
                    "f_statistic": float(f_B),
                    "p_value": float(p_B),
                    "significant": bool(p_B < 0.05)
                },
                "interaction": {
                    "name": f"{ind_var1} x {ind_var2}",
                    "df": int(df_AB),
                    "ss": float(ss_AB),
                    "ms": float(ms_AB),
                    "f_statistic": float(f_AB),
                    "p_value": float(p_AB),
                    "significant": bool(p_AB < 0.05)
                },
                "error": {
                    "df": int(df_error),
                    "ss": float(ss_error),
                    "ms": float(ms_error)
                },
                "total": {
                    "df": int(df_total),
                    "ss": float(ss_total)
                }
            }

            if ms_error > 0 and r > 0 and df_error > 0:
                t_5 = stats.t.ppf(0.975, df_error)
                t_1 = stats.t.ppf(0.995, df_error)
                
                se_A = np.sqrt(2.0 * ms_error / (b * r))
                cd_results.append({
                    "parameter": f"Factor A: {ind_var1}",
                    "se_d": float(se_A),
                    "cd_5": float(t_5 * se_A),
                    "cd_1": float(t_1 * se_A)
                })
                se_B = np.sqrt(2.0 * ms_error / (a * r))
                cd_results.append({
                    "parameter": f"Factor B: {ind_var2}",
                    "se_d": float(se_B),
                    "cd_5": float(t_5 * se_B),
                    "cd_1": float(t_1 * se_B)
                })
                se_AB = np.sqrt(2.0 * ms_error / r)
                cd_results.append({
                    "parameter": f"Interaction: {ind_var1} x {ind_var2}",
                    "se_d": float(se_AB),
                    "cd_5": float(t_5 * se_AB),
                    "cd_1": float(t_1 * se_AB)
                })

            sns.pointplot(data=df_clean, x=ind_var1, y=dep_var, hue=ind_var2, palette=palette_colors(palette, df_clean[ind_var2].nunique()), markers=["o", "s", "^", "D", "v"], linestyles=["-", "--", "-.", ":", "-"], capsize=0.1, ax=ax)
            ax.set_title(f"Two-Way RBD Interaction: {dep_var} by {ind_var1} & {ind_var2}", fontsize=12, fontweight="bold", pad=15)
            ax.set_xlabel(ind_var1, fontsize=10)
            ax.set_ylabel(dep_var, fontsize=10)
            ax.legend(title=ind_var2, frameon=True, facecolor="white", edgecolor="#E2E8F0")

        elif test_type == "splitplot" and ind_var2:
            if not rep_var:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Replication/Block variable (rep_var) is required for Split-plot Design."
                )

            require_complete_design(df_clean, [ind_var1, ind_var2, rep_var], "Split-plot Design")

            a_vals = sorted(df_clean[ind_var1].unique().tolist())
            b_vals = sorted(df_clean[ind_var2].unique().tolist())
            r_vals = sorted(df_clean[rep_var].unique().tolist())
            a = len(a_vals)
            b = len(b_vals)
            r = len(r_vals)
            N = len(df_clean)

            grand_mean = df_clean[dep_var].mean()
            ss_total = ((df_clean[dep_var] - grand_mean)**2).sum()

            rep_means = df_clean.groupby(rep_var)[dep_var].mean()
            ss_rep = a * b * sum((m - grand_mean)**2 for m in rep_means)

            a_means = df_clean.groupby(ind_var1)[dep_var].mean()
            ss_A = b * r * sum((m - grand_mean)**2 for m in a_means)

            rep_a_means = df_clean.groupby([ind_var1, rep_var])[dep_var].mean()
            ss_error_a = b * sum((rep_a_means[ai, ri] - a_means[ai] - rep_means[ri] + grand_mean)**2 for ai in a_vals for ri in r_vals)

            b_means = df_clean.groupby(ind_var2)[dep_var].mean()
            ss_B = a * r * sum((m - grand_mean)**2 for m in b_means)

            ab_means = df_clean.groupby([ind_var1, ind_var2])[dep_var].mean()
            ss_AB = r * sum((ab_means[ai, bi] - a_means[ai] - b_means[bi] + grand_mean)**2 for ai in a_vals for bi in b_vals)

            ss_error_b = max(0.0, ss_total - ss_rep - ss_A - ss_error_a - ss_B - ss_AB)

            df_rep = r - 1
            df_A = a - 1
            df_error_a = (r - 1) * (a - 1)
            df_B = b - 1
            df_AB = (a - 1) * (b - 1)
            df_error_b = a * (r - 1) * (b - 1)
            df_total = N - 1

            ms_rep = ss_rep / df_rep if df_rep > 0 else 0.0
            ms_A = ss_A / df_A if df_A > 0 else 0.0
            ms_error_a = ss_error_a / df_error_a if df_error_a > 0 else 0.0
            ms_B = ss_B / df_B if df_B > 0 else 0.0
            ms_AB = ss_AB / df_AB if df_AB > 0 else 0.0
            ms_error_b = ss_error_b / df_error_b if df_error_b > 0 else 0.0

            f_rep = ms_rep / ms_error_a if ms_error_a > 0 else 0.0
            f_A = ms_A / ms_error_a if ms_error_a > 0 else 0.0
            f_B = ms_B / ms_error_b if ms_error_b > 0 else 0.0
            f_AB = ms_AB / ms_error_b if ms_error_b > 0 else 0.0

            p_rep = stats.f.sf(f_rep, df_rep, df_error_a)
            p_A = stats.f.sf(f_A, df_A, df_error_a)
            p_B = stats.f.sf(f_B, df_B, df_error_b)
            p_AB = stats.f.sf(f_AB, df_AB, df_error_b)

            anova_table = {
                "method": "Split-plot Design ANOVA",
                "df_rep": int(df_rep),
                "ss_rep": float(ss_rep),
                "ms_rep": float(ms_rep),
                "f_rep": float(f_rep),
                "p_rep": float(p_rep),
                "significant_rep": bool(p_rep < 0.05),

                "factorA": {
                    "name": ind_var1,
                    "df": int(df_A),
                    "ss": float(ss_A),
                    "ms": float(ms_A),
                    "f_statistic": float(f_A),
                    "p_value": float(p_A),
                    "significant": bool(p_A < 0.05)
                },
                "error_a": {
                    "df": int(df_error_a),
                    "ss": float(ss_error_a),
                    "ms": float(ms_error_a)
                },
                "factorB": {
                    "name": ind_var2,
                    "df": int(df_B),
                    "ss": float(ss_B),
                    "ms": float(ms_B),
                    "f_statistic": float(f_B),
                    "p_value": float(p_B),
                    "significant": bool(p_B < 0.05)
                },
                "interaction": {
                    "name": f"{ind_var1} x {ind_var2}",
                    "df": int(df_AB),
                    "ss": float(ss_AB),
                    "ms": float(ms_AB),
                    "f_statistic": float(f_AB),
                    "p_value": float(p_AB),
                    "significant": bool(p_AB < 0.05)
                },
                "error_b": {
                    "df": int(df_error_b),
                    "ss": float(ss_error_b),
                    "ms": float(ms_error_b)
                },
                "total": {
                    "df": int(df_total),
                    "ss": float(ss_total)
                }
            }

            if r > 0:
                t_a_5 = stats.t.ppf(0.975, df_error_a) if df_error_a > 0 else 2.0
                t_a_1 = stats.t.ppf(0.995, df_error_a) if df_error_a > 0 else 2.0
                t_b_5 = stats.t.ppf(0.975, df_error_b) if df_error_b > 0 else 2.0
                t_b_1 = stats.t.ppf(0.995, df_error_b) if df_error_b > 0 else 2.0

                se_A = np.sqrt(2.0 * ms_error_a / (r * b))
                cd_results.append({
                    "parameter": f"Main Plot Factor A: {ind_var1}",
                    "se_d": float(se_A),
                    "cd_5": float(t_a_5 * se_A),
                    "cd_1": float(t_a_1 * se_A)
                })

                se_B = np.sqrt(2.0 * ms_error_b / (r * a))
                cd_results.append({
                    "parameter": f"Sub Plot Factor B: {ind_var2}",
                    "se_d": float(se_B),
                    "cd_5": float(t_b_5 * se_B),
                    "cd_1": float(t_b_1 * se_B)
                })

                se_B_same_A = np.sqrt(2.0 * ms_error_b / r)
                cd_results.append({
                    "parameter": f"Sub Plot B at same Main Plot A",
                    "se_d": float(se_B_same_A),
                    "cd_5": float(t_b_5 * se_B_same_A),
                    "cd_1": float(t_b_1 * se_B_same_A)
                })

                se_A_same_B = np.sqrt(2.0 * ((b - 1) * ms_error_b + ms_error_a) / (r * b))
                denom = (b - 1) * ms_error_b + ms_error_a
                t_pooled_5 = (((b - 1) * ms_error_b * t_b_5 + ms_error_a * t_a_5) / denom) if denom > 0 else t_b_5
                t_pooled_1 = (((b - 1) * ms_error_b * t_b_1 + ms_error_a * t_a_1) / denom) if denom > 0 else t_b_1
                
                cd_results.append({
                    "parameter": f"Main Plot A at same Sub Plot B",
                    "se_d": float(se_A_same_B),
                    "cd_5": float(t_pooled_5 * se_A_same_B),
                    "cd_1": float(t_pooled_1 * se_A_same_B)
                })

            sns.pointplot(data=df_clean, x=ind_var1, y=dep_var, hue=ind_var2, palette=palette_colors(palette, df_clean[ind_var2].nunique()), markers=["o", "s", "^", "D", "v"], linestyles=["-", "--", "-.", ":", "-"], capsize=0.1, ax=ax)
            ax.set_title(f"Split-plot Interaction: {dep_var} by {ind_var1} & {ind_var2}", fontsize=12, fontweight="bold", pad=15)
            ax.set_xlabel(ind_var1, fontsize=10)
            ax.set_ylabel(dep_var, fontsize=10)
            ax.legend(title=ind_var2, frameon=True, facecolor="white", edgecolor="#E2E8F0")

        elif test_type in ["crd_multifactor", "rbd_multifactor"]:
            # Perform multi-factor CRD or RBD ANOVA
            if test_type == "rbd_multifactor":
                if not rep_var:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Replication/Block variable (rep_var) is required for Multi-Factor RBD."
                    )
                require_complete_design(df_clean, factor_list + [rep_var], "Multi-Factor RBD")
                X, effects_indices = build_multifactor_rbd_design_matrix(df_clean, factor_list, rep_var)
            else:
                require_all_cells(df_clean, factor_list, "Multi-Factor CRD")
                X, effects_indices = build_multifactor_design_matrix(df_clean, factor_list)

            Y = df_clean[dep_var].values
            n_samples = len(df_clean)
            n_coefs = X.shape[1]

            if n_samples <= n_coefs:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Insufficient sample size ({n_samples}) for Multi-Factor ANOVA. Needs more rows than combinations (coefficients={n_coefs})."
                )

            # Fit full model and get RSS
            rss_full = fit_rss(X, Y)

            # Compute RSS and SS for each effect by dropping its columns
            effects_results = []
            for effect_name, indices in effects_indices.items():
                keep_cols = [c for c in range(X.shape[1]) if c not in indices]
                X_reduced = X[:, keep_cols]
                rss_reduced = fit_rss(X_reduced, Y)
                ss_effect = max(0.0, rss_reduced - rss_full)
                df_effect = len(indices)
                effects_results.append({
                    "name": effect_name,
                    "ss": ss_effect,
                    "df": df_effect
                })

            ss_total = ((Y - Y.mean())**2).sum()
            df_total = n_samples - 1
            df_error = n_samples - n_coefs
            ss_error = max(0.0, rss_full)
            ms_error = ss_error / df_error if df_error > 0 else 0.0

            # Build effects list for anova_table
            anova_effects = []
            for eff in effects_results:
                ms_effect = eff["ss"] / eff["df"] if eff["df"] > 0 else 0.0
                f_stat = ms_effect / ms_error if ms_error > 0 else 0.0
                p_val = stats.f.sf(f_stat, eff["df"], df_error) if df_error > 0 else 1.0
                anova_effects.append({
                    "source": eff["name"],
                    "ss": float(eff["ss"]),
                    "df": int(eff["df"]),
                    "ms": float(ms_effect),
                    "f_statistic": float(f_stat),
                    "p_value": float(p_val),
                    "significant": bool(p_val < 0.05)
                })

            # Check if block factor is significant
            significant_rep = False
            if test_type == "rbd_multifactor" and rep_var in effects_indices:
                for eff in anova_effects:
                    if eff["source"] == rep_var:
                        significant_rep = eff["significant"]
                        break

            anova_table = {
                "method": f"Multi-Factor {'RBD' if test_type == 'rbd_multifactor' else 'CRD'} ANOVA (Type III SS)",
                "effects": anova_effects,
                "error": {
                    "source": "Error",
                    "df": int(df_error),
                    "ss": float(ss_error),
                    "ms": float(ms_error)
                },
                "total": {
                    "source": "Total",
                    "df": int(df_total),
                    "ss": float(ss_total)
                },
                "significant_rep": significant_rep
            }

            # Map effects dict for backward-compatibility or direct access
            for eff in anova_effects:
                if eff["source"] == factor_list[0]:
                    anova_table["factorA"] = eff
                elif len(factor_list) > 1 and eff["source"] == factor_list[1]:
                    anova_table["factorB"] = eff
                elif len(factor_list) > 1 and eff["source"] == f"{factor_list[0]}:{factor_list[1]}":
                    anova_table["interaction"] = eff

            if "factorA" not in anova_table:
                # Fallback to the first effect in the list
                anova_table["factorA"] = anova_effects[0]

            # Compute replication number r
            num_combos = np.prod([len(df_clean[f].unique()) for f in factor_list])
            if test_type == "rbd_multifactor" and rep_var:
                r = len(df_clean[rep_var].unique())
            else:
                r = len(df_clean) / num_combos if num_combos > 0 else 1.0

            if ms_error > 0 and r > 0 and df_error > 0:
                t_5 = stats.t.ppf(0.975, df_error)
                t_1 = stats.t.ppf(0.995, df_error)
                
                # Critical difference for the full treatment combinations
                se_combos = np.sqrt(2.0 * ms_error / r)
                cd_results.append({
                    "parameter": "Treatment Combinations",
                    "se_d": float(se_combos),
                    "cd_5": float(t_5 * se_combos),
                    "cd_1": float(t_1 * se_combos)
                })

            # Plot treatment combinations
            plot_df = df_clean.copy()
            plot_df['Treatment'] = plot_df[factor_list].agg(' / '.join, axis=1)
            plot_df = plot_df.sort_values('Treatment')
            
            sns.barplot(data=plot_df, x='Treatment', y=dep_var, color=palette_colors(palette)[0], ax=ax, capsize=0.12,
                        err_kws={'linewidth': 1.0, 'color': COLORS['text']})
            ax.set_title(f"Treatment Combinations: {dep_var} by {', '.join(factor_list)}", fontsize=12, fontweight="bold", pad=15)
            ax.set_xlabel("Treatment (Factor Combinations)", fontsize=10)
            ax.set_ylabel(dep_var, fontsize=10)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

        if test_type in ["twoway", "rbd_twoway", "splitplot", "crd_multifactor", "rbd_multifactor"]:
            residuals = None
            try:
                if test_type == "twoway":
                    cell_means = df_clean.groupby([ind_var1, ind_var2])[dep_var].transform('mean')
                    residuals = df_clean[dep_var] - cell_means
                elif test_type == "rbd_twoway":
                    block_means = df_clean.groupby(rep_var)[dep_var].transform('mean')
                    cell_means = df_clean.groupby([ind_var1, ind_var2])[dep_var].transform('mean')
                    grand_mean = df_clean[dep_var].mean()
                    residuals = df_clean[dep_var] - block_means - cell_means + grand_mean
                elif test_type == "crd_multifactor":
                    cell_means = df_clean.groupby(factor_list)[dep_var].transform('mean')
                    residuals = df_clean[dep_var] - cell_means
                elif test_type == "rbd_multifactor":
                    block_means = df_clean.groupby(rep_var)[dep_var].transform('mean')
                    cell_means = df_clean.groupby(factor_list)[dep_var].transform('mean')
                    grand_mean = df_clean[dep_var].mean()
                    residuals = df_clean[dep_var] - block_means - cell_means + grand_mean
                else:  # splitplot
                    rep_a_means = df_clean.groupby([rep_var, ind_var1])[dep_var].transform('mean')
                    cell_means = df_clean.groupby([ind_var1, ind_var2])[dep_var].transform('mean')
                    a_means = df_clean.groupby(ind_var1)[dep_var].transform('mean')
                    residuals = df_clean[dep_var] - rep_a_means - cell_means + a_means

                if residuals is not None and len(residuals) >= 3:
                    w_stat, w_p = stats.shapiro(residuals)
                    shapiro_results = {
                        "Model Residuals": {
                            "stat": float(w_stat),
                            "p_value": float(w_p),
                            "normal": bool(w_p >= 0.05)
                        }
                    }
                else:
                    shapiro_results = {
                        "Model Residuals": {
                            "error": "Sample size too small for normality test (n < 3)"
                        }
                    }
            except Exception as e_norm:
                shapiro_results = {
                    "Model Residuals": {
                        "error": f"Failed to compute normality check: {str(e_norm)}"
                    }
                }

            try:
                if test_type in ["crd_multifactor", "rbd_multifactor"]:
                    interaction_groups = df_clean.groupby(factor_list)
                else:
                    interaction_groups = df_clean.groupby([ind_var1, ind_var2])
                group_lists_levene = [group[dep_var].values for name, group in interaction_groups]
                valid_groups = [g for g in group_lists_levene if len(g) >= 2]
                if len(valid_groups) >= 2:
                    lev_stat, lev_p = stats.levene(*valid_groups)
                    levene_results = {
                        "stat": float(lev_stat),
                        "p_value": float(lev_p),
                        "equal_var": bool(lev_p >= 0.05)
                    }
            except Exception:
                levene_results = None

        plt.tight_layout()
        plt.savefig(plot_buf, format="png", dpi=150)
        plot_buf.seek(0)
        plt.close(fig)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to run ANOVA calculations due to data issues: {str(e)}"
        )

    img_b64 = base64.b64encode(plot_buf.getvalue()).decode('utf-8')

    descriptives = {}
    if test_type in ["oneway", "rbd_oneway", "lsd"]:
        groups = df_clean.groupby(ind_var1)
        for name, group in groups:
            n_val = int(len(group))
            std_val = float(group[dep_var].std(ddof=1)) if n_val > 1 else 0.0
            se_val = float(std_val / np.sqrt(n_val)) if n_val > 0 else 0.0
            descriptives[str(name)] = {
                "n": n_val,
                "mean": float(group[dep_var].mean()),
                "std": std_val,
                "se": se_val
            }
    elif test_type in ["twoway", "rbd_twoway", "splitplot", "crd_multifactor", "rbd_multifactor"]:
        if test_type in ["crd_multifactor", "rbd_multifactor"]:
            groups = df_clean.groupby(factor_list)
        else:
            groups = df_clean.groupby([ind_var1, ind_var2])
        for name, group in groups:
            if isinstance(name, tuple):
                cell_name = " / ".join(str(x) for x in name)
            else:
                cell_name = str(name)
            n_val = int(len(group))
            std_val = float(group[dep_var].std(ddof=1)) if n_val > 1 else 0.0
            se_val = float(std_val / np.sqrt(n_val)) if n_val > 0 else 0.0
            descriptives[cell_name] = {
                "n": n_val,
                "mean": float(group[dep_var].mean()),
                "std": std_val,
                "se": se_val
            }

    # Calculate Grand Mean
    grand_mean = float(df_clean[dep_var].mean())

    # Determine MSE and r based on test type
    mse = 0.0
    r = 1.0

    if test_type == "oneway":
        mse = anova_table.get("ms_within", 0.0)
        group_names = sorted(list(df_clean[ind_var1].unique()))
        r = len(df_clean) / len(group_names) if len(group_names) > 0 else 1.0
    elif test_type == "rbd_oneway":
        mse = anova_table.get("ms_within", 0.0)
        if rep_var:
            reps = sorted(df_clean[rep_var].unique().tolist())
            r = len(reps) if len(reps) > 0 else 1.0
    elif test_type == "lsd":
        mse = anova_table.get("ms_within", 0.0)
        r = len(df_clean[ind_var1].unique())  # treatments count
    elif test_type == "twoway":
        mse = anova_table.get("error", {}).get("ms", 0.0)
        a_levels = len(df_clean[ind_var1].unique())
        b_levels = len(df_clean[ind_var2].unique()) if ind_var2 else 1
        r = len(df_clean) / (a_levels * b_levels) if (a_levels * b_levels) > 0 else 1.0
    elif test_type == "rbd_twoway":
        mse = anova_table.get("error", {}).get("ms", 0.0)
        if rep_var:
            r_vals = sorted(df_clean[rep_var].unique().tolist())
            r = len(r_vals) if len(r_vals) > 0 else 1.0
    elif test_type == "splitplot":
        mse = anova_table.get("error_b", {}).get("ms", 0.0)
        if rep_var:
            r_vals = sorted(df_clean[rep_var].unique().tolist())
            r = len(r_vals) if len(r_vals) > 0 else 1.0
    elif test_type == "crd_multifactor":
        mse = anova_table.get("error", {}).get("ms", 0.0)
        num_combos = np.prod([len(df_clean[f].unique()) for f in factor_list])
        r = len(df_clean) / num_combos if num_combos > 0 else 1.0
    elif test_type == "rbd_multifactor":
        mse = anova_table.get("error", {}).get("ms", 0.0)
        if rep_var:
            r_vals = sorted(df_clean[rep_var].unique().tolist())
            r = len(r_vals) if len(r_vals) > 0 else 1.0

    # Calculate CV & SE(m)
    cv = (np.sqrt(mse) / grand_mean) * 100 if grand_mean != 0.0 else 0.0
    sem = np.sqrt(mse / r) if r > 0.0 else 0.0

    # Post-hoc comparisons and mean-separation letters. The letters are derived from exactly
    # the pairwise tests reported, so the comparison table and the letters can never disagree.
    posthoc_results = []
    posthoc_letters = {}
    if len(descriptives) >= 2:
        labels = list(descriptives.keys())
        cell_means = [descriptives[label]["mean"] for label in labels]
        cell_ns = [descriptives[label]["n"] for label in labels]
        try:
            if posthoc_method == "games_howell":
                pair_error = welch_pair_error(cell_ns, [descriptives[label]["std"] ** 2 for label in labels])
            elif test_type == "splitplot":
                main_level = {
                    " / ".join(str(x) for x in name): name[0]
                    for name, _ in df_clean.groupby([ind_var1, ind_var2])
                }
                pair_error = splitplot_pair_error(
                    labels, main_level,
                    anova_table["error_a"]["ms"], anova_table["error_a"]["df"],
                    anova_table["error_b"]["ms"], anova_table["error_b"]["df"],
                    df_clean[ind_var2].nunique(), df_clean[rep_var].nunique()
                )
            else:
                if test_type in ("oneway", "rbd_oneway", "lsd"):
                    df_error = anova_table.get("df_within", 0)
                else:
                    df_error = anova_table.get("error", {}).get("df", 0)
                pair_error = pooled_pair_error(cell_ns, mse, df_error)
            posthoc_results = pairwise_comparisons(labels, cell_means, pair_error, method=posthoc_method)
            posthoc_letters = compact_letter_display(labels, cell_means, posthoc_results)
        except PosthocError as e:
            notes.append(f"Post-hoc comparisons were not computed: {e}")
        if test_type == "splitplot" and posthoc_results:
            notes.append(
                "Split-plot letters compare all A x B combinations: pairs within the same main plot use Error(b); "
                "pairs across main plots use the combined error (b-1)Eb + Ea with Satterthwaite degrees of freedom."
            )
    # The classic chart is drawn before the post-hoc test runs, so a one-factor design gets a second
    # chart here: treatment means with their error bars and the letters the comparison table reports.
    letters_plot = None
    if posthoc_letters and len(descriptives) >= 2:
        factor_label = ind_var1 if test_type in ("oneway", "rbd_oneway", "lsd") else "Treatment"
        letters_plot = mean_separation_png(
            descriptives, posthoc_letters,
            title=f"{pretty_label(dep_var)} by {pretty_label(factor_label)}",
            xlabel=factor_label, ylabel=dep_var, palette=palette,
            error_label="± SE",
        )

    return {
        "shapiro_results": shapiro_results,
        "levene_results": levene_results,
        "anova_table": anova_table,
        "posthoc_results": posthoc_results,
        "descriptives": descriptives,
        "cd_results": cd_results,
        "plot": letters_plot or img_b64,
        "plot_classic": img_b64,
        "grand_mean": grand_mean,
        "cv": cv,
        "sem": sem,
        "posthoc_letters": posthoc_letters,
        "posthoc_method": posthoc_method,
        "posthoc_method_label": METHOD_LABELS[posthoc_method],
        "notes": notes
    }
