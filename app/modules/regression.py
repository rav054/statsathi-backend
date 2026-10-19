"""OLS and PLS regression."""

import pandas as pd
import numpy as np
import scipy.stats as stats
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status, Form

from ..auth import get_current_user
from ..models import User
from ..measures import pls_summary

from .common import read_uploaded_table

router = APIRouter()

@router.post("/regression")
def analyze_regression(
    file: UploadFile = File(...),
    regression_type: str = Form(...), # "simple", "multiple", "plsr"
    dep_vars_str: str = Form(...), # Comma separated
    ind_vars_str: str = Form(...), # Comma separated
    n_components: int = Form(2), # For PLSR
    current_user: User = Depends(get_current_user)
):
    import numpy as np
    from scipy import stats
    
    df = read_uploaded_table(file)

    dep_vars = [c.strip() for c in dep_vars_str.split(',') if c.strip()]
    ind_vars = [c.strip() for c in ind_vars_str.split(',') if c.strip()]

    if not dep_vars or not ind_vars:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Both dependent and independent variables must be specified."
        )

    all_vars = dep_vars + ind_vars
    missing_cols = [c for c in all_vars if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Variables not found in dataset: {', '.join(missing_cols)}"
        )

    # Convert all columns to numeric, raising errors for invalid text
    for col in all_vars:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Column '{col}' contains non-numeric data that cannot be analyzed."
            )

    # Clean rows with missing values
    df_clean = df.dropna(subset=all_vars).copy()
    if len(df_clean) < 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset contains empty cells or invalid dimensions for PLSR/Regression. Regression requires at least 5 valid observations after removing missing values or non-numeric cells."
        )

    # Check for zero variance in independent variables
    for col in ind_vars:
        if df_clean[col].var() == 0 or np.isnan(df_clean[col].var()):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Independent variable '{col}' has zero variance (all values are identical). PLSR/Regression cannot be computed."
            )
            
    # Check for zero variance in dependent variables
    for col in dep_vars:
        if df_clean[col].var() == 0 or np.isnan(df_clean[col].var()):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dependent variable '{col}' has zero variance (all values are identical). PLSR/Regression cannot be computed."
            )

    if regression_type in ["simple", "multiple"]:
        # Standard OLS Multiple Linear Regression
        Y = df_clean[dep_vars[0]].values
        X = df_clean[ind_vars].values
        
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # Add intercept
        X_design = np.column_stack((np.ones(n_samples), X))
        
        try:
            # Beta = (X^T X)^-1 X^T Y
            XtX = X_design.T @ X_design
            beta = np.linalg.inv(XtX) @ X_design.T @ Y
            
            # Fitted values and residuals
            Y_pred = X_design @ beta
            residuals = Y - Y_pred
            
            # Sum of Squares
            rss = np.sum(residuals**2)
            y_mean = np.mean(Y)
            tss = np.sum((Y - y_mean)**2)
            
            # Degrees of Freedom
            p = n_features + 1
            df_resid = n_samples - p
            df_model = n_features
            
            if df_resid <= 0:
                raise ValueError("Insufficient degrees of freedom. Add more observations or reduce variables.")
                
            # R2 and Adjusted R2
            r2 = 1.0 - (rss / tss) if tss > 0 else 1.0
            adj_r2 = 1.0 - (rss / df_resid) / (tss / (n_samples - 1)) if tss > 0 and df_resid > 0 else 1.0
            
            # Standard error of coefficients
            s2 = rss / df_resid
            cov_beta = s2 * np.linalg.inv(XtX)
            se_beta = np.sqrt(np.diag(cov_beta))
            
            # t-stats and p-values
            t_stats = beta / se_beta
            p_values = [float(2 * stats.t.sf(np.abs(t), df_resid)) for t in t_stats]
            t_crit = stats.t.ppf(0.975, df_resid)

            def finite_list(values):
                return [float(v) if np.isfinite(v) else None for v in values]

            # Diagnostics: leverage (hat values), internally standardized residuals and Cook's distance
            xtx_inv = np.linalg.inv(XtX)
            leverage = np.einsum("ij,jk,ik->i", X_design, xtx_inv, X_design)
            with np.errstate(divide="ignore", invalid="ignore"):
                standardized = residuals / (np.sqrt(s2) * np.sqrt(1 - leverage))
                cooks = standardized ** 2 * leverage / (p * (1 - leverage))
            order = np.argsort(standardized)
            positions = (np.arange(1, n_samples + 1) - 0.375) / (n_samples + 0.25)  # Blom plotting positions
            diagnostics = {
                "rows": [int(i) + 2 for i in df_clean.index],  # spreadsheet row numbers
                "fitted": Y_pred.tolist(),
                "standardized_residuals": finite_list(standardized),
                "leverage": leverage.tolist(),
                "cooks_distance": finite_list(cooks),
                "cooks_threshold": 4 / n_samples,
                "qq_theoretical": stats.norm.ppf(positions).tolist(),
                "qq_sample": finite_list(standardized[order]),
                "qq_rows": [int(df_clean.index[i]) + 2 for i in order],
            }

            # Simple regression: fitted line with 95% confidence (mean) and prediction (new observation) bands
            fit_band = None
            if n_features == 1:
                x_values = X[:, 0]
                grid = np.linspace(x_values.min(), x_values.max(), 60)
                grid_design = np.column_stack((np.ones(grid.size), grid))
                fitted_line = grid_design @ beta
                se_mean = np.sqrt(np.einsum("ij,jk,ik->i", grid_design, cov_beta, grid_design))
                se_prediction = np.sqrt(se_mean ** 2 + s2)
                fit_band = {
                    "x_name": ind_vars[0],
                    "x_values": x_values.tolist(),
                    "x": grid.tolist(),
                    "fitted": fitted_line.tolist(),
                    "ci_lower": (fitted_line - t_crit * se_mean).tolist(),
                    "ci_upper": (fitted_line + t_crit * se_mean).tolist(),
                    "pi_lower": (fitted_line - t_crit * se_prediction).tolist(),
                    "pi_upper": (fitted_line + t_crit * se_prediction).tolist(),
                }

            # F-stat
            ms_model = (tss - rss) / df_model if df_model > 0 else 0.0
            ms_resid = rss / df_resid if df_resid > 0 else 0.0
            f_stat = ms_model / ms_resid if ms_resid > 0 else 0.0
            f_pvalue = float(stats.f.sf(f_stat, df_model, df_resid)) if df_model > 0 and df_resid > 0 else None
            
            # Coefficients payload
            coefficients_list = []
            var_names = ["Intercept"] + ind_vars
            for i, name in enumerate(var_names):
                coefficients_list.append({
                    "variable": name,
                    "coefficient": float(beta[i]),
                    "std_err": float(se_beta[i]),
                    "t_stat": float(t_stats[i]),
                    "p_value": float(p_values[i]),
                    "ci_lower": float(beta[i] - t_crit * se_beta[i]),
                    "ci_upper": float(beta[i] + t_crit * se_beta[i])
                })
                
            return {
                "regression_type": regression_type,
                "r2": float(r2),
                "adj_r2": float(adj_r2),
                "f_statistic": float(f_stat),
                "f_pvalue": f_pvalue,
                "df_model": int(df_model),
                "df_resid": int(df_resid),
                "coefficients": coefficients_list,
                "actual": Y.tolist(),
                "predicted": Y_pred.tolist(),
                "residuals": residuals.tolist(),
                "diagnostics": diagnostics,
                "fit_band": fit_band
            }
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Regression analysis failed: {str(e)}"
            )

    elif regression_type == "plsr":
        # PLS Regression
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.metrics import r2_score
        
        X = df_clean[ind_vars].values
        Y = df_clean[dep_vars].values
        
        # Ensure Y is 2D
        if len(Y.shape) == 1:
            Y = Y.reshape(-1, 1)
            
        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_targets = Y.shape[1]
        
        # Components sanity check
        if n_components > n_features:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dataset contains empty cells or invalid dimensions for PLSR. Number of components ({n_components}) cannot exceed the number of predictor variables ({n_features})."
            )
        if n_components >= n_samples:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dataset contains empty cells or invalid dimensions for PLSR. Number of components ({n_components}) must be less than the number of observations ({n_samples})."
            )
        comp = n_components
            
        try:
            pls = PLSRegression(n_components=comp)
            pls.fit(X, Y)
            
            Y_pred = pls.predict(X)
            
            # R2 for targets
            r2_vals = r2_score(Y, Y_pred, multioutput='raw_values').tolist()
            
            # Share of standardized X and Y variance explained per component, and VIP scores
            pls_stats = pls_summary(pls, X, Y)
            explained_variance_x = pls_stats["explained_variance_x"]
            # sklearn predicts from centered X; convert the intercept to the original X scale
            intercepts = np.ravel(pls.intercept_) - X.mean(axis=0) @ np.atleast_2d(pls.coef_).T
            
            # Format coefficients matrix
            coefs_list = []
            for i, ind_v in enumerate(ind_vars):
                targets_coef = {}
                for j, dep_v in enumerate(dep_vars):
                    # Handle version differences for coef_ shape (n_features vs n_targets as first dimension)
                    if pls.coef_.shape == (n_features, n_targets):
                        val = float(pls.coef_[i, j])
                    else:
                        val = float(pls.coef_[j, i])
                    targets_coef[dep_v] = val
                coefs_list.append({
                    "variable": ind_v,
                    "coefficients": targets_coef
                })
                
            # Formatting scores and loadings
            x_loadings_payload = []
            for i, ind_v in enumerate(ind_vars):
                x_loadings_payload.append({
                    "variable": ind_v,
                    "loadings": pls.x_loadings_[i].tolist()
                })
                
            y_loadings_payload = []
            for j, dep_v in enumerate(dep_vars):
                y_loadings_payload.append({
                    "variable": dep_v,
                    "loadings": pls.y_loadings_[j].tolist()
                })

            return {
                "regression_type": "plsr",
                "n_components_used": int(comp),
                "r2_values": r2_vals,
                "explained_variance_x": explained_variance_x,
                "explained_variance_y": pls_stats["explained_variance_y"],
                "vip": [{"variable": v, "vip": score} for v, score in zip(ind_vars, pls_stats["vip"])],
                "intercepts": {dep: float(b) for dep, b in zip(dep_vars, np.ravel(intercepts))},
                "notes": [
                    "Explained variance is the share of the standardized X and Y variance captured by each component.",
                    "VIP > 1 marks predictors that matter most for the model (Wold et al., 2001).",
                    "Coefficients are in the original units of each variable; predictors were scaled internally."
                ],
                "coefficients": coefs_list,
                "x_loadings": x_loadings_payload,
                "y_loadings": y_loadings_payload,
                "x_scores": pls.x_scores_.tolist(),
                "y_scores": pls.y_scores_.tolist(),
                "actual": Y.tolist(),
                "predicted": Y_pred.tolist()
            }
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"PLSR analysis failed: {str(e)}"
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported regression type."
        )
