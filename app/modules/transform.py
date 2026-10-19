"""Data transformations and scaling."""

import pandas as pd
import numpy as np
import scipy.stats as stats
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status, Form

from ..auth import get_current_user
from ..models import User
from ..measures import shape_summary

from .common import read_uploaded_table

router = APIRouter()

@router.post("/transform")
def transform_dataset(
    file: UploadFile = File(...),
    columns: str = Form(...),
    method: str = Form(...),
    current_user: User = Depends(get_current_user)
):
    import json
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    df = read_uploaded_table(file)

    try:
        if columns.strip().startswith('['):
            col_list = json.loads(columns)
        else:
            col_list = [c.strip() for c in columns.split(',') if c.strip()]
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid columns parameter. Provide a JSON array or comma-separated column names."
        )

    if not col_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No columns specified for transformation."
        )

    norm_method = method.strip().lower()
    allowed_methods = {
        'log10': 'Log10',
        'ln': 'Ln',
        'sqrt': 'Sqrt',
        'arcsine': 'Arcsine',
        'boxcox': 'BoxCox',
        'yeojohnson': 'YeoJohnson',
        'zscore': 'ZScore',
        'minmax': 'MinMax',
        'snv': 'SNV',
        'msc': 'MSC',
        'sg_smooth': 'SGSmooth',
        'sg_1der': 'SG1stDeriv',
        'sg_2der': 'SG2ndDeriv'
    }

    if norm_method not in allowed_methods:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported transformation method '{method}'. Choose from: {list(allowed_methods.keys())}"
        )

    transformed_cols = []
    lambdas = {}
    transform_notes = []

    if norm_method in ['snv', 'msc', 'sg_smooth', 'sg_1der', 'sg_2der']:
        import scipy.signal as signal

        for col in col_list:
            if col not in df.columns:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Column '{col}' not found in the uploaded dataset."
                )

        try:
            matrix_df = df[col_list].apply(pd.to_numeric, errors='coerce')
            matrix_vals = matrix_df.values.astype(float)
            n_samples, k_cols = matrix_vals.shape

            if norm_method == 'snv':
                spectral_data = df[col_list].apply(pd.to_numeric, errors='coerce')
                row_means = spectral_data.mean(axis=1)
                row_stds = spectral_data.std(axis=1, ddof=1)
                row_stds_safe = row_stds.replace(0, np.nan).fillna(1e-8).apply(lambda v: 1e-8 if abs(v) < 1e-8 else v)
                snv_df = spectral_data.sub(row_means, axis=0).div(row_stds_safe, axis=0)
                res_matrix = snv_df.values

            elif norm_method == 'msc':
                if k_cols > 1:
                    ref_spectrum = np.nanmean(matrix_vals, axis=0)
                    ref_mean = np.nanmean(ref_spectrum)
                    ref_var = np.nanvar(ref_spectrum)

                    res_matrix = np.zeros_like(matrix_vals)
                    for i in range(n_samples):
                        sample_spec = matrix_vals[i, :]
                        if np.isnan(sample_spec).any() or ref_var == 0:
                            res_matrix[i, :] = sample_spec
                        else:
                            cov = np.nanmean((sample_spec - np.nanmean(sample_spec)) * (ref_spectrum - ref_mean))
                            b = cov / ref_var if ref_var > 0 else 1.0
                            a = np.nanmean(sample_spec) - b * ref_mean
                            res_matrix[i, :] = (sample_spec - a) / (b if b != 0 else 1.0)
                else:
                    res_matrix = matrix_vals - np.nanmean(matrix_vals)

            elif norm_method in ['sg_smooth', 'sg_1der', 'sg_2der']:
                deriv_order = 0 if norm_method == 'sg_smooth' else (1 if norm_method == 'sg_1der' else 2)
                poly_order = 2 if deriv_order < 2 else 3

                clean_matrix = pd.DataFrame(matrix_vals).bfill().ffill().fillna(0).values

                if k_cols >= 3:
                    window_len = min(7, k_cols if k_cols % 2 == 1 else k_cols - 1)
                    if window_len < 3:
                        window_len = 3
                    if window_len <= poly_order:
                        poly_order = max(1, window_len - 1)
                    res_matrix = signal.savgol_filter(clean_matrix, window_length=window_len, polyorder=poly_order, deriv=deriv_order, axis=1)
                else:
                    window_len = min(7, n_samples if n_samples % 2 == 1 else n_samples - 1)
                    if window_len < 3:
                        window_len = 3
                    if window_len <= poly_order:
                        poly_order = max(1, window_len - 1)
                    res_matrix = signal.savgol_filter(clean_matrix, window_length=window_len, polyorder=poly_order, deriv=deriv_order, axis=0)

            if norm_method in ['sg_smooth', 'sg_1der', 'sg_2der']:
                transform_notes.append(
                    f"Savitzky-Golay filter: window length {window_len}, polynomial order {poly_order}, "
                    f"derivative order {deriv_order}, applied across {'columns' if k_cols >= 3 else 'rows'}."
                )
            for idx, col in enumerate(col_list):
                new_col_name = f"{col}_{allowed_methods[norm_method]}"
                df[new_col_name] = res_matrix[:, idx]
                transformed_cols.append(new_col_name)

        except Exception as ex:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to apply {allowed_methods[norm_method]} transformation: {str(ex)}"
            )

    else:
        for col in col_list:
            if col not in df.columns:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Column '{col}' not found in the uploaded dataset."
                )

            series = pd.to_numeric(df[col], errors='coerce')
            if series.isnull().all():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Column '{col}' does not contain valid numeric data."
                )

            new_col_name = f"{col}_{allowed_methods[norm_method]}"

            try:
                if norm_method == 'log10':
                    non_nulls = series.dropna()
                    if (non_nulls <= 0).any():
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Cannot apply Log/Box-Cox to zero or negative values in '{col}'. Try Yeo-Johnson instead."
                        )
                    df[new_col_name] = np.log10(series)

                elif norm_method == 'ln':
                    non_nulls = series.dropna()
                    if (non_nulls <= 0).any():
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Cannot apply Log/Box-Cox to zero or negative values in '{col}'. Try Yeo-Johnson instead."
                        )
                    df[new_col_name] = np.log(series)

                elif norm_method == 'sqrt':
                    non_nulls = series.dropna()
                    if (non_nulls < 0).any():
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Cannot apply Square Root to negative values in '{col}'. Try Yeo-Johnson instead."
                        )
                    df[new_col_name] = np.sqrt(series)

                elif norm_method == 'arcsine':
                    non_nulls = series.dropna()
                    if non_nulls.max() > 1.0 and non_nulls.max() <= 100.0:
                        scaled = series / 100.0
                        transform_notes.append(f"'{col}' has values above 1, so it was treated as percentages and divided by 100 before the arcsine transformation.")
                        non_nulls_scaled = scaled.dropna()
                    else:
                        scaled = series
                        non_nulls_scaled = non_nulls

                    if (non_nulls_scaled < 0).any() or (non_nulls_scaled > 1.0).any():
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Arcsine transformation requires proportions between 0 and 1 (or percentages 0-100%). Column '{col}' contains values outside this range."
                        )
                    df[new_col_name] = np.arcsin(np.sqrt(scaled))

                elif norm_method == 'boxcox':
                    non_nulls = series.dropna()
                    if (non_nulls <= 0).any():
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Cannot apply Log/Box-Cox to zero or negative values in '{col}'. Try Yeo-Johnson instead."
                        )
                    res_boxcox, lam = stats.boxcox(non_nulls.values)
                    lambdas[col] = float(lam)
                    transformed_series = pd.Series(index=series.index, dtype=float)
                    transformed_series.loc[non_nulls.index] = res_boxcox
                    df[new_col_name] = transformed_series

                elif norm_method == 'yeojohnson':
                    non_nulls = series.dropna()
                    res_yj, lam = stats.yeojohnson(non_nulls.values)
                    lambdas[col] = float(lam)
                    transformed_series = pd.Series(index=series.index, dtype=float)
                    transformed_series.loc[non_nulls.index] = res_yj
                    df[new_col_name] = transformed_series

                elif norm_method == 'zscore':
                    non_nulls = series.dropna()
                    sd = non_nulls.std(ddof=1)
                    if not sd > 0:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Z-score needs variation in '{col}' (all values are identical)."
                        )
                    # sample standard deviation (n - 1), matching the descriptive statistics module
                    scaled_vals = ((non_nulls - non_nulls.mean()) / sd).values
                    transformed_series = pd.Series(index=series.index, dtype=float)
                    transformed_series.loc[non_nulls.index] = scaled_vals
                    df[new_col_name] = transformed_series

                elif norm_method == 'minmax':
                    non_nulls = series.dropna()
                    scaler = MinMaxScaler()
                    scaled_vals = scaler.fit_transform(non_nulls.values.reshape(-1, 1)).flatten()
                    transformed_series = pd.Series(index=series.index, dtype=float)
                    transformed_series.loc[non_nulls.index] = scaled_vals
                    df[new_col_name] = transformed_series

                transformed_cols.append(new_col_name)

            except HTTPException as he:
                raise he
            except Exception as ex:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to apply {method} to '{col}': {str(ex)}"
                )

    diagnostics = {
        original: {
            "before": shape_summary(pd.to_numeric(df[original], errors="coerce").to_numpy(dtype=float)),
            "after": shape_summary(pd.to_numeric(df[new_col], errors="coerce").to_numpy(dtype=float)),
        }
        for original, new_col in zip(col_list, transformed_cols)
    }
    clean_df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
    records = clean_df.to_dict(orient="records")

    return {
        "message": f"Successfully applied {allowed_methods[norm_method]} transformation to {len(col_list)} column(s).",
        "method": norm_method,
        "method_display": allowed_methods[norm_method],
        "original_columns": col_list,
        "transformed_columns": transformed_cols,
        "lambdas": lambdas,
        "diagnostics": diagnostics,
        "notes": transform_notes,
        "all_columns": list(df.columns),
        "records": records,
        "csv_content": df.to_csv(index=False)
    }
