"""Structural equation modeling endpoint (engines live in app/sem_engine.py)."""

import pandas as pd
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status, Form

from ..auth import get_current_user
from ..models import User
from ..sem_engine import SemError, fit_cb_sem, fit_pls_sem

from .common import read_uploaded_table

router = APIRouter()

@router.post("/sem")
def analyze_sem(
    file: UploadFile = File(...),
    sem_type: str = Form(...),
    specification: str = Form(...),
    bootstrap_samples: int = Form(1000),
    current_user: User = Depends(get_current_user)
):
    import json
    
    df = read_uploaded_table(file)

    try:
        spec = json.loads(specification)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid SEM model specification JSON: {str(e)}"
        )

    latents = spec.get("latent_variables", [])
    paths = spec.get("paths", [])
    
    if not latents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model specification must define at least one latent variable."
        )

    # Extract all indicators used
    all_indicators = []
    for lv in latents:
        all_indicators.extend(lv.get("indicators", []))
    
    if not all_indicators:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one latent variable must have indicator items mapped."
        )

    duplicated = sorted({c for c in all_indicators if all_indicators.count(c) > 1})
    if duplicated:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Each indicator can belong to only one latent variable. Assigned more than once: {', '.join(duplicated)}"
        )

    missing_cols = [c for c in all_indicators if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Indicator variables not found in dataset: {', '.join(missing_cols)}"
        )

    if sem_type not in ("pls", "cb"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="sem_type must be 'pls' (PLS-SEM) or 'cb' (CB-SEM)."
        )

    # Indicators must be numeric; text cells become missing values
    for col in all_indicators:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    non_numeric = [c for c in all_indicators if df[c].notna().sum() == 0]
    if non_numeric:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Indicators must be numeric. No numeric values found in: {', '.join(non_numeric)}"
        )

    df_clean = df.dropna(subset=all_indicators).copy()
    dropped = len(df) - len(df_clean)
    if len(df_clean) < 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="SEM analysis requires at least 10 valid observations after removing missing values."
        )

    try:
        if sem_type == "cb":
            results_payload = fit_cb_sem(df_clean, latents, paths)
        else:
            results_payload = fit_pls_sem(df_clean, latents, paths, bootstrap_samples=bootstrap_samples)
    except SemError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e

    if dropped:
        results_payload["notes"].append(f"{dropped} rows with missing indicator values were excluded (listwise deletion).")
    return results_payload
