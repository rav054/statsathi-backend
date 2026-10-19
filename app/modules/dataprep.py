"""Data preparation: profile a dataset, then run a recipe of cleaning steps over it."""

import json
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from .. import dataprep as prep
from .. import expressions, imputation, pooled_analyses
from ..auth import get_current_user
from ..models import User
from .common import read_uploaded_table

router = APIRouter()

MAX_STEPS = 50
MAX_ROWS = 200_000
MAX_IMPUTATION_ROWS = 50_000


def bad_request(message):
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)


def read_table(file):
    df = read_uploaded_table(file, str_columns=True)
    if df.empty:
        raise bad_request("The dataset has no rows.")
    if len(df) > MAX_ROWS:
        raise bad_request(f"Datasets up to {MAX_ROWS:,} rows can be prepared here.")
    return df


@router.post("/prep/profile")
def profile_dataset(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    df = read_table(file)
    return {"profile": prep.profile(df), "preview": prep.preview(df)}


@router.post("/prep/formula")
def check_formula(
    file: UploadFile = File(...),
    formula: str = Form(...),
    steps: str = Form("[]"),  # the steps before the calculated column, so it can use what they create
    merge_file: Optional[UploadFile] = File(None),
    current_user: User = Depends(get_current_user)
):
    """Try a formula on the data before it is added to the recipe: the first values, or what is wrong."""
    try:
        earlier = json.loads(steps)
    except json.JSONDecodeError:
        raise bad_request("The steps must be a JSON list.")
    df = read_table(file)
    extra = read_table(merge_file) if merge_file is not None and merge_file.filename else None
    try:
        if earlier:
            df, _ = prep.apply_steps(df, earlier, extra=extra)
        values = expressions.evaluate(df, formula)
    except (prep.StepError, expressions.FormulaError) as error:
        return {"ok": False, "message": str(error)}
    sample = values.head(6).astype(object).where(values.head(6).notna(), None).tolist()
    return {"ok": True, "sample": sample, "missing": int(values.isna().sum()), "rows": int(len(values))}


def split_names(text):
    return [name.strip() for name in (text or "").split(",") if name.strip()]


@router.post("/prep/multiple-imputation")
def run_multiple_imputation(
    file: UploadFile = File(...),
    columns_str: str = Form(...),  # numeric columns whose gaps are imputed
    predictors_str: Optional[str] = Form(None),  # columns that inform the imputation (default: other numeric columns)
    m: int = Form(20),
    max_iter: int = Form(10),
    seed: int = Form(2026),
    bounds: str = Form("none"),  # none | non_negative | observed_range
    regression_dependent: Optional[str] = Form(None),  # optional OLS pooled over the imputations
    regression_predictors_str: Optional[str] = Form(None),
    analyses: Optional[str] = Form(None),  # JSON: {"group_comparison": {...}, "correlation": {...}, "anova": {...}}
    current_user: User = Depends(get_current_user)
):
    df = read_table(file)
    if len(df) > MAX_IMPUTATION_ROWS:
        raise bad_request(f"Multiple imputation is available for datasets up to {MAX_IMPUTATION_ROWS:,} rows.")
    targets = split_names(columns_str)
    missing = [c for c in targets if c not in df.columns]
    if missing:
        raise bad_request(f"Column(s) not in the dataset: {', '.join(missing)}.")
    without_gaps = [c for c in targets if df[c].notna().all()]
    if without_gaps:
        raise bad_request(f"Nothing to impute in {', '.join(without_gaps)}: these columns have no missing values.")
    if not 1 <= max_iter <= 50:
        raise bad_request("Iterations must be between 1 and 50.")

    if predictors_str:
        predictors, identifiers = split_names(predictors_str), []
    else:
        predictors, identifiers = imputation.default_predictors(df, targets)

    regression_predictors = split_names(regression_predictors_str)
    if regression_dependent:
        unknown = [c for c in [regression_dependent] + regression_predictors if c not in df.columns]
        if unknown:
            raise bad_request(f"Regression column(s) not in the dataset: {', '.join(unknown)}.")

    try:
        requested = json.loads(analyses) if analyses else {}
    except json.JSONDecodeError:
        raise bad_request("The pooled analyses must be sent as JSON.")
    if not isinstance(requested, dict):
        raise bad_request("The pooled analyses must be a JSON object.")
    unknown = pooled_analyses.check_columns(df, requested)
    if unknown:
        raise bad_request(f"Analysis column(s) not in the dataset: {', '.join(map(str, unknown))}.")

    try:
        completed = imputation.multiple_impute(df, targets, predictors, m=m, seed=seed, max_iter=max_iter, bounds=bounds)
        means = imputation.pooled_means(completed, targets)
        regression = imputation.pooled_regression(completed, regression_dependent, regression_predictors) \
            if regression_dependent else None
        pooled = pooled_analyses.run_pooled_analyses(completed, requested)
    except imputation.ImputationError as error:
        raise bad_request(str(error))

    diagnostics = imputation.imputation_diagnostics(df, completed, targets)
    for entry in diagnostics:
        entry["pooled_mean"] = means[entry["column"]]

    notes = [
        f"{m} imputations by chained equations (scikit-learn IterativeImputer with Bayesian ridge regression, "
        f"draws from the posterior predictive), {max_iter} iterations each, seeds {seed}-{seed + m - 1}.",
        "Results are pooled with Rubin's rules and Barnard-Rubin degrees of freedom, as in R's mice::pool.",
        "Multiple imputation assumes the values are missing at random given the predictors; include every "
        "variable that relates to the gaps or to the analysis.",
    ]
    if identifiers:
        notes.append(f"{', '.join(identifiers)} left out of the predictors: a different whole number on every row "
                     f"looks like an identifier. Choose predictors yourself to include it.")
    if any(entry["fmi"] > 0.5 for entry in means.values()):
        notes.append("Some fractions of missing information are above 0.5 - consider more imputations (m) "
                     "so the pooled results are stable.")
    if regression is not None and regression_dependent not in targets and df[regression_dependent].isna().any():
        notes.append(f"{regression_dependent} was not imputed, so its incomplete rows are left out of each regression.")

    return {
        "settings": {"columns": targets, "predictors": predictors, "m": m, "max_iter": max_iter, "seed": seed,
                     "bounds": bounds, "n_rows": int(len(df))},
        "columns": diagnostics,
        "regression": regression,
        **pooled,
        "notes": notes,
        "long_csv": imputation.long_format(df, completed).to_csv(index=False),
    }


@router.post("/prep/apply")
def apply_recipe(
    file: UploadFile = File(...),
    steps: str = Form(...),  # JSON list of step objects, run in order
    merge_file: Optional[UploadFile] = File(None),  # the second table a "merge" step joins
    current_user: User = Depends(get_current_user)
):
    try:
        recipe = json.loads(steps)
    except json.JSONDecodeError:
        raise bad_request("The steps must be a JSON list.")
    if not isinstance(recipe, list) or not recipe:
        raise bad_request("Add at least one step.")
    if len(recipe) > MAX_STEPS:
        raise bad_request(f"A recipe can have up to {MAX_STEPS} steps.")

    df = read_table(file)
    extra = read_table(merge_file) if merge_file is not None and merge_file.filename else None

    try:
        result, log = prep.apply_steps(df, recipe, extra=extra)
    except prep.StepError as error:
        raise bad_request(str(error))

    return {
        "log": log,
        "before": prep.profile(df),
        "after": prep.profile(result),
        "preview": prep.preview(result),
        "csv_content": result.to_csv(index=False),
    }
