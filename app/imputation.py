"""Filling missing values with a model, once or many times.

Single imputation (``impute``) replaces each gap with one predicted value - by chained regressions (MICE-style,
scikit-learn's IterativeImputer) or from the nearest complete rows (KNN). The filled table looks complete, but any
analysis run on it treats the guesses as if they had been measured, so standard errors come out too small.

Multiple imputation (``multiple_impute`` + ``rubin``) is the honest version: the gaps are filled m times, each time
with draws from the posterior predictive distribution (IterativeImputer with ``sample_posterior=True``), the analysis
is repeated on every completed table, and the m answers are combined with Rubin's rules. The spread between the m
answers carries the uncertainty that the missing values add.

Pooling follows van Buuren, Flexible Imputation of Missing Data (2nd ed.), section 2.3, with the Barnard-Rubin (1999)
small-sample degrees of freedom - the same formulas as R's ``mice::pool``.
"""

import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (registers IterativeImputer)
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge

BOUNDS = ("none", "non_negative", "observed_range")


class ImputationError(ValueError):
    """Settings that cannot be imputed, with a message meant for the user."""


# ------------------------------------------------------------------------------------------------ the model matrix

def looks_like_identifier(series):
    """A numeric column with a different whole number on every row (plot or serial numbers) predicts nothing."""
    values = series.dropna()
    return (len(values) > 2 and pd.api.types.is_numeric_dtype(series) and values.nunique() == len(values)
            and bool(np.all(np.asarray(values, dtype=float) == np.round(np.asarray(values, dtype=float)))))


def default_predictors(df, targets):
    """Every other numeric column except row identifiers; returns (predictors, identifiers left out)."""
    others = [c for c in df.columns if c not in targets and pd.api.types.is_numeric_dtype(df[c])]
    identifiers = [c for c in others if looks_like_identifier(df[c])]
    return [c for c in others if c not in identifiers], identifiers



def _design(df, targets, predictors):
    """Numeric matrix for the imputer: the target columns first, then numeric and dummy-coded category predictors.

    Returns the matrix, the names of its columns, and how many of them are targets.
    """
    if not targets:
        raise ImputationError("Choose at least one column to impute.")
    unknown = [c for c in list(targets) + list(predictors) if c not in df.columns]
    if unknown:
        raise ImputationError(f"Column(s) not in the dataset: {', '.join(map(str, unknown))}.")
    for name in targets:
        if not pd.api.types.is_numeric_dtype(df[name]):
            raise ImputationError(f"\"{name}\" is not numeric; only numeric columns can be imputed with a model.")
        if df[name].notna().sum() < 3:
            raise ImputationError(f"\"{name}\" has fewer than 3 observed values to learn from.")

    parts = [df[list(targets)].astype(float)]
    for name in predictors:
        if name in targets:
            continue
        column = df[name]
        if pd.api.types.is_numeric_dtype(column):
            parts.append(column.astype(float).to_frame())
        else:
            if column.isna().any():
                raise ImputationError(f"The category column \"{name}\" has gaps; tidy or fill it before using it as a predictor.")
            dummies = pd.get_dummies(column.astype(str), prefix=str(name), drop_first=True, dtype=float)
            if dummies.shape[1] > 50:
                raise ImputationError(f"\"{name}\" has too many categories to use as a predictor.")
            parts.append(dummies)
    matrix = pd.concat(parts, axis=1)
    return matrix, list(matrix.columns), len(targets)


def _bounds(matrix, n_targets, bounds):
    if bounds not in BOUNDS:
        raise ImputationError(f"\"bounds\" must be one of {', '.join(BOUNDS)}.")
    lower = np.full(matrix.shape[1], -np.inf)
    upper = np.full(matrix.shape[1], np.inf)
    if bounds == "non_negative":
        lower[:n_targets] = 0.0
    elif bounds == "observed_range":
        lower[:n_targets] = matrix.iloc[:, :n_targets].min().to_numpy()
        upper[:n_targets] = matrix.iloc[:, :n_targets].max().to_numpy()
    return lower, upper


def _iterative(matrix, n_targets, *, seed, max_iter, bounds, posterior):
    lower, upper = _bounds(matrix, n_targets, bounds)
    imputer = IterativeImputer(
        estimator=BayesianRidge(), sample_posterior=posterior, max_iter=max_iter, random_state=seed,
        min_value=lower, max_value=upper, keep_empty_features=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ConvergenceWarning: the chained equations are stopped at max_iter
        filled = imputer.fit_transform(matrix.to_numpy())
    return filled[:, :n_targets]


# --------------------------------------------------------------------------------------------- single imputation

def impute(df, targets, predictors=None, method="iterative", seed=2026, max_iter=10, n_neighbors=5, bounds="none"):
    """One completed copy of the table: gaps in `targets` filled, every other value untouched."""
    predictors = [c for c in (predictors or []) if c not in targets]
    matrix, _, n_targets = _design(df, targets, predictors)
    if method == "iterative":
        filled = _iterative(matrix, n_targets, seed=seed, max_iter=max_iter, bounds=bounds, posterior=False)
    elif method == "knn":
        if n_neighbors < 1:
            raise ImputationError("KNN needs at least one neighbour.")
        numeric = matrix.select_dtypes("number")
        # neighbours are found on standardised columns, so one measured in grams does not dominate one in tonnes
        scale = numeric.std(ddof=0).replace(0, 1)
        centre = numeric.mean()
        standard = (numeric - centre) / scale
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            knn = KNNImputer(n_neighbors=int(n_neighbors), keep_empty_features=True).fit_transform(standard.to_numpy())
        filled = knn[:, :n_targets] * scale.to_numpy()[:n_targets] + centre.to_numpy()[:n_targets]
        lower, upper = _bounds(matrix, n_targets, bounds)
        filled = np.clip(filled, lower[:n_targets], upper[:n_targets])
    else:
        raise ImputationError("\"method\" must be iterative or knn.")

    result = df.copy()
    for position, name in enumerate(targets):
        gaps = result[name].isna().to_numpy()
        values = result[name].astype(float).to_numpy()
        values[gaps] = filled[gaps, position]
        result[name] = values
    return result


# ------------------------------------------------------------------------------------------- multiple imputation

def multiple_impute(df, targets, predictors=None, m=5, seed=2026, max_iter=10, bounds="none"):
    """m completed copies of the table, each drawn with its own random seed from the posterior predictive."""
    if not 2 <= int(m) <= 100:
        raise ImputationError("The number of imputations must be between 2 and 100.")
    predictors = [c for c in (predictors or []) if c not in targets]
    matrix, _, n_targets = _design(df, targets, predictors)
    completed = []
    for draw in range(int(m)):
        filled = _iterative(matrix, n_targets, seed=int(seed) + draw, max_iter=int(max_iter), bounds=bounds, posterior=True)
        copy = df.copy()
        for position, name in enumerate(targets):
            gaps = copy[name].isna().to_numpy()
            values = copy[name].astype(float).to_numpy()
            values[gaps] = filled[gaps, position]
            copy[name] = values
        completed.append(copy)
    return completed


def rubin(estimates, variances, df_complete=np.inf, confidence=0.95):
    """Combine m estimates of one quantity and their squared standard errors with Rubin's rules.

    Returns the pooled estimate, its standard error, the within (W), between (B) and total (T) variances, the relative
    increase in variance (riv), the proportion of variance due to the missing data (lambda), the fraction of missing
    information (fmi), the Barnard-Rubin degrees of freedom, a confidence interval and the p-value for estimate = 0.
    """
    q = np.asarray(estimates, dtype=float)
    u = np.asarray(variances, dtype=float)
    m = q.size
    if m < 2 or u.size != m:
        raise ImputationError("Rubin's rules need at least two estimates, each with a variance.")

    q_bar = float(q.mean())
    within = float(u.mean())
    between = float(q.var(ddof=1))
    total = within + (1 + 1 / m) * between

    if total <= 0:
        lam, riv = 0.0, 0.0
    else:
        lam = (1 + 1 / m) * between / total
        riv = (1 + 1 / m) * between / within if within > 0 else np.inf

    # Barnard-Rubin: blend the large-sample df with the df of the complete-data analysis. As in mice::pool, lambda is
    # floored at 1e-4 for this, so a quantity the imputations did not change still gets finite degrees of freedom.
    lam_df = max(lam, 1e-4)
    df_old = (m - 1) / lam_df ** 2
    if np.isfinite(df_complete):
        df_observed = (df_complete + 1) / (df_complete + 3) * df_complete * (1 - lam)
        dof = df_old * df_observed / (df_old + df_observed)
    else:
        dof = df_old
    fmi = (riv + 2 / (dof + 3)) / (riv + 1) if np.isfinite(riv) else 1.0

    se = float(np.sqrt(total))
    crit = stats.t.ppf(0.5 + confidence / 2, dof) if np.isfinite(dof) else stats.norm.ppf(0.5 + confidence / 2)
    statistic = q_bar / se if se > 0 else np.nan
    p_value = (2 * stats.t.sf(abs(statistic), dof) if np.isfinite(dof) else 2 * stats.norm.sf(abs(statistic))) \
        if np.isfinite(statistic) else np.nan
    return {
        "estimate": q_bar,
        "se": se,
        "within": within,
        "between": between,
        "total": total,
        "riv": float(riv),
        "lambda": float(lam),
        "fmi": float(fmi),
        "df": float(dof),
        "ci_low": q_bar - crit * se,
        "ci_high": q_bar + crit * se,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "m": int(m),
    }


def pooled_means(completed, columns):
    """The mean of each column, pooled over the completed tables (complete-data variance s²/n, df n - 1)."""
    pooled = {}
    for name in columns:
        estimates, variances = [], []
        for table in completed:
            values = table[name].astype(float).dropna()
            estimates.append(values.mean())
            variances.append(values.var(ddof=1) / len(values))
        n = len(completed[0][name].dropna())
        pooled[name] = rubin(estimates, variances, df_complete=n - 1)
    return pooled


def pooled_regression(completed, dependent, predictors):
    """An OLS regression fitted on every completed table, with coefficients pooled by Rubin's rules.

    R² is pooled on Fisher's z scale of its square root (Harel 2009).
    """
    if not predictors:
        raise ImputationError("Choose at least one predictor for the pooled regression.")
    fits = []
    for table in completed:
        data = table[[dependent] + list(predictors)].dropna()
        if len(data) <= len(predictors) + 1:
            raise ImputationError("Too few complete rows for the regression.")
        X = sm.add_constant(data[list(predictors)].astype(float), has_constant="add")
        fits.append(sm.OLS(data[dependent].astype(float), X).fit())

    names = list(fits[0].params.index)
    df_complete = fits[0].df_resid
    coefficients = []
    for name in names:
        pooled = rubin([f.params[name] for f in fits], [f.bse[name] ** 2 for f in fits], df_complete=df_complete)
        coefficients.append({"term": "Intercept" if name == "const" else name, **pooled})

    z = np.arctanh(np.sqrt(np.clip([f.rsquared for f in fits], 0, 1)))
    r_squared = float(np.tanh(z.mean()) ** 2)
    return {
        "dependent": dependent,
        "predictors": list(predictors),
        "n": int(fits[0].nobs),
        "coefficients": coefficients,
        "r_squared": r_squared,
        "r_squared_each": [float(f.rsquared) for f in fits],
    }


def imputation_diagnostics(original, completed, columns):
    """For each imputed column: how many gaps, and observed values beside imputed ones (a plausibility check)."""
    report = []
    for name in columns:
        gaps = original[name].isna()
        observed = original[name].dropna().astype(float)
        imputed = pd.concat([table.loc[gaps, name] for table in completed]).astype(float)
        per_draw = [float(table.loc[gaps, name].mean()) for table in completed] if gaps.any() else []
        report.append({
            "column": name,
            "missing": int(gaps.sum()),
            "missing_pct": round(100 * float(gaps.mean()), 2),
            "observed_mean": float(observed.mean()) if len(observed) else None,
            "observed_sd": float(observed.std(ddof=1)) if len(observed) > 1 else None,
            "observed_min": float(observed.min()) if len(observed) else None,
            "observed_max": float(observed.max()) if len(observed) else None,
            "imputed_mean": float(imputed.mean()) if len(imputed) else None,
            "imputed_sd": float(imputed.std(ddof=1)) if len(imputed) > 1 else None,
            "imputed_min": float(imputed.min()) if len(imputed) else None,
            "imputed_max": float(imputed.max()) if len(imputed) else None,
            "imputed_mean_by_draw": per_draw,
        })
    return report


def long_format(original, completed):
    """All imputations stacked in one table, as mice's complete(action = "long", include = TRUE) writes them:
    .imp = 0 is the original data with its gaps, .imp = 1..m the completed copies; .id is the original row."""
    frames = []
    for number, table in enumerate([original] + list(completed)):
        frame = table.copy()
        frame.insert(0, ".id", np.arange(1, len(table) + 1))
        frame.insert(0, ".imp", number)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
