"""Analyses repeated on every multiply imputed dataset and combined into one answer.

* Two-group comparison: the mean difference and its standard error, pooled with Rubin's rules.
* Correlations: Fisher's z = atanh(r) pooled with Rubin's rules (variance 1/(n - 3), or 1.06/(n - 3) for Spearman's
  coefficient, Fieller, Hartley & Pearson 1957), then transformed back.
* ANOVA: each term's coefficients (sum-to-zero coding, so a main effect is tested as in a Type III table) are tested
  jointly with the D1 multivariate Wald statistic of Li, Raghunathan & Rubin (1991); level means and pairwise
  differences are pooled with Rubin's rules and summarised with compact letters.

Rubin's rules and the Barnard-Rubin degrees of freedom come from ``imputation.rubin``.
"""

from itertools import combinations

import numpy as np
import scipy.stats as stats
import statsmodels.formula.api as smf

from .imputation import ImputationError, rubin
from .posthoc import compact_letter_display


def _q(name):
    """A column name quoted for a model formula (names may contain spaces or symbols)."""
    return 'Q("' + str(name).replace("\\", "\\\\").replace('"', '\\"') + '")'


# ------------------------------------------------------------------------------------------------ two groups

def pooled_group_comparison(completed, outcome, group, confidence=0.95):
    """Mean difference between the two levels of `group`, pooled over the imputations (Welch standard error)."""
    levels = sorted(completed[0][group].dropna().astype(str).unique())
    if len(levels) != 2:
        raise ImputationError(f"\"{group}\" must have exactly two groups for this comparison (it has {len(levels)}).")
    estimates, variances, ds, means = [], [], [], {level: [] for level in levels}
    df_complete = None
    for table in completed:
        data = table[[outcome, group]].dropna()
        a = data.loc[data[group].astype(str) == levels[0], outcome].astype(float)
        b = data.loc[data[group].astype(str) == levels[1], outcome].astype(float)
        if len(a) < 2 or len(b) < 2:
            raise ImputationError("Each group needs at least two observations.")
        estimates.append(a.mean() - b.mean())
        variances.append(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        pooled_sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        ds.append((a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else np.nan)
        means[levels[0]].append(a.mean())
        means[levels[1]].append(b.mean())
        df_complete = len(a) + len(b) - 2
    pooled = rubin(estimates, variances, df_complete=df_complete, confidence=confidence)
    return {
        "outcome": outcome,
        "group": group,
        "levels": levels,
        "difference": f"{levels[0]} - {levels[1]}",
        "means": {level: float(np.mean(values)) for level, values in means.items()},
        **pooled,
        "cohens_d": float(np.nanmean(ds)),  # averaged over imputations
    }


# ------------------------------------------------------------------------------------------------ correlations

def pooled_correlations(completed, variables, method="pearson", confidence=0.95):
    """Every pair of variables: r pooled on Fisher's z scale."""
    if method not in ("pearson", "spearman"):
        raise ImputationError("The correlation method must be pearson or spearman.")
    if len(variables) < 2:
        raise ImputationError("Choose at least two variables to correlate.")
    inflation = 1.06 if method == "spearman" else 1.0
    pairs = []
    for x, y in combinations(variables, 2):
        z, u, rs, n_min = [], [], [], None
        for table in completed:
            data = table[[x, y]].dropna().astype(float)
            n = len(data)
            if n < 4:
                raise ImputationError(f"{x} and {y} have fewer than four complete pairs.")
            r = data[x].corr(data[y], method=method)
            r = float(np.clip(r, -0.999999, 0.999999))
            rs.append(r)
            z.append(np.arctanh(r))
            u.append(inflation / (n - 3))
            n_min = n if n_min is None else min(n_min, n)
        pooled = rubin(z, u, df_complete=n_min - 3, confidence=confidence)
        pairs.append({
            "x": x,
            "y": y,
            "r": float(np.tanh(pooled["estimate"])),
            "ci_low": float(np.tanh(pooled["ci_low"])),
            "ci_high": float(np.tanh(pooled["ci_high"])),
            "p_value": pooled["p_value"],
            "fmi": pooled["fmi"],
            "df": pooled["df"],
            "n": int(n_min),
            "r_each": rs,
        })
    return {"method": method, "variables": list(variables), "pairs": pairs}


# ------------------------------------------------------------------------------------------------ ANOVA

def d1_wald(estimates, covariances, df_complete=np.inf):
    """Pool a multi-parameter Wald test (the D1 statistic of Li, Raghunathan & Rubin, 1991).

    `estimates` is a list of m coefficient vectors (length k) and `covariances` the matching k × k covariance
    matrices. The F reference degrees of freedom follow LRR (1991) and are capped at the complete-data residual df,
    so the pooled test is never more liberal than the same test with no missing data.
    """
    q = np.asarray(estimates, dtype=float)
    u = np.asarray(covariances, dtype=float)
    m, k = q.shape
    q_bar = q.mean(axis=0)
    u_bar = u.mean(axis=0)
    between = np.atleast_2d(np.cov(q, rowvar=False, ddof=1)) if m > 1 else np.zeros((k, k))
    u_inv = np.linalg.pinv(u_bar)
    r1 = float((1 + 1 / m) * np.trace(between @ u_inv) / k)
    statistic = float(q_bar @ u_inv @ q_bar / (k * (1 + r1)))

    t = k * (m - 1)
    if r1 <= 1e-12:
        df2 = np.inf
    elif t > 4:
        df2 = 4 + (t - 4) * (1 + (1 - 2 / t) / r1) ** 2
    else:
        df2 = t * (1 + 1 / k) * (1 + 1 / r1) ** 2 / 2
    df2 = float(min(df2, df_complete))
    p_value = float(stats.f.sf(statistic, k, df2)) if np.isfinite(df2) else float(stats.chi2.sf(statistic * k, k))
    return {"F": statistic, "df1": int(k), "df2": df2, "p_value": p_value, "riv": r1}


def _formula(response, factors, block, interaction):
    terms = [f"C({_q(f)}, Sum)" for f in factors]
    rhs = " * ".join(terms) if interaction and len(terms) == 2 else " + ".join(terms)
    if block:
        rhs += f" + C({_q(block)}, Sum)"
    return f"{_q(response)} ~ {rhs}"


def _term_label(term, factors, block):
    names = []
    for factor in [*factors, block]:
        if factor and f"C({_q(factor)}, Sum)" in term:
            names.append(str(factor))
    return " × ".join(names) if names else term


def pooled_anova(completed, response, factors, block=None, interaction=True, comparison="bonferroni", alpha=0.05):
    """A fixed-effects ANOVA fitted on every completed table, each term tested with D1, means compared pairwise."""
    factors = [f for f in factors if f]
    if not 1 <= len(factors) <= 2:
        raise ImputationError("Choose one or two treatment factors.")
    if comparison not in ("bonferroni", "lsd"):
        raise ImputationError("Pairwise comparisons must be bonferroni or lsd.")
    columns = [response, *factors] + ([block] if block else [])

    fits = []
    for table in completed:
        data = table[columns].dropna().copy()
        for name in factors + ([block] if block else []):
            data[name] = data[name].astype(str)
        if data[factors[0]].nunique() < 2:
            raise ImputationError(f"\"{factors[0]}\" needs at least two levels.")
        fits.append((smf.ols(_formula(response, factors, block, interaction), data=data).fit(), data))

    first_model = fits[0][0]
    slices = first_model.model.data.design_info.term_name_slices
    df_resid = float(first_model.df_resid)
    if df_resid <= 0:
        raise ImputationError("There are no residual degrees of freedom: add replication or drop the interaction.")

    table_rows = []
    for term, where in slices.items():
        if term == "Intercept":
            continue
        tested = d1_wald([fit.params.to_numpy()[where] for fit, _ in fits],
                         [fit.cov_params().to_numpy()[where, where] for fit, _ in fits], df_complete=df_resid)
        table_rows.append({"term": _term_label(term, factors, block), **tested})

    comparisons = {}
    for factor in factors:
        levels = sorted(fits[0][1][factor].unique(), key=lambda v: (len(v), v))
        level_means, pair_rows = [], []
        per_draw = []
        for fit, data in fits:
            groups = data.groupby(factor)[response]
            per_draw.append((groups.mean(), groups.size(), float(fit.mse_resid)))
        for level in levels:
            pooled = rubin([means[level] for means, _, _ in per_draw],
                           [mse / sizes[level] for _, sizes, mse in per_draw], df_complete=df_resid)
            level_means.append({"level": level, "mean": pooled["estimate"], "se": pooled["se"],
                                "ci_low": pooled["ci_low"], "ci_high": pooled["ci_high"]})
        n_pairs = len(levels) * (len(levels) - 1) // 2
        for i, j in combinations(range(len(levels)), 2):
            a, b = levels[i], levels[j]
            pooled = rubin([means[b] - means[a] for means, _, _ in per_draw],
                           [mse * (1 / sizes[a] + 1 / sizes[b]) for _, sizes, mse in per_draw], df_complete=df_resid)
            p = min(1.0, pooled["p_value"] * n_pairs) if comparison == "bonferroni" else pooled["p_value"]
            pair_rows.append({"group1": a, "group2": b, "mean_diff": pooled["estimate"], "se_diff": pooled["se"],
                              "df": pooled["df"], "p_value": p, "significant": bool(p < alpha), "fmi": pooled["fmi"]})
        letters = compact_letter_display(levels, [row["mean"] for row in level_means], pair_rows)
        for row in level_means:
            row["group"] = letters[row["level"]]
        comparisons[factor] = {"means": level_means, "pairs": pair_rows}

    return {
        "response": response,
        "factors": factors,
        "block": block,
        "interaction": bool(interaction and len(factors) == 2),
        "n": int(first_model.nobs),
        "df_residual": df_resid,
        "terms": table_rows,
        "comparisons": comparisons,
        "comparison_method": comparison,
        "alpha": alpha,
    }


def run_pooled_analyses(completed, analyses):
    """Run the analyses requested as {"group_comparison": {...}, "correlation": {...}, "anova": {...}}."""
    out = {}
    request = analyses or {}
    if request.get("group_comparison"):
        spec = request["group_comparison"]
        out["group_comparison"] = pooled_group_comparison(completed, spec.get("outcome"), spec.get("group"))
    if request.get("correlation"):
        spec = request["correlation"]
        out["correlation"] = pooled_correlations(completed, spec.get("variables") or [], spec.get("method", "pearson"))
    if request.get("anova"):
        spec = request["anova"]
        try:
            out["anova"] = pooled_anova(completed, spec.get("response"), spec.get("factors") or [], spec.get("block") or None,
                                        bool(spec.get("interaction", True)), spec.get("comparison", "bonferroni"),
                                        float(spec.get("alpha", 0.05)))
        except ImputationError:
            raise
        except Exception as error:  # e.g. a factor level with no observations, or a non-numeric response
            raise ImputationError(f"The pooled ANOVA could not be fitted: {error}")
    return out


def check_columns(df, analyses):
    """Names in the analysis request that are not in the dataset."""
    names = []
    request = analyses or {}
    if request.get("group_comparison"):
        names += [request["group_comparison"].get("outcome"), request["group_comparison"].get("group")]
    if request.get("correlation"):
        names += list(request["correlation"].get("variables") or [])
    if request.get("anova"):
        spec = request["anova"]
        names += [spec.get("response"), *(spec.get("factors") or []), spec.get("block")]
    return [n for n in names if n and n not in df.columns]
