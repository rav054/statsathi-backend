"""Structural equation modeling engines behind /analyze/sem.

CB-SEM   Maximum likelihood through semopy, matching lavaan's default ML estimates
         (validated on Bollen's Political Democracy model, tests/test_sem_reference.py).
PLS-SEM  Lohmoeller's PLS path algorithm with Mode A outer weights and the path
         weighting inner scheme (the SmartPLS / seminr defaults), bootstrap standard
         errors and percentile confidence intervals (Hair et al., A Primer on
         PLS-SEM, 3rd ed., 2022).

Both engines take the specification built by the SEM canvas:
    latents = [{"id", "name", "indicators": [column, ...]}, ...]
    paths   = [{"from": latent_id, "to": latent_id}, ...]
"""
import math
from itertools import combinations

import numpy as np
from scipy import stats

BOOTSTRAP_SEED = 20260910


class SemError(ValueError):
    """The model cannot be estimated as specified; the message is shown to the user."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _clean_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parse_model(latents, paths):
    ids = [lv["id"] for lv in latents]
    if len(set(ids)) != len(ids):
        raise SemError("Latent variable ids must be unique.")
    names = {lv["id"]: lv.get("name") or lv["id"] for lv in latents}
    blocks = {lv["id"]: list(lv.get("indicators", [])) for lv in latents}

    edges = []
    for path in paths:
        src, dst = path.get("from"), path.get("to")
        if src not in blocks or dst not in blocks:
            raise SemError(f"Path {src} -> {dst} refers to an undefined latent variable.")
        if src == dst:
            raise SemError(f"A path from '{names[src]}' to itself is not allowed.")
        if (src, dst) in edges:
            raise SemError(f"The path {names[src]} -> {names[dst]} is specified twice.")
        edges.append((src, dst))

    remaining = set(ids)
    while remaining:
        free = [lv for lv in remaining if not any(d == lv and s in remaining for s, d in edges)]
        if not free:
            loop = ", ".join(sorted(names[lv] for lv in remaining))
            raise SemError(f"The structural model contains a feedback loop ({loop}). Only recursive models are supported.")
        remaining -= set(free)
    return ids, names, blocks, edges


def _standardize(values, labels):
    sd = values.std(axis=0, ddof=1)
    constant = [label for label, s in zip(labels, sd) if not s > 0]
    if constant:
        raise SemError(f"Indicators with zero variance cannot be used: {', '.join(constant)}.")
    return (values - values.mean(axis=0)) / sd


def _cronbach_alpha(z_block):
    k = z_block.shape[1]
    if k < 2:
        return 1.0
    total_variance = z_block.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1.0 - z_block.var(axis=0, ddof=1).sum() / total_variance))


def _reliability(ids, names, blocks, loadings, z, column):
    rows, ave = [], {}
    for lv in ids:
        lam = np.array([loadings[(lv, item)] for item in blocks[lv]])
        ave[lv] = float(np.mean(lam ** 2))
        composite = float(lam.sum() ** 2 / (lam.sum() ** 2 + np.sum(1.0 - lam ** 2)))
        rows.append({
            "latent": lv,
            "latent_name": names[lv],
            "n_indicators": len(blocks[lv]),
            "cronbach_alpha": _cronbach_alpha(z[:, [column[i] for i in blocks[lv]]]),
            "composite_reliability": composite,
            "ave": ave[lv],
        })
    return rows, ave


def _htmt(ids, blocks, abs_corr, column):
    """Heterotrait-monotrait ratio (Henseler, Ringle & Sarstedt 2015) on absolute item correlations."""
    def mean_within(items):
        if len(items) < 2:
            return 1.0
        idx = [column[i] for i in items]
        sub = abs_corr[np.ix_(idx, idx)]
        return float(sub[np.triu_indices(len(idx), 1)].mean())

    result = []
    for a, b in combinations(ids, 2):
        hetero = abs_corr[np.ix_([column[i] for i in blocks[a]], [column[i] for i in blocks[b]])].mean()
        result.append({"latent1": a, "latent2": b,
                       "htmt": float(hetero / math.sqrt(mean_within(blocks[a]) * mean_within(blocks[b])))})
    return result


def _fornell_larcker(ids, ave, latent_corr):
    matrix = [[math.sqrt(ave[a]) if a == b else float(latent_corr[i, j]) for j, b in enumerate(ids)]
              for i, a in enumerate(ids)]
    violations = [
        {"latent1": a, "latent2": b}
        for (i, a), (j, b) in combinations(enumerate(ids), 2)
        if abs(latent_corr[i, j]) >= min(math.sqrt(ave[a]), math.sqrt(ave[b]))
    ]
    return {"latents": ids, "matrix": matrix, "violations": violations}


def _latent_corr_dict(ids, corr):
    return {a: {b: float(corr[i, j]) for j, b in enumerate(ids)} for i, a in enumerate(ids)}


def srmr(sample_cov, implied_cov):
    """Standardized root mean square residual (Bentler), as reported by lavaan."""
    sd = np.sqrt(np.diag(sample_cov))
    residual = (sample_cov - implied_cov) / np.outer(sd, sd)
    lower = np.tril_indices(sample_cov.shape[0])
    return float(np.sqrt(np.mean(residual[lower] ** 2)))


def semopy_srmr(model):
    sigma, _ = model.calc_sigma()
    return srmr(model.mx_cov, sigma)


# ---------------------------------------------------------------------------
# CB-SEM
# ---------------------------------------------------------------------------
def fit_cb_sem(data, latents, paths, residual_covariances=()):
    import semopy

    ids, names, blocks, edges = _parse_model(latents, paths)
    too_few = [names[lv] for lv in ids if len(blocks[lv]) < 2]
    if too_few:
        raise SemError(
            "CB-SEM needs at least 2 indicators per latent variable (3 or more is recommended). "
            f"Too few indicators: {', '.join(too_few)}. Use PLS-SEM for single-item constructs."
        )

    items = [item for lv in ids for item in blocks[lv]]
    column = {item: n for n, item in enumerate(items)}
    # semopy's model syntax cannot contain spaces or symbols, so estimate on safe aliases
    item_alias = {item: f"x{n}" for n, item in enumerate(items)}
    lv_alias = {lv: f"eta{n}" for n, lv in enumerate(ids)}
    alias_item = {v: k for k, v in item_alias.items()}
    alias_lv = {v: k for k, v in lv_alias.items()}

    lines = [f"{lv_alias[lv]} =~ " + " + ".join(item_alias[i] for i in blocks[lv]) for lv in ids]
    for dst in ids:
        predecessors = [src for src, d in edges if d == dst]
        if predecessors:
            lines.append(f"{lv_alias[dst]} ~ " + " + ".join(lv_alias[s] for s in predecessors))
    endogenous = {d for _, d in edges}
    exogenous = [lv for lv in ids if lv not in endogenous]
    for a, b in combinations(exogenous, 2):  # lavaan default: exogenous latent variables covary
        lines.append(f"{lv_alias[a]} ~~ {lv_alias[b]}")
    for a, b in residual_covariances:
        lines.append(f"{item_alias[a]} ~~ {item_alias[b]}")

    frame = data[items].rename(columns=item_alias)
    model = semopy.Model("\n".join(lines))
    try:
        result = model.fit(frame)
    except Exception as e:
        raise SemError(f"CB-SEM estimation failed: {e}") from e
    if not getattr(result, "success", True):
        raise SemError("CB-SEM estimation did not converge. Check that every latent variable has "
                       "strongly correlated indicators and that the model is identified.")

    estimates = model.inspect(std_est=True)
    loadings, loading_rows, path_rows, psi = {}, {}, [], {}
    for _, row in estimates.iterrows():
        lval, op, rval = row["lval"], row["op"], row["rval"]
        values = {
            "estimate": _clean_float(row["Estimate"]),
            "std_estimate": _clean_float(row["Est. Std"]),
            "std_error": _clean_float(row["Std. Err"]),
            "z_value": _clean_float(row["z-value"]),
            "p_value": _clean_float(row["p-value"]),
        }
        if op == "~" and rval in alias_lv and lval in alias_item:
            lv, item = alias_lv[rval], alias_item[lval]
            loadings[(lv, item)] = values["std_estimate"]
            loading_rows[(lv, item)] = {"latent": lv, "indicator": item, "loading": values["std_estimate"],
                                        "estimate": values["estimate"], "std_error": values["std_error"],
                                        "p_value": values["p_value"]}
        elif op == "~" and rval in alias_lv and lval in alias_lv:
            path_rows.append({"from": alias_lv[rval], "to": alias_lv[lval], "coefficient": values["std_estimate"],
                              "estimate": values["estimate"], "std_error": values["std_error"],
                              "z_value": values["z_value"], "p_value": values["p_value"]})
        elif op == "~~" and lval in alias_lv and rval in alias_lv:
            psi[(alias_lv[lval], alias_lv[rval])] = values["std_estimate"]

    if any(v is None for v in loadings.values()) or any(p["coefficient"] is None for p in path_rows):
        raise SemError("The standardized solution could not be computed; the model may not be identified.")
    if any(p["p_value"] is None for p in path_rows):
        raise SemError("Standard errors could not be computed (singular information matrix); "
                       "the model may not be identified.")
    for p in path_rows:
        p["significant"] = p["p_value"] < 0.05
    path_rows.sort(key=lambda p: edges.index((p["from"], p["to"])))

    # Model-implied latent correlations from the standardized solution
    m = len(ids)
    pos = {lv: j for j, lv in enumerate(ids)}
    beta, psi_mat = np.zeros((m, m)), np.zeros((m, m))
    for p in path_rows:
        beta[pos[p["to"]], pos[p["from"]]] = p["coefficient"]
    for (a, b), value in psi.items():
        psi_mat[pos[a], pos[b]] = psi_mat[pos[b], pos[a]] = value if value is not None else 0.0
    inverse = np.linalg.inv(np.eye(m) - beta)
    latent_cov = inverse @ psi_mat @ inverse.T
    latent_corr = latent_cov / np.sqrt(np.outer(np.diag(latent_cov), np.diag(latent_cov)))

    z = _standardize(data[items].to_numpy(dtype=float), items)
    reliability, ave = _reliability(ids, names, blocks, loadings, z, column)
    abs_corr = np.abs(np.corrcoef(z, rowvar=False))

    notes = [
        "Coefficients and loadings are standardized estimates; unstandardized estimates, standard errors "
        "and z-tests come from maximum likelihood (first loading of each construct fixed to 1).",
    ]
    heywood = [f"{names[lv]} / {item}" for (lv, item), lam in loadings.items() if abs(lam) > 1.0]
    if heywood:
        notes.append("Standardized loadings above 1 (Heywood case) for: " + ", ".join(heywood)
                     + ". The solution is improper; revise the measurement model.")

    stat_values = semopy.calc_stats(model).T["Value"]
    fit = {
        "chi_square": _clean_float(stat_values.get("chi2")),
        "df": _clean_float(stat_values.get("DoF")),
        "p_value": _clean_float(stat_values.get("chi2 p-value")),
        "cfi": _clean_float(stat_values.get("CFI")),
        "tli": _clean_float(stat_values.get("TLI")),
        "rmsea": _clean_float(stat_values.get("RMSEA")),
        "srmr": _clean_float(semopy_srmr(model)),
        "aic": _clean_float(stat_values.get("AIC")),
        "bic": _clean_float(stat_values.get("BIC")),
    }
    fit_error = None
    if fit["df"] is not None and fit["df"] <= 0:
        fit_error = "The model is saturated (df = 0), so global model fit cannot be tested."
    elif any(fit[k] is None for k in ("chi_square", "df", "cfi", "tli", "rmsea", "srmr")):
        fit_error = "Global fit indices could not be computed for this model."
    if fit_error:
        fit = None
    else:
        fit["df"] = int(fit["df"])

    return {
        "engine": "CB-SEM: maximum likelihood (semopy, equivalent to lavaan ML)",
        "sem_type": "cb",
        "n_obs": int(len(data)),
        "path_coefficients": path_rows,
        "outer_loadings": [loading_rows[(lv, item)] for lv in ids for item in blocks[lv]],
        "outer_weights": [],
        "reliability_indices": reliability,
        "discriminant_validity": {
            "htmt": _htmt(ids, blocks, abs_corr, column),
            "fornell_larcker": _fornell_larcker(ids, ave, latent_corr),
        },
        "latent_correlations": _latent_corr_dict(ids, latent_corr),
        "r2_values": {lv: float(1.0 - psi[(lv, lv)]) for lv in ids if lv in endogenous and psi.get((lv, lv)) is not None},
        "fit_indices": fit,
        "fit_error": fit_error,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# PLS-SEM
# ---------------------------------------------------------------------------
def _composite_scores(z, blocks_idx, weights):
    scores = np.empty((z.shape[0], len(blocks_idx)))
    scaled = []
    for j, (idx, w) in enumerate(zip(blocks_idx, weights)):
        raw = z[:, idx] @ w
        sd = raw.std(ddof=1)
        if not sd > 0:
            raise SemError("A construct score has zero variance; check the indicators and paths.")
        scaled.append(w / sd)
        scores[:, j] = raw / sd
    return scaled, scores


def _pls_path_algorithm(z, blocks_idx, predecessors, successors, initial=None, max_iter=300, tol=1e-7):
    """Iterate Mode A outer estimation with the path weighting inner scheme until weights converge."""
    n = z.shape[0]
    weights = [np.ones(len(idx)) for idx in blocks_idx] if initial is None else [w.copy() for w in initial]
    weights, scores = _composite_scores(z, blocks_idx, weights)
    for iteration in range(1, max_iter + 1):
        corr = scores.T @ scores / (n - 1)
        proxies = np.zeros_like(scores)
        for j in range(len(blocks_idx)):
            if predecessors[j]:
                p = predecessors[j]
                try:
                    coef = np.linalg.solve(corr[np.ix_(p, p)], corr[p, j])
                except np.linalg.LinAlgError as e:
                    raise SemError("Predictor constructs are perfectly collinear; remove or merge redundant constructs.") from e
                proxies[:, j] += scores[:, p] @ coef
            for k in successors[j]:
                proxies[:, j] += corr[j, k] * scores[:, k]
        new_weights, new_scores = _composite_scores(z, blocks_idx, [z[:, idx].T @ proxies[:, j] for j, idx in enumerate(blocks_idx)])
        change = max(float(np.max(np.abs(nw - w))) for nw, w in zip(new_weights, weights))
        weights, scores = new_weights, new_scores
        if change < tol:
            return weights, scores, iteration
    raise SemError(f"The PLS algorithm did not converge within {max_iter} iterations.")


def _structural_coefficients(scores, predecessors):
    n = scores.shape[0]
    corr = scores.T @ scores / (n - 1)
    coefficients, r2 = {}, {}
    for j, p in enumerate(predecessors):
        if p:
            try:
                beta = np.linalg.solve(corr[np.ix_(p, p)], corr[p, j])
            except np.linalg.LinAlgError as e:
                raise SemError("Predictor constructs are perfectly collinear; remove or merge redundant constructs.") from e
            coefficients[j] = dict(zip(p, beta))
            r2[j] = float(corr[p, j] @ beta)
    return coefficients, r2, corr


def fit_pls_sem(data, latents, paths, bootstrap_samples=1000, seed=BOOTSTRAP_SEED):
    ids, names, blocks, edges = _parse_model(latents, paths)
    if not edges:
        raise SemError("PLS-SEM needs at least one structural path between latent variables.")
    isolated = [names[lv] for lv in ids if not any(lv in edge for edge in edges)]
    if isolated:
        raise SemError(f"Every construct must be connected by a path in PLS-SEM. Not connected: {', '.join(isolated)}.")
    bootstrap_samples = int(bootstrap_samples)
    if not 100 <= bootstrap_samples <= 10000:
        raise SemError("Bootstrap samples must be between 100 and 10,000.")

    items = [item for lv in ids for item in blocks[lv]]
    column = {item: n for n, item in enumerate(items)}
    x = data[items].to_numpy(dtype=float)
    n = x.shape[0]
    z = _standardize(x, items)
    pos = {lv: j for j, lv in enumerate(ids)}
    blocks_idx = [np.array([column[i] for i in blocks[lv]]) for lv in ids]
    predecessors = [[pos[s] for s, d in edges if d == lv] for lv in ids]
    successors = [[pos[d] for s, d in edges if s == lv] for lv in ids]

    weights, scores, iterations = _pls_path_algorithm(z, blocks_idx, predecessors, successors)
    # Orient each construct so that its indicators load positively on average
    for j, idx in enumerate(blocks_idx):
        if (z[:, idx].T @ scores[:, j]).sum() < 0:
            weights[j], scores[:, j] = -weights[j], -scores[:, j]

    coefficients, r2, latent_corr = _structural_coefficients(scores, predecessors)
    estimates = np.array([coefficients[pos[d]][pos[s]] for s, d in edges])

    rng = np.random.default_rng(seed)
    draws, failed = [], 0
    for _ in range(bootstrap_samples):
        sample = x[rng.integers(0, n, n)]
        sd = sample.std(axis=0, ddof=1)
        if np.any(sd <= 0):
            failed += 1
            continue
        try:
            boot_weights, boot_scores, _ = _pls_path_algorithm(
                (sample - sample.mean(axis=0)) / sd, blocks_idx, predecessors, successors, initial=weights)
            for j in range(len(ids)):  # construct-level sign correction
                if boot_weights[j] @ weights[j] < 0:
                    boot_scores[:, j] = -boot_scores[:, j]
            boot_coefficients, _, _ = _structural_coefficients(boot_scores, predecessors)
        except SemError:
            failed += 1
            continue
        draws.append([boot_coefficients[pos[d]][pos[s]] for s, d in edges])
    if len(draws) < bootstrap_samples // 2:
        raise SemError("Bootstrapping failed for most resamples; the sample is too small or the model is unstable.")
    draws = np.array(draws)
    std_errors = draws.std(axis=0, ddof=1)
    ci_lower, ci_upper = np.percentile(draws, [2.5, 97.5], axis=0)

    path_rows = []
    for k, (s, d) in enumerate(edges):
        t_value = estimates[k] / std_errors[k] if std_errors[k] > 0 else None
        p_value = float(2.0 * stats.t.sf(abs(t_value), len(draws) - 1)) if t_value is not None else None
        path_rows.append({
            "from": s, "to": d,
            "coefficient": float(estimates[k]),
            "std_error": _clean_float(std_errors[k]),
            "t_value": _clean_float(t_value),
            "p_value": p_value,
            "ci_lower": float(ci_lower[k]),
            "ci_upper": float(ci_upper[k]),
            "significant": bool(p_value is not None and p_value < 0.05),
        })

    loadings, loading_rows, weight_rows = {}, [], []
    for j, lv in enumerate(ids):
        for item, w in zip(blocks[lv], weights[j]):
            loading = float(z[:, column[item]] @ scores[:, j] / (n - 1))
            loadings[(lv, item)] = loading
            loading_rows.append({"latent": lv, "indicator": item, "loading": loading})
            weight_rows.append({"latent": lv, "indicator": item, "weight": float(w)})

    reliability, ave = _reliability(ids, names, blocks, loadings, z, column)
    abs_corr = np.abs(np.corrcoef(z, rowvar=False))
    r2_values = {ids[j]: value for j, value in r2.items()}
    adjusted_r2 = {
        ids[j]: float(1.0 - (1.0 - value) * (n - 1) / (n - len(predecessors[j]) - 1))
        for j, value in r2.items() if n - len(predecessors[j]) - 1 > 0
    }

    notes = [
        f"Path significance from {len(draws)} bootstrap resamples (seed {seed}); p-values use t with "
        f"{len(draws) - 1} df and 95% percentile confidence intervals are reported.",
        "PLS-SEM does not provide covariance-based fit indices (chi-square, CFI, TLI, RMSEA). Assess the model "
        "with R², indicator loadings, reliability, HTMT and path significance.",
        "HTMT uses absolute indicator correlations; a single-item construct is treated as perfectly reliable.",
    ]
    if failed:
        notes.append(f"{failed} bootstrap resamples could not be estimated and were skipped.")

    return {
        "engine": "PLS-SEM: PLS path algorithm (Mode A, path weighting) with bootstrapping",
        "sem_type": "pls",
        "n_obs": int(n),
        "iterations": int(iterations),
        "path_coefficients": path_rows,
        "outer_loadings": loading_rows,
        "outer_weights": weight_rows,
        "reliability_indices": reliability,
        "discriminant_validity": {
            "htmt": _htmt(ids, blocks, abs_corr, column),
            "fornell_larcker": _fornell_larcker(ids, ave, latent_corr),
        },
        "latent_correlations": _latent_corr_dict(ids, latent_corr),
        "r2_values": r2_values,
        "adjusted_r2_values": adjusted_r2,
        "bootstrap": {"samples": int(len(draws)), "requested": bootstrap_samples, "failed": int(failed), "seed": seed},
        "fit_indices": None,
        "fit_error": None,
        "notes": notes,
    }
