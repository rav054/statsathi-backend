"""Genotype × environment (G×E) analysis of multi-environment trials.

- AMMI (Gauch 1988; Zobel et al. 1988): additive main effects, with the interaction partitioned by singular value
  decomposition into IPCA terms with Gollob (1968) degrees of freedom; AMMI stability value (Purchase et al. 2000)
- GGE biplot (Yan et al. 2000; Yan & Kang 2003; Yan & Tinker 2006): environment-centred data, singular value
  partitioning, which-won-where, mean vs stability, discriminativeness vs representativeness, ideal genotype
- Stability statistics: Eberhart & Russell (1966) regression slope and deviations (the slope is also Finlay &
  Wilkinson's 1963 adaptability), Wricke's (1962) ecovalence, Shukla's (1972) stability variance, and the
  coefficient of variation across environments
"""

import numpy as np
import scipy.stats as stats
from scipy.spatial import ConvexHull

# Singular value partitioning: exponents of the singular values given to genotype and environment coordinates
SVP_EXPONENTS = {"genotype": (1.0, 0.0), "environment": (0.0, 1.0), "symmetric": (0.5, 0.5)}


def f_test(ms_effect, df_effect, ms_error, df_error):
    if ms_effect is None or ms_error is None or ms_error <= 0 or df_effect <= 0 or df_error <= 0:
        return None, None
    f = ms_effect / ms_error
    return float(f), float(stats.f.sf(f, df_effect, df_error))


def anova_row(source, df, ss, f=None, p=None, component=False):
    return {"source": source, "df": int(df), "ss": float(ss), "ms": float(ss / df) if df > 0 else None,
            "f": f, "p": p, "component": component}


def double_centre(means):
    means = np.asarray(means, dtype=float)
    return means - means.mean(axis=1, keepdims=True) - means.mean(axis=0, keepdims=True) + means.mean()


def orient_components(u, vt):
    """Deterministic signs: the largest environment element of each component is positive."""
    u, vt = u.copy(), vt.copy()
    for k in range(vt.shape[0]):
        if vt[k, np.argmax(np.abs(vt[k]))] < 0:
            u[:, k] *= -1
            vt[k] *= -1
    return u, vt


# ---------------------------------------------------------------------------
# AMMI
# ---------------------------------------------------------------------------
def ammi_anova(values, genotype, environment, n_genotypes, n_environments, replicates=1, block=None):
    """Combined ANOVA of a balanced trial with the G×E interaction split into IPCA terms.

    values, genotype, environment (and block, the replicate number within each environment) are long arrays with
    `replicates` observations in every genotype × environment cell. With one value per cell there is no error term
    and no F-tests.
    """
    y = np.asarray(values, dtype=float)
    g = np.asarray(genotype, dtype=int)
    e = np.asarray(environment, dtype=int)
    G, E, r = n_genotypes, n_environments, replicates

    cell = np.zeros((G, E))
    np.add.at(cell, (g, e), y)
    cell /= r
    grand = float(y.mean())
    genotype_means, environment_means = cell.mean(axis=1), cell.mean(axis=0)
    interaction = double_centre(cell)

    ss_total = float(((y - grand) ** 2).sum())
    ss_env = float(G * r * ((environment_means - grand) ** 2).sum())
    ss_gen = float(E * r * ((genotype_means - grand) ** 2).sum())
    ss_ge = float(r * (interaction ** 2).sum())
    df_env, df_gen, df_ge = E - 1, G - 1, (G - 1) * (E - 1)
    if block is not None:
        b = np.asarray(block, dtype=int)
        block_means = np.zeros((E, r))
        np.add.at(block_means, (e, b), y)
        block_means /= G
        ss_block = float(G * ((block_means - environment_means[:, None]) ** 2).sum())
        df_block = E * (r - 1)
        df_error = E * (G - 1) * (r - 1)
    else:
        ss_block, df_block = 0.0, 0
        df_error = G * E * (r - 1)
    ss_error = max(ss_total - ss_env - ss_gen - ss_ge - ss_block, 0.0) if df_error > 0 else 0.0
    ms_error = ss_error / df_error if df_error > 0 else None

    rows = []
    if block is not None:  # environments are tested against replicates within environments
        f, p = f_test(ss_env / df_env, df_env, ss_block / df_block, df_block)
    else:
        f, p = f_test(ss_env / df_env, df_env, ms_error, df_error)
    rows.append(anova_row("Environments", df_env, ss_env, f, p))
    if block is not None:
        rows.append(anova_row("Replications within environments", df_block, ss_block,
                              *f_test(ss_block / df_block, df_block, ms_error, df_error)))
    rows.append(anova_row("Genotypes", df_gen, ss_gen, *f_test(ss_gen / df_gen, df_gen, ms_error, df_error)))
    rows.append(anova_row("Genotype × environment", df_ge, ss_ge, *f_test(ss_ge / df_ge, df_ge, ms_error, df_error)))

    u, s, vt = np.linalg.svd(interaction, full_matrices=False)
    tolerance = 1e-10 * max(1.0, float(s[0]) if s.size else 1.0)
    ipca, used_ss, used_df, cumulative = [], 0.0, 0, 0.0
    for k in range(min(G - 1, E - 1)):
        df_k = G + E - 1 - 2 * (k + 1)  # Gollob (1968)
        if df_k <= 0 or s[k] <= tolerance:
            break
        ss_k = float(r * s[k] ** 2)
        f, p = f_test(ss_k / df_k, df_k, ms_error, df_error)
        percent = 100 * ss_k / ss_ge if ss_ge > 0 else 0.0
        cumulative += percent
        ipca.append({"component": f"IPCA{k + 1}", "df": df_k, "ss": ss_k, "ms": ss_k / df_k, "f": f, "p": p,
                     "percent": percent, "cumulative": cumulative, "singular_value": float(s[k])})
        rows.append(anova_row(f"IPCA{k + 1}", df_k, ss_k, f, p, component=True))
        used_ss += ss_k
        used_df += df_k
    residual_df, residual_ss = df_ge - used_df, max(ss_ge - used_ss, 0.0)
    if residual_df > 0:
        rows.append(anova_row("Residual", residual_df, residual_ss,
                              *f_test(residual_ss / residual_df, residual_df, ms_error, df_error), component=True))
    if df_error > 0:
        rows.append(anova_row("Pooled error" if block is not None else "Error", df_error, ss_error))
    rows.append(anova_row("Total", y.size - 1, ss_total))

    return {"rows": rows, "ipca": ipca, "ms_error": ms_error, "df_error": int(df_error), "grand_mean": grand,
            "interaction": interaction, "replicates": int(r), "blocks": block is not None, "ss_ge": ss_ge}


def ammi_scores(interaction, k=2):
    """Genotype and environment IPCA scores: singular vectors multiplied by the square root of the singular value."""
    u, s, vt = np.linalg.svd(np.asarray(interaction, dtype=float), full_matrices=False)
    u, vt = orient_components(u, vt)
    root = np.sqrt(s[:k])
    return u[:, :k] * root, vt[:k].T * root, s


def ammi_stability_value(genotype_scores, ss_ipca1, ss_ipca2):
    """ASV = sqrt[(SS_IPCA1 / SS_IPCA2 × IPCA1 score)² + IPCA2 score²]; smaller is more stable."""
    if ss_ipca2 is None or ss_ipca2 <= 0:
        return None
    scores = np.asarray(genotype_scores, dtype=float)
    return np.sqrt((ss_ipca1 / ss_ipca2 * scores[:, 0]) ** 2 + scores[:, 1] ** 2)


# ---------------------------------------------------------------------------
# Stability statistics (on genotype × environment means)
# ---------------------------------------------------------------------------
def stability_statistics(means, ms_error=None, replicates=1, df_error=0):
    Y = np.asarray(means, dtype=float)
    G, E = Y.shape
    grand = Y.mean()
    genotype_means, environment_means = Y.mean(axis=1), Y.mean(axis=0)
    index = environment_means - grand
    sxx = float((index ** 2).sum())
    if sxx <= 0:
        raise ValueError("All environments have the same mean, so regression stability cannot be estimated.")

    centred = Y - genotype_means[:, None]
    slope = centred @ index / sxx
    deviations = centred - slope[:, None] * index[None, :]
    ss_deviation = (deviations ** 2).sum(axis=1)
    ms_deviation = ss_deviation / (E - 2)
    se_slope = np.sqrt(ms_deviation / sxx)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (slope - 1) / se_slope
        p_slope = np.where(se_slope > 0, 2 * stats.t.sf(np.abs(t), E - 2), np.nan)
        ss_row = (centred ** 2).sum(axis=1)
        r2 = np.where(ss_row > 0, 1 - ss_deviation / ss_row, np.nan)
        cv = np.where(genotype_means != 0, 100 * Y.std(axis=1, ddof=1) / np.abs(genotype_means), np.nan)

    s2di = p_deviation = None
    if ms_error is not None and df_error > 0 and ms_error > 0:
        error_of_mean = ms_error / replicates
        s2di = ms_deviation - error_of_mean
        p_deviation = stats.f.sf(ms_deviation / error_of_mean, E - 2, df_error)

    ecovalence = (double_centre(Y) ** 2).sum(axis=1)
    total = ecovalence.sum()
    shukla = G / ((G - 2) * (E - 1)) * ecovalence - total / ((G - 1) * (G - 2) * (E - 1))
    return {
        "environment_index": index,
        "bi": slope,
        "bi_se": se_slope,
        "bi_p": p_slope,
        "deviation_ms": ms_deviation,
        "s2di": s2di,
        "s2di_p": p_deviation,
        "r2": r2,
        "ecovalence": ecovalence,
        "ecovalence_percent": 100 * ecovalence / total if total > 0 else np.zeros(G),
        "shukla": shukla,
        "cv": cv,
    }


# ---------------------------------------------------------------------------
# GGE biplot
# ---------------------------------------------------------------------------
def gge_decomposition(means, scaling="none"):
    """SVD of the environment-centred genotype × environment table (optionally scaled by each environment's SD)."""
    Y = np.asarray(means, dtype=float)
    X = Y - Y.mean(axis=0)
    if scaling == "sd":
        sd = Y.std(axis=0, ddof=1)
        if np.any(sd <= 0):
            raise ValueError("An environment has the same value for every genotype, so it cannot be scaled by its "
                             "standard deviation. Use 'no scaling'.")
        X = X / sd
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    k = min(2, s.size)
    u, vt = u[:, :k].copy(), vt[:k].copy()
    # PC1 points towards the average environment; PC2 by its largest environment element
    if vt[0].sum() < 0:
        u[:, 0] *= -1
        vt[0] *= -1
    if k > 1 and vt[1, np.argmax(np.abs(vt[1]))] < 0:
        u[:, 1] *= -1
        vt[1] *= -1
    total = float((s ** 2).sum())
    return {"centred": X, "U": u, "V": vt.T, "singular_values": s,
            "percent": [float(100 * v ** 2 / total) if total > 0 else 0.0 for v in s[:k]]}


def gge_coordinates(decomposition, svp):
    genotype_exponent, environment_exponent = SVP_EXPONENTS[svp]
    s = decomposition["singular_values"][:decomposition["U"].shape[1]]
    return decomposition["U"] * s ** genotype_exponent, decomposition["V"] * s ** environment_exponent


def convex_hull_vertices(points):
    """Hull vertices in counter-clockwise order (the two extremes when all points lie on a line)."""
    points = np.asarray(points, dtype=float)
    try:
        return [int(v) for v in ConvexHull(points).vertices]
    except Exception:
        direction = points[np.argmax(np.linalg.norm(points, axis=1))]
        t = points @ direction
        return sorted({int(np.argmin(t)), int(np.argmax(t))})


def which_won_where(genotypes, environments):
    """Polygon view: hull vertices, the rays from the origin perpendicular to each hull edge (sector boundaries),
    and the winning genotype (highest value predicted by the two components) in each environment."""
    G = np.asarray(genotypes, dtype=float)
    E = np.asarray(environments, dtype=float)
    predicted = G @ E.T
    vertices = convex_hull_vertices(G)
    rays = []
    if len(vertices) >= 3:
        for i in range(len(vertices)):
            a, b = G[vertices[i]], G[vertices[(i + 1) % len(vertices)]]
            edge = b - a
            normal = np.array([edge[1], -edge[0]])
            if normal @ a < 0:
                normal = -normal
            length = np.linalg.norm(normal)
            if length > 0:
                rays.append(normal / length)
    return {"vertices": vertices, "rays": [ray.tolist() for ray in rays],
            "winners": [int(w) for w in predicted.argmax(axis=0)], "predicted": predicted}


def average_environment_axis(genotypes, environments):
    """Average-environment coordination (AEC): mean performance and stability of genotypes, representativeness and
    discriminating power of environments, and the ideal genotype and environment."""
    G = np.asarray(genotypes, dtype=float)
    E = np.asarray(environments, dtype=float)
    mean_environment = E.mean(axis=0)
    length = float(np.linalg.norm(mean_environment))
    unit = mean_environment / length if length > 0 else np.array([1.0, 0.0])
    perpendicular = np.array([-unit[1], unit[0]])
    genotype_length = np.linalg.norm(G, axis=1)
    environment_length = np.linalg.norm(E, axis=1)
    ideal_genotype = unit * genotype_length.max()
    ideal_environment = unit * environment_length.max()
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = np.where(environment_length > 0, (E @ unit) / environment_length, 0.0)
    return {
        "mean_environment": mean_environment,
        "unit": unit,
        "projection": G @ unit,
        "instability": np.abs(G @ perpendicular),
        "ideal_genotype": ideal_genotype,
        "genotype_distance": np.linalg.norm(G - ideal_genotype, axis=1),
        "environment_length": environment_length,
        "representativeness": cosine,
        "angle": np.degrees(np.arccos(np.clip(cosine, -1, 1))),
        "ideal_environment": ideal_environment,
        "environment_distance": np.linalg.norm(E - ideal_environment, axis=1),
    }


def environment_correlations(means):
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = np.corrcoef(np.asarray(means, dtype=float), rowvar=False)
    return [[float(v) if np.isfinite(v) else None for v in row] for row in np.atleast_2d(matrix)]
