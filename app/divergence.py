"""Genetic divergence among genotypes: Mahalanobis D² (Rao 1952) and Tocher's clustering.

- Error sums of squares and products (SSCP) from a replicated trial (genotypes, optional replicate blocks) and Wilks'
  Λ test of the genotype mean vectors with Rao's F approximation
- D² between genotype means: the means are transformed to uncorrelated variables with unit error variance (Cholesky,
  equivalent to pivotal condensation), so D² is the squared Euclidean distance of the transformed means; the same
  transformation gives each character's contribution to divergence ("times ranked first", Singh 1981)
- Tocher's method of grouping (Rao 1952), average intra- and inter-cluster distances
- Canonical variates: principal axes of the transformed means, whose Euclidean distances reproduce D²
"""

import numpy as np
import scipy.stats as stats
from scipy import linalg


def design_matrix(n, *factors):
    """Intercept plus treatment-coded dummy columns for each factor (integer codes 0 … levels − 1)."""
    columns = [np.ones(n)]
    for codes in factors:
        codes = np.asarray(codes, dtype=int)
        for level in range(1, int(codes.max()) + 1):
            columns.append((codes == level).astype(float))
    return np.column_stack(columns)


def residual_sscp(Y, X):
    """Residual sums of squares and cross-products of a multivariate least-squares fit, and its degrees of freedom."""
    Y = np.asarray(Y, dtype=float)
    coefficients, *_ = np.linalg.lstsq(X, Y, rcond=None)
    residuals = Y - X @ coefficients
    return residuals.T @ residuals, int(Y.shape[0] - np.linalg.matrix_rank(X))


def manova_wilks(Y, genotype, block=None):
    """Error and genotype SSCP matrices and Wilks' Λ for genotypes (adjusted for blocks when given)."""
    Y = np.asarray(Y, dtype=float)
    n, p = Y.shape
    others = [block] if block is not None else []
    error, df_error = residual_sscp(Y, design_matrix(n, genotype, *others))
    reduced, df_reduced = residual_sscp(Y, design_matrix(n, *others))
    hypothesis = reduced - error
    q = df_reduced - df_error

    result = {"error_sscp": error, "hypothesis_sscp": hypothesis, "df_error": df_error, "df_hypothesis": q,
              "lambda": None, "f": None, "df1": None, "df2": None, "p_value": None}
    sign_e, log_e = np.linalg.slogdet(error)
    sign_t, log_t = np.linalg.slogdet(error + hypothesis)
    if sign_e <= 0 or sign_t <= 0 or df_error < p:
        return result
    wilks = float(np.exp(log_e - log_t))
    # Rao's F approximation
    t = np.sqrt((p ** 2 * q ** 2 - 4) / (p ** 2 + q ** 2 - 5)) if p ** 2 + q ** 2 - 5 > 0 else 1.0
    df1 = p * q
    w = df_error + q - (p + q + 1) / 2
    df2 = w * t - (p * q - 2) / 2
    root = wilks ** (1 / t)
    f = (1 - root) / root * df2 / df1
    result.update({"lambda": wilks, "f": float(f), "df1": float(df1), "df2": float(df2),
                   "p_value": float(stats.f.sf(f, df1, df2))})
    return result


def mahalanobis_d2(means, covariance):
    """D² matrix between rows of `means`, and the means transformed to uncorrelated unit-variance variables.

    Raises numpy.linalg.LinAlgError when the covariance matrix is not positive definite.
    """
    means = np.asarray(means, dtype=float)
    lower = np.linalg.cholesky(np.asarray(covariance, dtype=float))
    transformed = linalg.solve_triangular(lower, means.T, lower=True).T
    difference = transformed[:, None, :] - transformed[None, :, :]
    return (difference ** 2).sum(axis=2), transformed


def character_contributions(transformed):
    """How often each character has the largest squared difference, over all pairs of genotypes (Singh 1981)."""
    transformed = np.asarray(transformed, dtype=float)
    i, j = np.triu_indices(transformed.shape[0], 1)
    squared = (transformed[i] - transformed[j]) ** 2
    counts = np.bincount(squared.argmax(axis=1), minlength=transformed.shape[1])
    return counts, 100 * counts / max(len(i), 1)


def tocher(d2):
    """Tocher's method (Rao 1952).

    The threshold θ is the largest of the genotypes' smallest D² values. Each cluster starts with the closest pair of
    unclustered genotypes; the unclustered genotype with the smallest total D² to the members joins while its average
    D² to the members does not exceed θ. Genotypes that cannot join any cluster form single-genotype clusters.
    """
    D = np.asarray(d2, dtype=float)
    G = D.shape[0]
    off_diagonal = D + np.diag(np.full(G, np.inf))
    theta = float(off_diagonal.min(axis=1).max())
    remaining = list(range(G))
    clusters = []
    while remaining:
        if len(remaining) == 1:
            clusters.append(remaining[:])
            break
        sub = off_diagonal[np.ix_(remaining, remaining)]
        a, b = np.unravel_index(int(np.argmin(sub)), sub.shape)
        first, second = remaining[a], remaining[b]
        if D[first, second] > theta:
            clusters.extend([[g] for g in remaining])
            break
        cluster = [first, second]
        remaining = [g for g in remaining if g not in cluster]
        while remaining:
            totals = D[np.ix_(remaining, cluster)].sum(axis=1)
            best = int(np.argmin(totals))
            if totals[best] / len(cluster) > theta:
                break
            cluster.append(remaining.pop(best))
        clusters.append(cluster)
    return clusters, theta


def cluster_distances(d2, clusters):
    """Average D² within each cluster (over pairs of members; 0 for one genotype) and between clusters (diagonal: within)."""
    D = np.asarray(d2, dtype=float)
    K = len(clusters)
    matrix = np.zeros((K, K))
    for a in range(K):
        for b in range(K):
            if a == b:
                members = clusters[a]
                if len(members) > 1:
                    block = D[np.ix_(members, members)]
                    matrix[a, a] = block[np.triu_indices(len(members), 1)].mean()
            else:
                matrix[a, b] = D[np.ix_(clusters[a], clusters[b])].mean()
    return matrix


def canonical_variates(transformed):
    """Scores of genotype means on the principal axes of the transformed (D²) space and the % of divergence per axis."""
    centred = np.asarray(transformed, dtype=float)
    centred = centred - centred.mean(axis=0)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    for k in range(vt.shape[0]):
        if vt[k, np.argmax(np.abs(vt[k]))] < 0:
            vt[k] *= -1
    eigenvalues = s ** 2
    total = eigenvalues.sum()
    return centred @ vt.T, (100 * eigenvalues / total if total > 0 else np.zeros_like(eigenvalues))
