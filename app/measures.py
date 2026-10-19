"""Effect sizes, tests and multivariate summaries shared by the analysis endpoints.

Every function is validated against scipy / statsmodels / sklearn or published
textbook values in tests/test_measures_reference.py.
"""
import math

import numpy as np
from scipy import stats


def _num(value):
    """float, or None when the value is missing, infinite or not numeric."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


# ---------------------------------------------------------------------------
# Two-group comparisons
# ---------------------------------------------------------------------------
def two_sample_t(x, y, alpha=0.05):
    """Student's (pooled) and Welch's t-tests with CIs, Cohen's d and Hedges' g."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n1, n2 = len(x), len(y)
    v1, v2 = x.var(ddof=1), y.var(ddof=1)
    diff = x.mean() - y.mean()

    df_student = n1 + n2 - 2
    pooled_var = ((n1 - 1) * v1 + (n2 - 1) * v2) / df_student
    se_student = math.sqrt(pooled_var * (1.0 / n1 + 1.0 / n2))
    se_welch = math.sqrt(v1 / n1 + v2 / n2)
    welch_den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    df_welch = (v1 / n1 + v2 / n2) ** 2 / welch_den if welch_den > 0 else float("nan")

    def result(se, df):
        if not (se > 0 and df > 0):
            return {"t_statistic": None, "degrees_of_freedom": _num(df), "p_value": None,
                    "se_difference": _num(se), "ci_difference": [None, None]}
        t_value = diff / se
        half_width = stats.t.ppf(1 - alpha / 2, df) * se
        return {"t_statistic": float(t_value), "degrees_of_freedom": float(df),
                "p_value": float(2 * stats.t.sf(abs(t_value), df)), "se_difference": float(se),
                "ci_difference": [float(diff - half_width), float(diff + half_width)]}

    cohens_d = diff / math.sqrt(pooled_var) if pooled_var > 0 else None
    hedges_g = cohens_d * (1 - 3 / (4 * df_student - 1)) if cohens_d is not None else None
    return {"mean_difference": float(diff), "student": result(se_student, df_student),
            "welch": result(se_welch, df_welch), "cohens_d": _num(cohens_d), "hedges_g": _num(hedges_g)}


def rank_biserial_independent(u_statistic, n1, n2):
    """P(X > Y) - P(X < Y) from the Mann-Whitney U of the first sample (Kerby 2014)."""
    return float(2.0 * u_statistic / (n1 * n2) - 1.0)


def rank_biserial_paired(differences):
    """Matched-pairs rank-biserial correlation (W+ - W-) / (W+ + W-); zero differences are dropped."""
    d = np.asarray(differences, dtype=float)
    nonzero = d[d != 0]
    if len(nonzero) == 0:
        return {"rank_biserial": None, "w_plus": 0.0, "w_minus": 0.0, "n_nonzero": 0}
    ranks = stats.rankdata(np.abs(nonzero))
    w_plus, w_minus = float(ranks[nonzero > 0].sum()), float(ranks[nonzero < 0].sum())
    return {"rank_biserial": (w_plus - w_minus) / (w_plus + w_minus), "w_plus": w_plus,
            "w_minus": w_minus, "n_nonzero": int(len(nonzero))}


def chi_square_independence(observed):
    """Pearson chi-square (no continuity correction), Cramer's V, expected counts; Yates and Fisher for 2x2."""
    observed = np.asarray(observed, dtype=float)
    chi2, p_value, dof, expected = stats.chi2_contingency(observed, correction=False)
    n = observed.sum()
    k = min(observed.shape) - 1
    result = {
        "statistic": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "cramers_v": float(math.sqrt(chi2 / (n * k))) if n * k > 0 else None,
        "expected": expected,
        "min_expected": float(expected.min()),
        "pct_expected_below_5": float(100.0 * np.mean(expected < 5)),
    }
    if observed.shape == (2, 2):
        chi2_yates, p_yates, _, _ = stats.chi2_contingency(observed, correction=True)
        result["yates"] = {"statistic": float(chi2_yates), "p_value": float(p_yates)}
        result["fisher_exact_p"] = float(stats.fisher_exact(observed)[1])
    return result


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------
def correlation_tests(frame, method="pearson"):
    """Pairwise-complete correlation, two-sided p-value and n for every pair of columns."""
    columns = list(frame.columns)
    k = len(columns)
    r = np.full((k, k), np.nan)
    p = np.full((k, k), np.nan)
    n = np.zeros((k, k), dtype=int)
    test = stats.pearsonr if method == "pearson" else stats.spearmanr
    for i in range(k):
        n[i, i] = int(frame[columns[i]].notna().sum())
        r[i, i] = 1.0
        for j in range(i + 1, k):
            pair = frame[[columns[i], columns[j]]].dropna()
            n[i, j] = n[j, i] = len(pair)
            a, b = pair.iloc[:, 0].to_numpy(dtype=float), pair.iloc[:, 1].to_numpy(dtype=float)
            if len(pair) >= 3 and a.std() > 0 and b.std() > 0:
                res = test(a, b)
                r[i, j] = r[j, i] = float(res.statistic)
                p[i, j] = p[j, i] = float(res.pvalue)
    return r, p, n


# ---------------------------------------------------------------------------
# Multivariate
# ---------------------------------------------------------------------------
def pca_decomposition(x, scale=True):
    """PCA by eigen-decomposition of the correlation (scale=True) or covariance matrix.

    Uses the sample (n - 1) covariance, so eigenvalues of standardized data are the
    eigenvalues of the correlation matrix and sum to the number of variables (the
    basis of the Kaiser criterion). Each eigenvector is oriented so that its largest
    absolute element is positive.
    """
    x = np.asarray(x, dtype=float)
    n, p = x.shape
    centered = x - x.mean(axis=0)
    sd = x.std(axis=0, ddof=1)
    if np.any(sd <= 0):
        raise ValueError("PCA variables must not be constant.")
    z = centered / sd if scale else centered
    matrix = z.T @ z / (n - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = np.clip(eigenvalues[order], 0.0, None), eigenvectors[:, order]
    for a in range(p):
        if eigenvectors[np.argmax(np.abs(eigenvectors[:, a])), a] < 0:
            eigenvectors[:, a] = -eigenvectors[:, a]
    n_components = min(n - 1, p)
    eigenvalues_kept, vectors_kept = eigenvalues[:n_components], eigenvectors[:, :n_components]
    return {
        "eigenvalues": eigenvalues_kept,
        "eigenvectors": vectors_kept,  # columns = components
        "explained_variance_ratio": eigenvalues_kept / eigenvalues.sum(),
        "scores": z @ vectors_kept,
        # correlation between each variable (rows) and each component (columns)
        "correlation_loadings": vectors_kept * np.sqrt(eigenvalues_kept) / np.sqrt(np.diag(matrix))[:, None],
    }


def pls_summary(pls, x, y):
    """Share of standardized X and Y variance explained per PLS component, and VIP scores.

    For a fitted sklearn PLSRegression (scale=True). VIP follows Wold et al. (2001):
    VIP_j = sqrt(p * sum_a SSY_a (w_ja / ||w_a||)^2 / sum_a SSY_a), so mean(VIP^2) = 1.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float).reshape(len(x), -1)
    xs = (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)
    ys = (y - y.mean(axis=0)) / y.std(axis=0, ddof=1)
    t, p_load, q_load, w = pls.x_scores_, pls.x_loadings_, pls.y_loadings_, pls.x_weights_
    score_ss = np.sum(t ** 2, axis=0)
    ss_x = score_ss * np.sum(p_load ** 2, axis=0)
    ss_y = score_ss * np.sum(q_load ** 2, axis=0)
    w_normalized = w / np.linalg.norm(w, axis=0)
    vip = np.sqrt(x.shape[1] * (w_normalized ** 2 @ ss_y) / ss_y.sum())
    return {
        "explained_variance_x": [float(v) for v in ss_x / np.sum(xs ** 2)],
        "explained_variance_y": [float(v) for v in ss_y / np.sum(ys ** 2)],
        "vip": [float(v) for v in vip],
    }


def cluster_quality(x_scaled, linkage_matrix, kmeans_labels, hierarchical_labels, kmeans_inertia, distances=None,
                    metric="euclidean"):
    """Cophenetic correlation of the tree, K-means inertia and silhouette widths.

    distances: the condensed distance vector the tree was built from (default: Euclidean); metric: its name, used for
    the silhouette width of the hierarchical clusters.
    """
    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import pdist
    from sklearn.metrics import silhouette_score

    def silhouette(labels, metric="euclidean"):
        k = len(set(labels))
        return float(silhouette_score(x_scaled, labels, metric=metric)) if 2 <= k <= len(labels) - 1 else None

    cophenetic, _ = cophenet(linkage_matrix, distances if distances is not None else pdist(x_scaled))
    return {
        "cophenetic_correlation": _num(cophenetic),
        "kmeans_inertia": _num(kmeans_inertia),
        "kmeans_silhouette": silhouette(kmeans_labels),
        "hierarchical_silhouette": silhouette(hierarchical_labels, metric),
    }


# ---------------------------------------------------------------------------
# Distribution shape
# ---------------------------------------------------------------------------
def skewness_kurtosis_se(n):
    """Standard errors of sample skewness and excess kurtosis (as reported by SPSS)."""
    se_skew = math.sqrt(6.0 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3))) if n > 2 else None
    se_kurt = math.sqrt(4.0 * (n ** 2 - 1) * se_skew ** 2 / ((n - 3) * (n + 5))) if n > 3 else None
    return se_skew, se_kurt


def distribution_extras(values):
    """Percentiles, SEs of skewness/kurtosis, Lilliefors and D'Agostino-Pearson tests, |z| > 3 outliers."""
    from statsmodels.stats.diagnostic import lilliefors

    x = np.asarray(values, dtype=float)
    n = len(x)
    se_skew, se_kurt = skewness_kurtosis_se(n)
    extras = {
        "percentiles": {f"p{q}": float(np.percentile(x, q)) for q in (10, 25, 50, 75, 90)},
        "se_skewness": se_skew,
        "se_kurtosis": se_kurt,
        "lilliefors": None,
        "dagostino_pearson": None,
        "z_outliers": [],
    }
    sd = x.std(ddof=1) if n > 1 else 0.0
    if n >= 5 and sd > 0:
        stat, p_value = lilliefors(x, dist="norm")
        extras["lilliefors"] = {"stat": float(stat), "p_value": float(p_value), "normal": bool(p_value >= 0.05)}
    if n >= 8 and sd > 0:
        stat, p_value = stats.normaltest(x)
        extras["dagostino_pearson"] = {"stat": float(stat), "p_value": float(p_value), "normal": bool(p_value >= 0.05)}
    if sd > 0:
        z = (x - x.mean()) / sd
        extras["z_outliers"] = [{"position": int(i), "val": float(x[i]), "z": float(z[i])} for i in np.where(np.abs(z) > 3)[0]]
    return extras


def shape_summary(values):
    """Skewness, excess kurtosis and Shapiro-Wilk p-value, for before/after comparisons."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    varies = n > 1 and x.std() > 0
    return {
        "n": int(n),
        "skewness": _num(stats.skew(x, bias=False)) if n > 2 and varies else None,
        "kurtosis": _num(stats.kurtosis(x, bias=False)) if n > 3 and varies else None,
        "shapiro_p": _num(stats.shapiro(x)[1]) if 3 <= n <= 5000 and varies else None,
    }
