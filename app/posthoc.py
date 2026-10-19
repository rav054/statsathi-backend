"""Post-hoc multiple comparisons and compact letter displays (CLD).

Every method works from the same inputs: the group means and, for each pair of
groups, the standard error of their difference with its error degrees of
freedom. That one interface covers unbalanced data (Tukey-Kramer), unequal
variances (Games-Howell) and designs with more than one error term (split-plot).

References
- Tukey-Kramer / Games-Howell: studentized range on |t| * sqrt(2).
- Duncan's MRT: protection level (1 - alpha)^(p - 1) for a range spanning p
  ordered means; p-values follow the R package agricolae (duncan.test).
- Letters: insert-and-absorb algorithm, Piepho (2004), J. Comput. Graph. Stat.
  13(2):456-466. Two groups share a letter if and only if they do not differ.
"""
from functools import lru_cache
from itertools import combinations
from math import sqrt

import numpy as np
from scipy import special, stats
from scipy.interpolate import CubicSpline

METHOD_LABELS = {
    "tukey": "Tukey's HSD (Tukey-Kramer)",
    "lsd": "Fisher's LSD (Critical Difference)",
    "duncan": "Duncan's Multiple Range Test (DMRT)",
    "bonferroni": "Bonferroni-corrected t-tests",
    "games_howell": "Games-Howell (unequal variances)",
}


class PosthocError(ValueError):
    """Raised when comparisons cannot be computed from the supplied inputs."""


# ---------------------------------------------------------------------------
# Studentized range distribution
# ---------------------------------------------------------------------------
# scipy.stats.studentized_range is accurate but takes ~20 ms per value, far too
# slow for the hundreds of pairs a factorial experiment produces. This vectorized
# version integrates the same density and agrees with scipy to better than 1e-8
# (tests/test_posthoc.py).
#
#   CDF(q; k, df) = integral_0^inf P_k(q s) f(s) ds,  s = sqrt(chi2_df / df)
#   P_k(w) = k * integral phi(z) [Phi(z) - Phi(z - w)]^(k - 1) dz
#
# P_k is tabulated once per k (Simpson's rule, cubic spline in w); the outer
# integral uses the trapezoid rule in log(s), which converges very fast here.
_Z = np.linspace(-9.0, 9.0, 721)
_Z_WEIGHTS = np.full(_Z.size, 2.0)
_Z_WEIGHTS[1::2] = 4.0
_Z_WEIGHTS[[0, -1]] = 1.0
_Z_WEIGHTS *= (_Z[1] - _Z[0]) / 3.0
_PHI_WEIGHTED = stats.norm.pdf(_Z) * _Z_WEIGHTS
_W_MAX = 16.0  # P_k(w) is 1 to machine precision beyond this for any practical k
_W = np.linspace(0.0, _W_MAX, 1601)
with np.errstate(divide="ignore"):
    _LOG_BAND = np.log(np.clip(stats.norm.cdf(_Z)[None, :] - stats.norm.cdf(_Z[None, :] - _W[:, None]), 0.0, 1.0))
_T_FRACTIONS = np.linspace(0.0, 1.0, 160)
_MAX_DF = 1e7


@lru_cache(maxsize=256)
def _range_cdf(k):
    values = np.clip(k * (np.exp((k - 1) * _LOG_BAND) @ _PHI_WEIGHTED), 0.0, 1.0)
    values[0] = 0.0
    return CubicSpline(_W, values)


def studentized_range_cdf(q, k, df):
    """Vectorized CDF of the studentized range; q, k and df broadcast together."""
    q, k, df = np.broadcast_arrays(
        np.asarray(q, dtype=float), np.asarray(k, dtype=int), np.minimum(np.asarray(df, dtype=float), _MAX_DF)
    )
    out = np.zeros(q.shape)
    for k_value in np.unique(k):
        mask = k == k_value
        spline = _range_cdf(int(k_value))
        qq, dd = q[mask], df[mask][:, None]
        # t = log(s) over the chi quantiles holding all but 2e-15 of the probability mass
        lo = np.log(stats.chi.ppf(1e-15, dd) / np.sqrt(dd))
        hi = np.log(stats.chi.isf(1e-15, dd) / np.sqrt(dd))
        t = lo + (hi - lo) * _T_FRACTIONS
        log_density = (
            (dd / 2) * np.log(dd) + dd * t - dd * np.exp(2 * t) / 2
            - (dd / 2 - 1) * np.log(2.0) - special.gammaln(dd / 2)
        )
        integrand = np.exp(log_density) * spline(np.minimum(qq[:, None] * np.exp(t), _W_MAX))
        step = (hi - lo)[:, 0] / (_T_FRACTIONS.size - 1)
        out[mask] = (integrand.sum(axis=1) - 0.5 * (integrand[:, 0] + integrand[:, -1])) * step
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Standard errors of differences
# ---------------------------------------------------------------------------
def pooled_pair_error(ns, mse, df_error):
    """SE of a difference from a pooled error mean square (exact for unequal n)."""
    def provider(i, j):
        return sqrt(mse * (1.0 / ns[i] + 1.0 / ns[j])), df_error
    return provider


def welch_pair_error(ns, variances):
    """Per-pair Welch SE and Welch-Satterthwaite df, as used by Games-Howell."""
    def provider(i, j):
        if ns[i] < 2 or ns[j] < 2:
            raise PosthocError("Games-Howell needs at least 2 observations in every group.")
        vi, vj = variances[i] / ns[i], variances[j] / ns[j]
        denom = vi ** 2 / (ns[i] - 1) + vj ** 2 / (ns[j] - 1)
        if denom <= 0:
            raise PosthocError("Games-Howell cannot be computed when groups have zero variance.")
        return sqrt(vi + vj), (vi + vj) ** 2 / denom
    return provider


def splitplot_pair_error(labels, main_level, ms_a, df_a, ms_b, df_b, b_levels, reps):
    """SE for comparing split-plot A x B cell means (Gomez & Gomez 1984, ch. 3).

    Same main-plot level:       SE = sqrt(2 Eb / r),                 Error(b) df.
    Different main-plot levels: SE = sqrt(2 [(b - 1) Eb + Ea] / (r b)),
                                Satterthwaite df for the combined error.
    """
    combined = (b_levels - 1) * ms_b + ms_a
    if ms_b <= 0 or combined <= 0 or df_a <= 0 or df_b <= 0 or reps <= 0:
        raise PosthocError("Split-plot comparisons need positive Error(a) and Error(b) mean squares.")
    df_combined = combined ** 2 / (((b_levels - 1) * ms_b) ** 2 / df_b + ms_a ** 2 / df_a)
    se_same = sqrt(2.0 * ms_b / reps)
    se_across = sqrt(2.0 * combined / (reps * b_levels))

    def provider(i, j):
        if main_level[labels[i]] == main_level[labels[j]]:
            return se_same, df_b
        return se_across, df_combined
    return provider


# ---------------------------------------------------------------------------
# Comparisons and letters
# ---------------------------------------------------------------------------
def pairwise_comparisons(labels, means, pair_error, method="tukey", alpha=0.05):
    """Compare every pair of groups.

    pair_error(i, j) must return (se_of_difference, error_df) for groups i and j.
    mean_diff is reported as mean(group2) - mean(group1).
    """
    if method not in METHOD_LABELS:
        raise PosthocError(f"Unknown post-hoc method '{method}'.")
    k = len(labels)
    if k < 2:
        return []

    means = [float(m) for m in means]
    order = np.argsort(-np.asarray(means), kind="stable")
    rank = np.empty(k, dtype=int)
    rank[order] = np.arange(k)

    pairs = list(combinations(range(k), 2))
    diffs = np.array([means[j] - means[i] for i, j in pairs])
    errors = [pair_error(i, j) for i, j in pairs]
    se = np.array([e[0] for e in errors], dtype=float)
    df = np.array([e[1] for e in errors], dtype=float)
    if not (np.all(np.isfinite(se)) and np.all(se > 0) and np.all(np.isfinite(df)) and np.all(df > 0)):
        raise PosthocError("Post-hoc comparisons need a positive error variance and error degrees of freedom.")

    t_abs = np.abs(diffs) / se
    if method == "lsd":
        p = 2.0 * stats.t.sf(t_abs, df)
    elif method == "bonferroni":
        p = 2.0 * stats.t.sf(t_abs, df) * len(pairs)
    elif method in ("tukey", "games_howell"):
        p = 1.0 - studentized_range_cdf(t_abs * sqrt(2.0), k, df)
    else:  # duncan
        spans = np.array([abs(rank[i] - rank[j]) + 1 for i, j in pairs])
        cdf = studentized_range_cdf(t_abs * sqrt(2.0), spans, df)
        p = 1.0 - cdf ** (1.0 / (spans - 1))
    p = np.clip(p, 0.0, 1.0)

    return [
        {
            "group1": labels[i],
            "group2": labels[j],
            "mean_diff": float(diffs[n]),
            "se_diff": float(se[n]),
            "df": float(df[n]),
            "p_value": float(p[n]),
            "significant": bool(p[n] < alpha),
        }
        for n, (i, j) in enumerate(pairs)
    ]


def _letter(index):
    """0 -> a, 25 -> z, 26 -> aa, ..."""
    s = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        s = chr(97 + rem) + s
    return s


def _absorb(columns):
    unique = []
    for col in columns:
        if col not in unique:
            unique.append(col)
    return [c for c in unique if not any(c < other for other in unique)]


def compact_letter_groups(labels, means, comparisons):
    """Return {label: [letter, ...]} using insert-and-absorb ("a" = highest means)."""
    k = len(labels)
    index = {label: n for n, label in enumerate(labels)}
    columns = [frozenset(range(k))]

    for comp in comparisons:
        if not comp["significant"]:
            continue
        i, j = index[comp["group1"]], index[comp["group2"]]
        split = []
        for col in columns:
            if i in col and j in col:
                split.append(col - {i})
                split.append(col - {j})
            else:
                split.append(col)
        columns = _absorb(split)

    position = {n: p for p, n in enumerate(sorted(range(k), key=lambda n: -means[n]))}
    columns.sort(key=lambda col: min(position[n] for n in col))

    groups = {label: [] for label in labels}
    for c_idx, col in enumerate(columns):
        for n in col:
            groups[labels[n]].append(_letter(c_idx))
    return groups


def compact_letter_display(labels, means, comparisons):
    """Return {label: "ab"}-style strings for tables and charts."""
    return {label: "".join(letters) for label, letters in compact_letter_groups(labels, means, comparisons).items()}
