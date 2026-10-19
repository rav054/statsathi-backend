"""Trend, change-point and homogeneity statistics for time series (climate, hydrology, long-term field data).

- Mann-Kendall test, modified Mann-Kendall for autocorrelated series (Hamed & Rao 1998), seasonal Mann-Kendall
  (Hirsch et al. 1982)
- Sen's slope with its confidence interval (Sen 1968; Gilbert 1987) and the ordinary least-squares trend
- Change-point / homogeneity tests: Pettitt (1979), SNHT (Alexandersson 1986), Buishand range (1982) and the
  von Neumann ratio, with Wijngaard et al. (2003) classification
- Innovative Trend Analysis (Şen 2012, 2014)
"""

import math

import numpy as np
import scipy.stats as stats
from statsmodels.tsa.stattools import acf

MONTE_CARLO_SEED = 20260912
MIN_OBSERVATIONS = 8
MAX_OBSERVATIONS = 2000


def trend_label(p_value, statistic, alpha):
    if p_value is None or np.isnan(p_value) or p_value >= alpha:
        return "no significant trend"
    return "increasing" if statistic > 0 else "decreasing"


# ---------------------------------------------------------------------------
# Mann-Kendall family
# ---------------------------------------------------------------------------
def mk_score(x):
    """Mann-Kendall S = Σ_{i<j} sign(x_j − x_i) for a series in time order."""
    x = np.asarray(x, dtype=float)
    return int(sum(np.sign(x[i + 1:] - x[i]).sum() for i in range(x.size - 1)))


def mk_variance(x):
    """Variance of S under the null hypothesis, corrected for tied values."""
    x = np.asarray(x, dtype=float)
    n = x.size
    _, counts = np.unique(x, return_counts=True)
    ties = counts[counts > 1].astype(float)
    return float((n * (n - 1) * (2 * n + 1) - np.sum(ties * (ties - 1) * (2 * ties + 1))) / 18)


def mk_z(s, variance):
    """Standard normal statistic with the usual continuity correction."""
    if variance <= 0:
        return 0.0
    if s > 0:
        return (s - 1) / math.sqrt(variance)
    if s < 0:
        return (s + 1) / math.sqrt(variance)
    return 0.0


def mann_kendall(x, alpha=0.05):
    x = np.asarray(x, dtype=float)
    n = x.size
    s = mk_score(x)
    variance = mk_variance(x)
    z = mk_z(s, variance)
    p = float(2 * stats.norm.sf(abs(z)))
    return {"n": int(n), "s": s, "var_s": variance, "z": float(z), "p_value": p,
            "tau": float(s / (n * (n - 1) / 2)), "trend": trend_label(p, s, alpha)}


def modified_mann_kendall(x, alpha=0.05):
    """Hamed & Rao (1998): the variance of S inflated for significant autocorrelation of the detrended ranks."""
    x = np.asarray(x, dtype=float)
    n = x.size
    base = mann_kendall(x, alpha)
    index = np.arange(1, n + 1)
    slope = stats.theilslopes(x, index).slope
    ranks = stats.rankdata(x - slope * index)
    r = acf(ranks, nlags=n - 1, fft=False)
    bound = stats.norm.ppf(1 - alpha / 2) / math.sqrt(n)
    total = sum((n - k) * (n - k - 1) * (n - k - 2) * r[k] for k in range(1, n - 1) if abs(r[k]) > bound)
    correction = 1 + 2 / (n * (n - 1) * (n - 2)) * total
    variance = base["var_s"] * correction
    z = mk_z(base["s"], variance)
    p = float(2 * stats.norm.sf(abs(z))) if variance > 0 else float("nan")
    return {"s": base["s"], "var_s": float(variance), "variance_correction": float(correction), "z": float(z),
            "p_value": p, "trend": trend_label(p, base["s"], alpha)}


def seasonal_mann_kendall(t, x, seasons, alpha=0.05):
    """Hirsch et al. (1982): S and its variance summed over seasons; slope = median of within-season slopes."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    seasons = np.asarray(seasons)
    s_total, variance_total, slopes, used = 0, 0.0, [], 0
    for season in np.unique(seasons):
        mask = seasons == season
        xs, ts = x[mask], t[mask]
        if xs.size < 2:
            continue
        used += 1
        s_total += mk_score(xs)
        variance_total += mk_variance(xs)
        for i in range(xs.size - 1):
            dt = ts[i + 1:] - ts[i]
            valid = dt != 0
            slopes.extend(((xs[i + 1:] - xs[i])[valid] / dt[valid]).tolist())
    z = mk_z(s_total, variance_total)
    p = float(2 * stats.norm.sf(abs(z)))
    return {"seasons": int(used), "s": int(s_total), "var_s": float(variance_total), "z": float(z), "p_value": p,
            "slope": float(np.median(slopes)) if slopes else None, "trend": trend_label(p, s_total, alpha)}


# ---------------------------------------------------------------------------
# Magnitude of the trend
# ---------------------------------------------------------------------------
def sens_slope(t, x, alpha=0.05):
    result = stats.theilslopes(x, t, alpha=1 - alpha)
    return {"slope": float(result.slope), "intercept": float(result.intercept),
            "ci_low": float(result.low_slope), "ci_high": float(result.high_slope)}


def linear_trend(t, x, alpha=0.05):
    t = np.asarray(t, dtype=float)
    result = stats.linregress(t, x)
    half = stats.t.ppf(1 - alpha / 2, t.size - 2) * result.stderr
    return {"slope": float(result.slope), "intercept": float(result.intercept), "r2": float(result.rvalue ** 2),
            "p_value": float(result.pvalue), "ci_low": float(result.slope - half), "ci_high": float(result.slope + half)}


def lag1_autocorrelation(x, alpha=0.05):
    x = np.asarray(x, dtype=float)
    r1 = float(acf(x, nlags=1, fft=False)[1])
    bound = float(stats.norm.ppf(1 - alpha / 2) / math.sqrt(x.size))
    return {"r1": r1, "bound": bound, "significant": bool(abs(r1) > bound)}


# ---------------------------------------------------------------------------
# Change point and homogeneity
# ---------------------------------------------------------------------------
def pettitt(x):
    """Pettitt test: U_t = Σ_{i≤t} Σ_{j>t} sign(x_i − x_j); K = max |U_t|; approximate p = 2 exp(−6K² / (n³ + n²))."""
    x = np.asarray(x, dtype=float)
    n = x.size
    v = np.sign(x[:, None] - x[None, :]).sum(axis=1)
    u = np.cumsum(v)[:-1]
    change = int(np.argmax(np.abs(u)))
    k = float(abs(u[change]))
    p = min(1.0, 2 * math.exp(-6 * k ** 2 / (n ** 3 + n ** 2)))
    return {"k": k, "change_index": change, "p_value": float(p), "u": u.astype(float).tolist(),
            "mean_before": float(x[:change + 1].mean()), "mean_after": float(x[change + 1:].mean())}


def snht_series(data):
    """SNHT T_k for every split point (rows of `data` are series): T_k = k·z̄₁² + (n−k)·z̄₂²."""
    data = np.atleast_2d(np.asarray(data, dtype=float))
    n = data.shape[1]
    z = (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, ddof=1, keepdims=True)
    cumulative = np.cumsum(z, axis=1)[:, :-1]
    k = np.arange(1, n)
    return cumulative ** 2 * (1 / k + 1 / (n - k))


def buishand_series(data):
    """Buishand cumulative deviations S_k / D (rows of `data` are series), D = population standard deviation."""
    data = np.atleast_2d(np.asarray(data, dtype=float))
    deviations = data - data.mean(axis=1, keepdims=True)
    scale = np.sqrt((deviations ** 2).mean(axis=1, keepdims=True))
    return np.cumsum(deviations, axis=1) / scale


def von_neumann_ratio(data):
    data = np.atleast_2d(np.asarray(data, dtype=float))
    deviations = data - data.mean(axis=1, keepdims=True)
    return (np.diff(data, axis=1) ** 2).sum(axis=1) / (deviations ** 2).sum(axis=1)


def homogeneity_tests(x, alpha=0.05, simulations=None, seed=MONTE_CARLO_SEED):
    """SNHT, Buishand Q and R, and von Neumann ratio with Monte Carlo p-values from normal series of the same length."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if simulations is None:
        simulations = 10000 if n <= 100 else 4000 if n <= 400 else 1000
    draws = np.random.default_rng(seed).standard_normal((simulations, n))
    root_n = math.sqrt(n)

    t_series = snht_series(x)[0]
    t0 = float(t_series.max())
    snht_sim = snht_series(draws).max(axis=1)

    s_series = buishand_series(x)[0]
    q = float(np.abs(s_series).max() / root_n)
    r = float((max(s_series.max(), 0) - min(s_series.min(), 0)) / root_n)
    s_sim = buishand_series(draws)
    q_sim = np.abs(s_sim).max(axis=1) / root_n
    r_sim = (np.maximum(s_sim.max(axis=1), 0) - np.minimum(s_sim.min(axis=1), 0)) / root_n

    ratio = float(von_neumann_ratio(x)[0])
    ratio_sim = von_neumann_ratio(draws)

    def upper(sim, observed):
        return float((1 + np.sum(sim >= observed)) / (simulations + 1))

    return {
        "simulations": int(simulations),
        "snht": {"t0": t0, "change_index": int(np.argmax(t_series)), "p_value": upper(snht_sim, t0),
                 "series": t_series.tolist()},
        "buishand": {"q": q, "r": r, "change_index": int(np.argmax(np.abs(s_series))),
                     "p_value_q": upper(q_sim, q), "p_value_r": upper(r_sim, r), "series": s_series.tolist()},
        "von_neumann": {"ratio": ratio, "p_value": float((1 + np.sum(ratio_sim <= ratio)) / (simulations + 1))},
    }


def homogeneity_class(p_values, alpha=0.05):
    """Wijngaard et al. (2003): useful (0–1 tests reject), doubtful (2), suspect (3–4)."""
    rejected = sum(1 for p in p_values if p is not None and p < alpha)
    label = "useful" if rejected <= 1 else "doubtful" if rejected == 2 else "suspect"
    return {"class": label, "tests_rejecting": rejected, "tests": len(p_values)}


# ---------------------------------------------------------------------------
# Innovative Trend Analysis
# ---------------------------------------------------------------------------
def innovative_trend_analysis(x):
    """Şen's ITA: sorted first half against sorted second half (the middle value is dropped for odd n)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    m = n // 2
    first = np.sort(x[:m])
    second = np.sort(x[n - m:])
    mean_first, mean_second = float(first.mean()), float(second.mean())
    ranges = []
    for name, idx in zip(("low", "medium", "high"), np.array_split(np.arange(m), 3)):
        a, b = float(first[idx].mean()), float(second[idx].mean())
        ranges.append({"range": name, "mean_first": a, "mean_second": b,
                       "change_percent": float(100 * (b - a) / abs(a)) if a != 0 else None})
    return {
        "first_half": first.tolist(),
        "second_half": second.tolist(),
        "slope": float(2 * (mean_second - mean_first) / n),  # per time step (Şen 2017)
        "indicator": float(10 * np.mean(second - first) / mean_first) if mean_first != 0 else None,  # Şen (2014)
        "mean_first": mean_first,
        "mean_second": mean_second,
        "ranges": ranges,
    }


# ---------------------------------------------------------------------------
# Everything for one series
# ---------------------------------------------------------------------------
def analyse_series(t, x, alpha=0.05, seasons=None):
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    pettitt_result = pettitt(x)
    homogeneity = homogeneity_tests(x, alpha)
    sen = sens_slope(t, x, alpha)
    span = float(t.max() - t.min())
    mean = float(x.mean())
    result = {
        "n": int(x.size),
        "time_start": float(t.min()),
        "time_end": float(t.max()),
        "mean": mean,
        "mann_kendall": mann_kendall(x, alpha),
        "modified_mann_kendall": modified_mann_kendall(x, alpha),
        "seasonal_mann_kendall": seasonal_mann_kendall(t, x, seasons, alpha) if seasons is not None else None,
        "sen": {**sen, "change_over_period": sen["slope"] * span,
                "percent_of_mean": float(100 * sen["slope"] * span / abs(mean)) if mean != 0 else None},
        "linear": linear_trend(t, x, alpha),
        "autocorrelation": lag1_autocorrelation(x, alpha),
        "pettitt": pettitt_result,
        "snht": homogeneity["snht"],
        "buishand": homogeneity["buishand"],
        "von_neumann": homogeneity["von_neumann"],
        "simulations": homogeneity["simulations"],
        "ita": innovative_trend_analysis(x),
    }
    result["homogeneity"] = homogeneity_class([
        pettitt_result["p_value"], homogeneity["snht"]["p_value"],
        homogeneity["buishand"]["p_value_r"], homogeneity["von_neumann"]["p_value"],
    ], alpha)
    return result
