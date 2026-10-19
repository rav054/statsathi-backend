"""Genotype × environment analysis of multi-environment trials: AMMI, GGE biplot and stability statistics."""

from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from .. import gxe as gxe_stats
from ..auth import get_current_user
from ..models import User
from . import chart_gallery, gxe_charts
from .common import read_uploaded_table

router = APIRouter()

MAX_GENOTYPES = 300
MAX_ENVIRONMENTS = 60
SCALINGS = {"none": "no scaling", "sd": "scaled by environment SD"}
SVP_CHOICES = ("auto", "symmetric", "genotype", "environment")
SVP_LABELS = {"symmetric": "symmetric", "genotype": "genotype-focused", "environment": "environment-focused"}
# Partitioning recommended for each view (Yan 2002; Yan & Tinker 2006)
AUTO_SVP = {"gge_biplot": "symmetric", "which_won_where": "symmetric", "mean_stability": "genotype",
            "ranking": "genotype", "discriminativeness": "environment"}


def bad_request(message):
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)


def number(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def levels(series):
    return sorted(series.unique(), key=chart_gallery.natural_key)


def ranks(values, descending):
    return pd.Series(np.asarray(values, dtype=float)).rank(ascending=not descending, method="min").astype(int).tolist()


@router.post("/gxe")
def analyze_gxe(
    file: UploadFile = File(...),
    genotype_col: str = Form(...),
    environment_col: str = Form(...),  # location, year, season or location-year combination
    response_col: str = Form(...),
    rep_col: Optional[str] = Form(None),  # replicate / block within each environment
    scaling: str = Form("none"),
    svp: str = Form("auto"),
    higher_is_better: bool = Form(True),
    palette: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user)
):
    if scaling not in SCALINGS:
        raise bad_request("Scaling must be 'none' or 'sd'.")
    if svp not in SVP_CHOICES:
        raise bad_request("Singular value partitioning must be auto, symmetric, genotype or environment.")
    df = read_uploaded_table(file)
    rep_col = rep_col or None
    roles = [genotype_col, environment_col, response_col] + ([rep_col] if rep_col else [])
    missing = [c for c in roles if c not in df.columns]
    if missing:
        raise bad_request(f"Column(s) not found in dataset: {', '.join(missing)}.")
    if len(set(roles)) != len(roles):
        raise bad_request("The genotype, environment, response and replicate columns must all be different.")

    notes = []
    frame = df[roles].copy()
    response = pd.to_numeric(frame[response_col], errors="coerce")
    if (response.isna() & frame[response_col].notna()).any():
        raise bad_request(f"'{response_col}' contains non-numeric values.")
    frame[response_col] = response
    before = len(frame)
    frame = frame.dropna().reset_index(drop=True)
    if before - len(frame):
        dropped = before - len(frame)
        notes.append(f"{dropped} row{'s' if dropped != 1 else ''} with missing values {'were' if dropped != 1 else 'was'} left out.")
    frame["_g"] = chart_gallery.as_category(frame[genotype_col])
    frame["_e"] = chart_gallery.as_category(frame[environment_col])
    genotypes, environments = levels(frame["_g"]), levels(frame["_e"])
    G, E = len(genotypes), len(environments)
    if G < 3 or E < 3:
        raise bad_request(f"G×E analysis needs at least 3 genotypes and 3 environments (found {G} and {E}).")
    if G > MAX_GENOTYPES or E > MAX_ENVIRONMENTS:
        raise bad_request(f"At most {MAX_GENOTYPES} genotypes and {MAX_ENVIRONMENTS} environments can be analysed at once.")

    counts = pd.crosstab(frame["_g"], frame["_e"]).reindex(index=genotypes, columns=environments, fill_value=0)
    empty = [(g, e) for g in genotypes for e in environments if counts.at[g, e] == 0]
    if empty:
        listed = ", ".join(f"{g} in {e}" for g, e in empty[:5])
        raise bad_request(f"Every genotype must be tested in every environment. Missing: {listed}"
                          f"{' and more' if len(empty) > 5 else ''}.")
    means = (frame.groupby(["_g", "_e"])[response_col].mean().unstack()
             .reindex(index=genotypes, columns=environments).to_numpy(dtype=float))
    if np.ptp(means) == 0:
        raise bad_request(f"'{response_col}' has the same value in every genotype × environment cell.")

    g_index = frame["_g"].map({g: i for i, g in enumerate(genotypes)}).to_numpy()
    e_index = frame["_e"].map({e: j for j, e in enumerate(environments)}).to_numpy()
    y = frame[response_col].to_numpy(dtype=float)
    cell_counts = counts.to_numpy()
    r = int(cell_counts.min())
    balanced = bool((cell_counts == r).all())

    block = None
    if rep_col and balanced and r >= 2:
        frame["_r"] = chart_gallery.as_category(frame[rep_col])
        if frame.duplicated(["_e", "_r", "_g"]).any():
            raise bad_request("A genotype appears more than once in the same replicate of an environment. Check the "
                              "replicate column, or leave it empty.")
        codes = np.zeros(len(frame), dtype=int)
        complete = True
        for rows in frame.groupby("_e").groups.values():
            rep_levels = levels(frame.loc[rows, "_r"])
            complete &= len(rep_levels) == r
            codes[np.asarray(rows)] = frame.loc[rows, "_r"].map({lvl: i for i, lvl in enumerate(rep_levels)}).to_numpy()
        if complete:
            block = codes
        else:
            notes.append("Replicates are not complete blocks in every environment, so the replicate column was not used "
                         "as a block term.")

    if balanced and r >= 2:
        anova = gxe_stats.ammi_anova(y, g_index, e_index, G, E, r, block)
    else:
        anova = gxe_stats.ammi_anova(means.ravel(), np.repeat(np.arange(G), E), np.tile(np.arange(E), G), G, E, 1, None)
        notes.append("There is one value per genotype in each environment, so there is no error term: the ANOVA shows sums "
                     "of squares without F-tests." if balanced else
                     "Genotypes have unequal numbers of replicates, so the analysis uses genotype × environment means and "
                     "F-tests are not available.")

    grand = float(means.mean())
    genotype_means, environment_means = means.mean(axis=1), means.mean(axis=0)
    try:
        stability = gxe_stats.stability_statistics(means, anova["ms_error"], anova["replicates"], anova["df_error"])
    except ValueError as exc:
        raise bad_request(str(exc))
    ipca = anova["ipca"]
    genotype_scores, environment_scores, _ = gxe_stats.ammi_scores(anova["interaction"], 2)
    asv = gxe_stats.ammi_stability_value(genotype_scores, ipca[0]["ss"], ipca[1]["ss"]) if len(ipca) >= 2 else None

    if not higher_is_better:
        notes.append("Lower values are better: the GGE biplot analyses the values multiplied by −1, so its 'best' "
                     "direction points to the lowest values.")
    try:
        decomposition = gxe_stats.gge_decomposition(means if higher_is_better else -means, scaling)
    except ValueError as exc:
        raise bad_request(str(exc))
    coordinates = {}

    def view(name):
        chosen = AUTO_SVP[name] if svp == "auto" else svp
        if chosen not in coordinates:
            coordinates[chosen] = gxe_stats.gge_coordinates(decomposition, chosen)
        return chosen, coordinates[chosen]

    percent = decomposition["percent"]
    basic_svp, (G_basic, E_basic) = view("gge_biplot")
    www_svp, (G_www, E_www) = view("which_won_where")
    www = gxe_stats.which_won_where(G_www, E_www)
    ms_svp, (G_ms, E_ms) = view("mean_stability")
    aec_genotype = gxe_stats.average_environment_axis(G_ms, E_ms)
    rank_svp, (G_rank, E_rank) = view("ranking")
    aec_ranking = gxe_stats.average_environment_axis(G_rank, E_rank)
    disc_svp, (G_disc, E_disc) = view("discriminativeness")
    aec_environment = gxe_stats.average_environment_axis(G_disc, E_disc)

    mean_rank = ranks(genotype_means, descending=higher_is_better)
    asv_rank = ranks(asv, descending=False) if asv is not None else None
    ysi = [a + m for a, m in zip(asv_rank, mean_rank)] if asv_rank else None
    ysi_rank = ranks(ysi, descending=False) if ysi else None
    ideal_rank = ranks(aec_ranking["genotype_distance"], descending=False)

    def optional(values, i):
        return number(values[i]) if values is not None else None

    genotype_rows = []
    for i, name in enumerate(genotypes):
        genotype_rows.append({
            "genotype": name,
            "mean": number(genotype_means[i]),
            "rank": mean_rank[i],
            "bi": number(stability["bi"][i]),
            "bi_se": number(stability["bi_se"][i]),
            "bi_p": number(stability["bi_p"][i]),
            "deviation_ms": number(stability["deviation_ms"][i]),
            "s2di": optional(stability["s2di"], i),
            "s2di_p": optional(stability["s2di_p"], i),
            "r2": number(stability["r2"][i]),
            "ecovalence": number(stability["ecovalence"][i]),
            "ecovalence_percent": number(stability["ecovalence_percent"][i]),
            "shukla": number(stability["shukla"][i]),
            "cv": number(stability["cv"][i]),
            "ipca1": number(genotype_scores[i, 0]),
            "ipca2": number(genotype_scores[i, 1]),
            "asv": optional(asv, i),
            "asv_rank": asv_rank[i] if asv_rank else None,
            "ysi": ysi[i] if ysi else None,
            "ysi_rank": ysi_rank[i] if ysi_rank else None,
            "gge_pc1": number(G_basic[i, 0]),
            "gge_pc2": number(G_basic[i, 1]),
            "aec_mean": number(aec_genotype["projection"][i]),
            "aec_instability": number(aec_genotype["instability"][i]),
            "ideal_distance": number(aec_ranking["genotype_distance"][i]),
            "ideal_rank": ideal_rank[i],
            "sector_winner": i in www["vertices"] and i in www["winners"],
        })

    environment_rows = []
    for j, name in enumerate(environments):
        environment_rows.append({
            "environment": name,
            "mean": number(environment_means[j]),
            "index": number(stability["environment_index"][j]),
            "ipca1": number(environment_scores[j, 0]),
            "ipca2": number(environment_scores[j, 1]),
            "gge_pc1": number(E_basic[j, 0]),
            "gge_pc2": number(E_basic[j, 1]),
            "vector_length": number(aec_environment["environment_length"][j]),
            "representativeness": number(aec_environment["representativeness"][j]),
            "angle": number(aec_environment["angle"][j]),
            "ideal_distance": number(aec_environment["environment_distance"][j]),
            "winner": genotypes[www["winners"][j]],
        })

    mega = {}
    for j, winner in enumerate(www["winners"]):
        mega.setdefault(genotypes[winner], []).append(environments[j])

    scaling_label = SCALINGS[scaling]
    charts = {
        "gge_biplot": gxe_charts.gge_biplot_chart(G_basic, E_basic, genotypes, environments, percent,
                                                  SVP_LABELS[basic_svp], scaling_label, palette),
        "which_won_where": gxe_charts.which_won_where_chart(G_www, E_www, genotypes, environments, www, percent, palette),
        "mean_stability": gxe_charts.mean_stability_chart(G_ms, E_ms, genotypes, aec_genotype, percent, palette),
        "discriminativeness": gxe_charts.discriminativeness_chart(E_disc, environments, aec_environment, percent, palette),
        "ranking": gxe_charts.ranking_chart(G_rank, genotypes, aec_ranking, percent, palette),
        "ammi1": gxe_charts.ammi1_chart(genotype_means, environment_means, genotype_scores[:, 0], environment_scores[:, 0],
                                        genotypes, environments, grand, ipca[0]["percent"] if ipca else 0.0, response_col,
                                        palette),
        "ammi2": gxe_charts.ammi2_chart(genotype_scores, environment_scores, genotypes, environments,
                                        [ipca[0]["percent"], ipca[1]["percent"]], palette) if len(ipca) >= 2 else None,
        "adaptability": gxe_charts.adaptability_chart(means, genotypes, environments, stability["bi"], response_col,
                                                      higher_is_better, palette),
        "heatmap": gxe_charts.heatmap_chart(means, genotypes, environments, response_col, higher_is_better, palette),
    }

    return {
        "genotypes": genotypes,
        "environments": environments,
        "response": response_col,
        "higher_is_better": bool(higher_is_better),
        "n_observations": int(len(frame)),
        "replicates": int(r) if balanced else None,
        "balanced": balanced,
        "means": {
            "matrix": [[number(v) for v in row] for row in means],
            "genotype_means": [number(v) for v in genotype_means],
            "environment_means": [number(v) for v in environment_means],
            "grand_mean": grand,
        },
        "anova": {
            "rows": anova["rows"],
            "error_df": anova["df_error"],
            "error_ms": number(anova["ms_error"]),
            "blocks": anova["blocks"],
        },
        "ammi": {"ipca": ipca},
        "genotype_stats": genotype_rows,
        "environment_stats": environment_rows,
        "gge": {
            "scaling": scaling,
            "svp": {"gge_biplot": basic_svp, "which_won_where": www_svp, "mean_stability": ms_svp, "ranking": rank_svp,
                    "discriminativeness": disc_svp},
            "percent": percent,
            "vertex_genotypes": [genotypes[i] for i in www["vertices"]],
            "mega_environments": [{"winner": w, "environments": envs} for w, envs in mega.items()],
            "ideal_genotype_ranking": [genotypes[i] for i in np.argsort(aec_ranking["genotype_distance"])],
        },
        "environment_correlations": gxe_stats.environment_correlations(means),
        "charts": charts,
        "notes": notes,
    }
