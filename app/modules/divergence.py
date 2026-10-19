"""Genetic divergence: Mahalanobis D², Tocher clustering, character contributions and canonical variates."""

from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from .. import divergence as div
from ..auth import get_current_user
from ..models import User
from . import chart_gallery, divergence_charts
from .common import read_uploaded_table

router = APIRouter()

MAX_GENOTYPES = 300
MAX_TRAITS = 30


def bad_request(message):
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)


def levels(series):
    return sorted(series.unique(), key=chart_gallery.natural_key)


def matrix(values):
    return [[float(v) for v in row] for row in np.asarray(values, dtype=float)]


@router.post("/divergence")
def analyze_divergence(
    file: UploadFile = File(...),
    genotype_col: str = Form(...),
    traits_str: str = Form(...),  # comma-separated traits, in the order used for the transformation
    rep_col: Optional[str] = Form(None),  # replicate / block
    palette: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user)
):
    df = read_uploaded_table(file)
    traits = list(dict.fromkeys(t.strip() for t in traits_str.split(",") if t.strip()))
    rep_col = rep_col or None
    if len(traits) < 2:
        raise bad_request("Choose at least 2 traits for D² analysis.")
    if len(traits) > MAX_TRAITS:
        raise bad_request(f"Choose at most {MAX_TRAITS} traits.")
    roles = [genotype_col] + ([rep_col] if rep_col else [])
    missing = [c for c in roles + traits if c not in df.columns]
    if missing:
        raise bad_request(f"Column(s) not found in dataset: {', '.join(missing)}.")
    if len(set(roles)) != len(roles) or set(roles) & set(traits):
        raise bad_request("The genotype and replicate columns must be different and cannot also be traits.")

    notes = []
    frame = df[roles + traits].copy()
    for trait in traits:
        values = pd.to_numeric(frame[trait], errors="coerce")
        if (values.isna() & frame[trait].notna()).any():
            raise bad_request(f"'{trait}' contains non-numeric values.")
        frame[trait] = values
    before = len(frame)
    frame = frame.dropna().reset_index(drop=True)
    if before - len(frame):
        dropped = before - len(frame)
        notes.append(f"{dropped} row{'s' if dropped != 1 else ''} with missing values {'were' if dropped != 1 else 'was'} left out.")

    frame["_g"] = chart_gallery.as_category(frame[genotype_col])
    genotypes = levels(frame["_g"])
    G, p = len(genotypes), len(traits)
    if G < 3:
        raise bad_request(f"D² analysis needs at least 3 genotypes (found {G}).")
    if G > MAX_GENOTYPES:
        raise bad_request(f"At most {MAX_GENOTYPES} genotypes can be analysed at once.")
    codes = frame["_g"].map({g: i for i, g in enumerate(genotypes)}).to_numpy()
    Y = frame[traits].to_numpy(dtype=float)
    means = pd.DataFrame(Y).groupby(codes).mean().to_numpy()
    counts = np.bincount(codes, minlength=G)
    replicated = bool(counts.max() >= 2)

    wilks = None
    if replicated:
        block = None
        if rep_col:
            frame["_r"] = chart_gallery.as_category(frame[rep_col])
            block = frame["_r"].map({r: i for i, r in enumerate(levels(frame["_r"]))}).to_numpy()
        test = div.manova_wilks(Y, codes, block)
        if test["df_error"] < p + 1:
            raise bad_request(f"The error term has {test['df_error']} degrees of freedom for {p} traits: D² needs more "
                              f"replication or fewer traits.")
        covariance = test["error_sscp"] / test["df_error"]
        if test["lambda"] is not None:
            wilks = {k: test[k] for k in ("lambda", "f", "df1", "df2", "p_value")}
            wilks.update({"df_error": test["df_error"], "df_genotypes": test["df_hypothesis"]})
        if counts.min() < 2:
            notes.append("Some genotypes have a single observation; the error covariance comes from the replicated genotypes.")
        covariance_source = "error"
    else:
        if G <= p + 1:
            raise bad_request(f"With one observation per genotype, D² needs more genotypes ({G}) than traits + 1 ({p + 1}).")
        covariance = np.cov(means, rowvar=False, ddof=1)
        covariance_source = "genotypes"
        notes.append("Each genotype has a single observation, so there is no error term: distances use the covariance "
                     "matrix among genotypes (Mahalanobis distance), and Wilks' test and canonical variates are not available.")

    try:
        d2, transformed = div.mahalanobis_d2(means, covariance)
    except np.linalg.LinAlgError:
        raise bad_request("The covariance matrix of the traits is singular: a trait is constant or a linear combination of "
                          "other traits. Remove redundant traits.")

    counts_first, percent_first = div.character_contributions(transformed)
    clusters, theta = div.tocher(d2)
    distances = div.cluster_distances(d2, clusters)
    names = [f"Cluster {divergence_charts.roman(i + 1)}" for i in range(len(clusters))]
    sizes = [len(m) for m in clusters]
    cluster_of = np.empty(G, dtype=int)
    for c, members in enumerate(clusters):
        cluster_of[members] = c
    cluster_means = np.array([means[members].mean(axis=0) for members in clusters])

    condensed = squareform(np.sqrt(np.clip(d2, 0, None)), checks=False)
    tree = hierarchy.linkage(condensed, "average")
    cophenetic = float(hierarchy.cophenet(tree, condensed)[0]) if G > 2 else None
    scores, divergence_percent = div.canonical_variates(transformed)

    charts = {
        "tocher": divergence_charts.tocher_diagram_chart(names, sizes, distances, palette),
        "d2_heatmap": divergence_charts.d2_heatmap_chart(d2, genotypes, clusters, palette),
        "dendrogram": divergence_charts.dendrogram_chart(tree, genotypes, cluster_of, names, sizes, cophenetic or 0.0, palette),
        "contributions": divergence_charts.contributions_chart(traits, counts_first, percent_first, palette),
        "cluster_means": divergence_charts.cluster_means_chart(names, traits, cluster_means, palette),
        "canonical": (divergence_charts.canonical_chart(scores, divergence_percent, genotypes, cluster_of, names, sizes, palette)
                      if replicated and scores.shape[1] >= 2 else None),
    }

    return {
        "genotypes": genotypes,
        "traits": traits,
        "replicated": replicated,
        "observations_per_genotype": [int(c) for c in counts],
        "n_observations": int(len(frame)),
        "covariance_source": covariance_source,
        "means": matrix(means),
        "covariance": matrix(covariance),
        "wilks": wilks,
        "d2": matrix(d2),
        "theta": theta,
        "clusters": [
            {"name": name, "members": [genotypes[g] for g in members], "size": len(members),
             "intra_d2": float(distances[c, c]), "intra_d": float(np.sqrt(distances[c, c]))}
            for c, (name, members) in enumerate(zip(names, clusters))
        ],
        "cluster_of": {genotypes[g]: names[cluster_of[g]] for g in range(G)},
        "cluster_distances": {"names": names, "d2": matrix(distances), "d": matrix(np.sqrt(distances))},
        "cluster_means": {"names": names, "values": matrix(cluster_means)},
        "contributions": [{"trait": t, "times_first": int(n), "percent": float(pc)}
                          for t, n, pc in zip(traits, counts_first, percent_first)],
        "canonical": ({"percent": [float(v) for v in divergence_percent], "scores": matrix(scores[:, :2])}
                      if replicated else None),
        "dendrogram_cophenetic": cophenetic,
        "charts": charts,
        "notes": notes,
    }
