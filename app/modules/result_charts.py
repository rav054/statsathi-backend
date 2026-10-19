"""Alternative charts for analysis results: raincloud, box + beeswarm, mean ± CI, estimation plots,
paired lines, mosaic plot and two-factor mean heatmap.

The analysis modules keep returning their classic figure; the frontend asks this endpoint for another
view of the same data, passing the test's p-value, post-hoc letters or reference value to annotate it.
"""

import base64
import json
from typing import Optional

import matplotlib.pyplot as plt
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from ..auth import get_current_user
from ..figure_style import PALETTES, export_figure, figure_style
from ..models import User
from . import chart_gallery
from .common import read_uploaded_table

router = APIRouter()

RESULT_CHARTS = ("raincloud", "boxswarm", "meanci", "estimation", "paired", "paired_estimation", "mosaic", "meanheatmap")
LAYOUTS = ("long", "wide", "wide_complete")


def bad_request(message):
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)


@router.post("/result-chart")
def result_chart(
    file: UploadFile = File(...),
    chart: str = Form(...),
    layout: str = Form("long"),  # "long": x/y/hue columns; "wide": one column per group; "wide_complete": complete rows only
    cols_str: Optional[str] = Form(None),
    x_var: Optional[str] = Form(None),
    y_var: Optional[str] = Form(None),
    hue_var: Optional[str] = Form(None),
    letters_json: Optional[str] = Form(None),  # compact letter display, e.g. {"T1": "a", "T2": "ab"}
    p_value: Optional[float] = Form(None),
    test_label: Optional[str] = Form(None),
    reference: Optional[float] = Form(None),  # e.g. the hypothesized mean of a one-sample test
    reference_label: Optional[str] = Form(None),
    equal_var: Optional[bool] = Form(None),
    palette: Optional[str] = Form(None),
    dpi: int = Form(150),
    current_user: User = Depends(get_current_user)
):
    if chart not in RESULT_CHARTS:
        raise bad_request(f"Chart must be one of: {', '.join(RESULT_CHARTS)}.")
    if layout not in LAYOUTS:
        raise bad_request(f"Layout must be one of: {', '.join(LAYOUTS)}.")
    df = read_uploaded_table(file)
    extra = {}

    if layout == "long":
        for column in (x_var, y_var, hue_var):
            if column and column not in df.columns:
                raise bad_request(f"Column '{column}' not found in dataset.")
        frame, x, y, hue = df, x_var, y_var, hue_var
    else:
        columns = list(dict.fromkeys(c.strip() for c in (cols_str or "").split(",") if c.strip()))
        if not columns:
            raise bad_request("Choose at least one column to plot.")
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise bad_request(f"Column(s) not found in dataset: {', '.join(missing)}.")
        data = df[columns].dropna() if layout == "wide_complete" else df[columns]
        frame = data.melt(value_vars=columns, var_name="Variable", value_name="Value").dropna(subset=["Value"])
        x, y, hue = "Variable", "Value", None
        extra["orders"] = {"Variable": columns}

    if letters_json:
        try:
            letters = json.loads(letters_json)
            if not isinstance(letters, dict):
                raise ValueError
        except ValueError:
            raise bad_request("letters_json must be a JSON object mapping group names to letters.")
        extra["letters"] = {str(k): str(v) for k, v in letters.items()}
    if reference is not None:
        extra["reference"] = {"value": reference, "label": reference_label}
    if equal_var is not None:
        extra["equal_var"] = equal_var

    title = test_label or None
    if p_value is not None:
        p_text = chart_gallery.format_p(p_value)
        two_groups = x in frame.columns and frame[x].dropna().nunique() == 2
        if chart in ("raincloud", "boxswarm", "meanci") and two_groups and not extra.get("letters") and not hue:
            extra["bracket"] = p_text
        else:
            title = f"{test_label} ({p_text})" if test_label else None

    spec = chart_gallery.analysis_spec(x=x, y=y, hue=hue, palette=palette, extra=extra, title=title)
    fig = None
    try:
        with figure_style(palette=palette if palette in PALETTES else None):
            fig = chart_gallery.render(chart, frame, spec)
            content, _ = export_figure(fig, "png", max(72, min(int(dpi), 600)))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Failed to draw the chart: {str(e)}")
    finally:
        if fig is not None:
            plt.close(fig)
    return {"plot": base64.b64encode(content).decode("utf-8")}
