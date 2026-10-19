"""Figure composer: combine chart images into one labelled multi-panel figure for journals."""

import base64
import json
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
import io

from .. import figure_composer as composer
from ..auth import get_current_user
from ..figure_style import EXPORT_MEDIA_TYPES, MM_PER_INCH, PRESETS, get_preset, normalize_format, preview_dpi
from ..models import User

router = APIRouter()

MAX_FILE_BYTES = 15 * 1024 * 1024
MAX_TOTAL_BYTES = 45 * 1024 * 1024


def bad_request(message):
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)


def parse_list(text, name):
    if not text:
        return None
    try:
        values = json.loads(text)
    except json.JSONDecodeError:
        raise bad_request(f"{name} must be a JSON list of strings.")
    if not isinstance(values, list):
        raise bad_request(f"{name} must be a JSON list of strings.")
    return ["" if value is None else str(value) for value in values]


def read_uploads(files):
    if not files:
        raise bad_request("Add at least one image.")
    if len(files) > composer.MAX_PANELS:
        raise bad_request(f"At most {composer.MAX_PANELS} panels can be combined into one figure.")
    streams, total = [], 0
    for upload in files:
        content = upload.file.read()
        total += len(content)
        if len(content) > MAX_FILE_BYTES:
            raise bad_request(f"'{upload.filename}' is larger than 15 MB.")
        if total > MAX_TOTAL_BYTES:
            raise bad_request("The images add up to more than 45 MB.")
        streams.append((upload.filename or "image", content))
    try:
        return streams, composer.load_images(streams)
    except ValueError as error:
        raise bad_request(str(error))


def build(files, columns, preset, spacing_mm, label_style, label_position, labels_json, captions_json, figure_caption,
          background, fit="row"):
    if preset not in PRESETS:
        raise bad_request(f"Size preset must be one of: {', '.join(PRESETS)}.")
    streams, images = read_uploads(files)
    options = dict(
        columns=int(columns) if columns else None,
        preset=preset,
        spacing_mm=spacing_mm,
        label_style=label_style,
        label_position=label_position,
        labels=parse_list(labels_json, "Labels"),
        captions=parse_list(captions_json, "Captions"),
        figure_caption=figure_caption or "",
        background=background,
        fit=fit,
    )
    return streams, images, options


def describe(streams, images, layout, preset):
    width_in, height_in = layout["figure_size"]
    return {
        "rows": layout["rows"],
        "columns": layout["columns"],
        "size_mm": {"width": round(width_in * MM_PER_INCH, 1), "height": round(height_in * MM_PER_INCH, 1)},
        "preset": {"key": preset, "label": get_preset(preset)["label"], "dpi": get_preset(preset)["dpi"]},
        "panels": [{"index": panel["index"], "filename": streams[panel["index"]][0],
                    "pixels": [images[panel["index"]].width, images[panel["index"]].height],
                    "label": composer.panel_label(panel["index"])} for panel in layout["panels"]],
    }


@router.post("/figure/compose")
def compose_figure(
    files: List[UploadFile] = File(...),
    columns: Optional[int] = Form(None),  # panels per row; blank = automatic
    preset: str = Form("double_column"),
    spacing_mm: float = Form(3.0),
    label_style: str = Form("lower"),  # lower, upper, lower_paren, upper_paren, none
    label_position: str = Form("outside"),  # outside, inside, none
    labels_json: Optional[str] = Form(None),  # per-panel label overrides
    captions_json: Optional[str] = Form(None),  # per-panel captions
    figure_caption: Optional[str] = Form(None),
    background: str = Form("white"),
    fit: str = Form("row"),  # row = fill each row, width = equal panel widths
    current_user: User = Depends(get_current_user)
):
    streams, images, options = build(files, columns, preset, spacing_mm, label_style, label_position, labels_json,
                                     captions_json, figure_caption, background, fit)
    try:
        content, _, layout = composer.render(images, "png", preview_dpi(preset), **options)
    except ValueError as error:
        raise bad_request(str(error))
    return {"image": base64.b64encode(content).decode("utf-8"), **describe(streams, images, layout, preset)}


@router.post("/figure/download")
def download_figure(
    files: List[UploadFile] = File(...),
    columns: Optional[int] = Form(None),
    preset: str = Form("double_column"),
    spacing_mm: float = Form(3.0),
    label_style: str = Form("lower"),
    label_position: str = Form("outside"),
    labels_json: Optional[str] = Form(None),
    captions_json: Optional[str] = Form(None),
    figure_caption: Optional[str] = Form(None),
    background: str = Form("white"),
    fit: str = Form("row"),
    download_format: str = Form("png"),
    dpi: Optional[int] = Form(None),
    current_user: User = Depends(get_current_user)
):
    try:
        fmt = normalize_format(download_format)
    except ValueError:
        raise bad_request(f"Format must be one of: {', '.join(EXPORT_MEDIA_TYPES)}.")
    if dpi is not None and not 72 <= dpi <= 1200:
        raise bad_request("Resolution must be between 72 and 1200 DPI.")
    streams, images, options = build(files, columns, preset, spacing_mm, label_style, label_position, labels_json,
                                     captions_json, figure_caption, background, fit)
    try:
        content, media_type, _ = composer.render(images, fmt, dpi, **options)
    except ValueError as error:
        raise bad_request(str(error))
    extension = "jpg" if fmt == "jpeg" else fmt
    headers = {"Content-Disposition": f"attachment; filename=statsathi_figure.{extension}"}
    return StreamingResponse(io.BytesIO(content), media_type=media_type, headers=headers)
