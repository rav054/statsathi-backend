"""Helpers shared by the analysis modules."""
import io

import pandas as pd
from fastapi import HTTPException, UploadFile, status

TABLE_EXTENSIONS = (".csv", ".xlsx", ".xls")


def read_upload_bytes(file: UploadFile, max_bytes: int = None) -> bytes:
    """The whole uploaded file, or HTTP 413 if it is larger than the table-upload limit."""
    from ..config import settings

    limit = max_bytes or settings.MAX_TABLE_UPLOAD_MB * 1024 * 1024
    file.file.seek(0)
    content = file.file.read(limit + 1)
    if len(content) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"'{file.filename or 'This file'}' is larger than {limit // (1024 * 1024)} MB. "
                   "Remove unused columns or rows, or split it, and upload it again."
        )
    return content


def read_uploaded_table(file: UploadFile, str_columns: bool = False) -> pd.DataFrame:
    """Read an uploaded CSV or Excel file into a DataFrame; HTTP 400 if it is unsupported or unreadable.

    The upload is copied into memory first: Excel readers need a fully seekable file, which
    FastAPI's SpooledTemporaryFile does not provide on Python 3.10.
    """
    filename = (file.filename or "").lower()
    if not filename.endswith(TABLE_EXTENSIONS):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV or Excel files are supported."
        )
    content = io.BytesIO(read_upload_bytes(file))
    try:
        df = pd.read_csv(content) if filename.endswith(".csv") else pd.read_excel(content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse dataset file: {e}"
        )
    if str_columns:
        df.columns = df.columns.astype(str)
    return df
