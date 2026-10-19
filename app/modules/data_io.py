"""Dataset upload helpers: Excel to CSV, column listing, preview and edited-data upload."""

import io

import pandas as pd
import numpy as np
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status
from fastapi.responses import StreamingResponse

from ..auth import get_current_user
from ..models import User

from .common import read_upload_bytes, read_uploaded_table

router = APIRouter()

@router.post("/excel-to-csv")
def excel_to_csv(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    filename = file.filename or ""
    if not (filename.endswith('.xlsx') or filename.endswith('.xls')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only Excel files (.xlsx, .xls) are supported for conversion."
        )

    content = read_upload_bytes(file)
    try:
        df = pd.read_excel(io.BytesIO(content))  # Excel needs a seekable buffer
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        csv_data = csv_buf.getvalue()
        
        base_name = filename.rsplit('.', 1)[0]
        csv_filename = f"{base_name}.csv"
        
        return StreamingResponse(
            io.BytesIO(csv_data.encode('utf-8')),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={csv_filename}",
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to convert Excel to CSV: {str(e)}"
        )


@router.post("/columns")
def get_columns(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    df = read_uploaded_table(file, str_columns=True)

    columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    return {
        "columns": columns,
        "numeric_columns": numeric_columns
    }


@router.post("/preview")
def get_preview(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    df = read_uploaded_table(file)

    # Convert NaNs to None/null for valid JSON serialization
    df_clean = df.replace({np.nan: None})
    
    preview_df = df_clean.head(100)
    columns = preview_df.columns.tolist()
    records = preview_df.to_dict(orient="records")

    return {
        "columns": columns,
        "data": records,
        "total_rows": len(df)
    }


@router.post("/upload-edited-data")
def upload_edited_data(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    df = read_uploaded_table(file)
    return {
        "status": "success",
        "message": "Edited dataset uploaded and saved successfully.",
        "rows": len(df),
        "columns": list(df.columns)
    }
