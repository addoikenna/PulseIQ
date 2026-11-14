"""Staged upload -> optional cleaning -> confirmed analysis.

Add with `app.include_router(preparation_router)`. The existing /analyze remains
unchanged for older clients. No generated dashboard is created during prepare.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io

from app.services.auth import get_current_user_id
from app.services.data_analyzer import analyze_dataframe
from app.services.data_storage import (
    DATASET_BUCKET, build_dataset_storage_path, dataframe_to_parquet_bytes,
    download_stored_dataset,
)
from app.services.file_handler import read_uploaded_file
from app.services.response_formatter import format_analysis_response
from app.services.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)
preparation_router = APIRouter(prefix="/preparation", tags=["Data Preparation"])
MAX_BYTES = 5 * 1024 * 1024
EXTENSIONS = (".csv", ".xlsx", ".xls")


class ConfirmAnalysisRequest(BaseModel):
    analysis_id: str
    version_id: str | None = None


def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name.replace("\\", "/").split("/")[-1])[:120] or "uploaded_file"


def owned_analysis(analysis_id: str, user_id: str):
    db = get_supabase_client()
    response = (db.table("analyses")
                .select("id,user_id,filename,analysis_json,dataset_storage_path,status")
                .eq("id", analysis_id).eq("user_id", user_id).limit(1).execute())
    if not response.data:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return response.data[0]


@preparation_router.post("/upload", summary="Upload and profile a dataset without generating analysis")
async def prepare_dataset(file: UploadFile = File(...), user_id: str = Depends(get_current_user_id)):
    if not file.filename or not file.filename.lower().endswith(EXTENSIONS):
        raise HTTPException(status_code=400, detail="Upload a CSV, XLSX or XLS file")
    content = await file.read(MAX_BYTES + 1)
    if not content or len(content) > MAX_BYTES:
        raise HTTPException(status_code=413 if content else 400,
                            detail="File must be non-empty and no larger than 5 MB")
    try:
        file.file.seek(0)
        df = read_uploaded_file(file)
        if df is None or df.empty:
            raise ValueError("The uploaded dataset is empty")
        parquet = dataframe_to_parquet_bytes(df)
    except Exception as error:
        logger.info("Dataset preparation validation failed: %s", error)
        raise HTTPException(status_code=422,
                            detail="Could not parse this dataset. Check column types and file format.") from error

    analysis_id = str(uuid4())
    base_path = build_dataset_storage_path(user_id, analysis_id)
    original_path = f"{user_id}/{analysis_id}/original/{safe_name(file.filename)}"
    db = get_supabase_client()
    uploaded = []
    now = datetime.now(timezone.utc).isoformat()
    try:
        db.storage.from_(DATASET_BUCKET).upload(
            path=original_path, file=content,
            file_options={"content-type": "application/octet-stream", "upsert": "false"})
        uploaded.append(original_path)
        db.storage.from_(DATASET_BUCKET).upload(
            path=base_path, file=parquet,
            file_options={"content-type": "application/octet-stream", "upsert": "false"})
        uploaded.append(base_path)
        db.table("analyses").insert({
            "id": analysis_id, "user_id": user_id, "filename": file.filename,
            "rows": int(len(df)), "columns": int(len(df.columns)),
            "status": "active", "analysis_json": {
                "status": "prepared", "analysis_id": analysis_id,
                "filename": file.filename, "message": "Dataset awaiting user confirmation",
            },
            "dataset_storage_path": base_path,
            "dataset_format": "parquet", "dataset_size_bytes": len(parquet),
            "dataset_row_count": int(len(df)), "dataset_column_count": int(len(df.columns)),
            "dataset_stored_at": now, "prediction_ready": False,
        }).execute()
    except Exception as error:
        logger.exception("Unable to prepare dataset")
        for path in uploaded:
            try:
                db.storage.from_(DATASET_BUCKET).remove([path])
            except Exception:
                logger.warning("Orphaned preparation object: %s", path)
        raise HTTPException(status_code=503, detail="Could not save the dataset for preparation") from error
    return {"status": "prepared", "analysis_id": analysis_id, "filename": file.filename,
            "rows": int(len(df)), "columns": int(len(df.columns)),
            "next_step": "Review data quality and optionally clean before generating analysis"}


@preparation_router.post("/confirm", summary="Generate analysis from a confirmed dataset version")
def confirm_dataset(request: ConfirmAnalysisRequest,
                    user_id: str = Depends(get_current_user_id)):
    analysis = owned_analysis(request.analysis_id, user_id)
    existing = analysis.get("analysis_json") or {}
    if existing.get("status") != "prepared":
        if (existing.get("status") == "success" and
                existing.get("selected_cleaning_version_id") == request.version_id):
            return existing  # idempotent retry with same confirmed selection
        raise HTTPException(status_code=409,
                            detail="Analysis was already generated. Start a new analysis to change its dataset.")

    db = get_supabase_client()
    path = analysis["dataset_storage_path"]
    events = []
    if request.version_id:
        version = (db.table("cleaning_versions").select("id,storage_path,events")
                   .eq("id", request.version_id).eq("analysis_id", request.analysis_id)
                   .eq("user_id", user_id).limit(1).execute())
        if not version.data:
            raise HTTPException(status_code=404, detail="Cleaned dataset version not found")
        path = version.data[0]["storage_path"]
        events = version.data[0].get("events") or []

    try:
        df = download_stored_dataset(path)
        summary = analyze_dataframe(df)
        summary["analysis_id"] = request.analysis_id
        summary["cleaning_report"] = {
            "source": "user_confirmed_cleaning_version" if request.version_id else "user_confirmed_starting_dataset",
            "events": events,
        }
        summary["dataset_metadata"] = {
            "dataset_storage_path": path, "dataset_format": "parquet",
            "dataset_row_count": int(len(df)),
            "dataset_column_count": int(len(df.columns)),
            "dataset_stored_at": datetime.now(timezone.utc).isoformat(),
            "dataset_expires_at": None, "prediction_ready": True,
        }
        output = jsonable_encoder(format_analysis_response(summary, analysis["filename"]))
        output["status"] = "success"
        output["analysis_id"] = request.analysis_id
        output["dataset_metadata"] = summary["dataset_metadata"]
        output["selected_cleaning_version_id"] = request.version_id
        # Supabase service-role is only used after ownership validation.
        db.table("analyses").update({
            "analysis_json": output,
            "rows": int(len(df)), "columns": int(len(df.columns)),
            "quality_score": output.get("overview", {}).get("data_quality_score"),
            "dataset_storage_path": path,
            "dataset_row_count": int(len(df)),
            "dataset_column_count": int(len(df.columns)),
            "dataset_size_bytes": len(dataframe_to_parquet_bytes(df)),
            "prediction_ready": True,
        }).eq("id", request.analysis_id).eq("user_id", user_id).execute()
        return output
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Analysis confirmation failed")
        raise HTTPException(status_code=503,
                            detail="Unable to generate analysis from the selected dataset") from error


@preparation_router.get("/analyses/{analysis_id}/original", summary="Download exact original uploaded file")
def download_original(analysis_id: str, user_id: str = Depends(get_current_user_id)):
    analysis = owned_analysis(analysis_id, user_id)
    original_path = f"{user_id}/{analysis_id}/original/{safe_name(analysis['filename'])}"
    try:
        payload = get_supabase_client().storage.from_(DATASET_BUCKET).download(original_path)
    except Exception:
        raise HTTPException(status_code=404, detail="Original uploaded file is unavailable")
    return StreamingResponse(io.BytesIO(payload), media_type="application/octet-stream",
                             headers={"Content-Disposition": f'attachment; filename="{safe_name(analysis["filename"])}"'})
