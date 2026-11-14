"""Authenticated cleaning preview and immutable dataset version APIs.

Register this router with app.include_router(cleaning_router) in app/main.py.
"""
from __future__ import annotations

import io
import uuid
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.auth import get_current_user_id
from app.services.data_storage import (download_stored_dataset, build_dataset_storage_path,
                                       dataframe_to_parquet_bytes, DATASET_BUCKET)
from app.services.supabase_client import get_supabase_client
from app.services.cleaning_engine import CleaningRule, apply_rules, profile

cleaning_router = APIRouter(prefix="/cleaning", tags=["Data Cleaning"])


# The frontend can build column-type-aware menus from these server capabilities.
# This endpoint contains no user data and does not execute cleaning rules.
@cleaning_router.get("/operations", summary="List supported cleaning operations")
def list_cleaning_operations():
    return {"operations": [
        {"operation": "map_values", "data_types": ["text", "numeric", "boolean"], "required": ["column", "mapping"]},
        {"operation": "replace_values", "data_types": ["text", "numeric", "boolean"], "required": ["column", "mapping"]},
        {"operation": "trim", "data_types": ["text"], "required": ["column"]},
        {"operation": "case", "data_types": ["text"], "required": ["column", "case_style"]},
        {"operation": "replace_missing", "data_types": ["all"], "required": ["column", "missing_value"]},
        {"operation": "impute_missing", "data_types": ["numeric", "text", "boolean"], "required": ["column", "strategy"],
         "note": "Mean/median require numeric data; mode supports other types. Use train-only preprocessing for ML."},
        {"operation": "convert_type", "data_types": ["all"], "required": ["column", "target_type"]},
        {"operation": "parse_date", "data_types": ["text", "datetime"], "required": ["column"]},
        {"operation": "rename_column", "data_types": ["all"], "required": ["column", "new_column"]},
        {"operation": "drop_column", "data_types": ["all"], "required": ["column"]},
        {"operation": "drop_duplicates", "data_types": ["all"], "required": []},
        {"operation": "flag_outliers", "data_types": ["numeric"], "required": ["column", "new_column"]},
        {"operation": "clip_range", "data_types": ["numeric"], "required": ["column", "lower_bound and/or upper_bound"]},
    ]}


class CleaningRequest(BaseModel):
    analysis_id: str
    rules: list[CleaningRule] = Field(default_factory=list, max_length=50)
    base_version_id: str | None = None
    version_name: str | None = Field(default=None, max_length=100)


def load_base(analysis_id: str, user_id: str, version_id: str | None):
    # The upload is allowed to be 'prepared' (prediction_ready=False).
    # Authenticate ownership explicitly: the service-role connection bypasses RLS.
    db = get_supabase_client()
    own = (db.table("analyses")
           .select("id,user_id,filename,dataset_storage_path")
           .eq("id", analysis_id).eq("user_id", user_id).limit(1).execute())
    if not own.data:
        raise HTTPException(status_code=404, detail="Dataset not found")
    analysis = own.data[0]
    # A fixed starting path ensures choosing a cleaned version for a dashboard
    # never alters the original cleaning baseline of the analysis.
    path = build_dataset_storage_path(user_id, analysis_id)
    if version_id:
        response = (db.table("cleaning_versions")
                    .select("id, storage_path")
                    .eq("id", version_id).eq("analysis_id", analysis_id)
                    .eq("user_id", user_id).limit(1).execute())
        if not response.data:
            raise HTTPException(status_code=404, detail="Dataset version not found")
        path = response.data[0]["storage_path"]
    return analysis, download_stored_dataset(path)


def checked_load(analysis_id, user_id, version_id):
    try:
        return load_base(analysis_id, user_id, version_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Analysis not found")


@cleaning_router.get("/analyses/{analysis_id}/profile")
def get_cleaning_profile(analysis_id: str, base_version_id: str | None = None,
                         user_id: str = Depends(get_current_user_id)):
    analysis, df = checked_load(analysis_id, user_id, base_version_id)
    return {"analysis_id": analysis_id, "filename": analysis.get("filename"),
            "base_version_id": base_version_id, **profile(df)}


@cleaning_router.post("/preview")
def preview_cleaning(request: CleaningRequest, user_id: str = Depends(get_current_user_id)):
    _, df = checked_load(request.analysis_id, user_id, request.base_version_id)
    try:
        changed, events = apply_rules(df, request.rules)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
    return {"analysis_id": request.analysis_id, "base_version_id": request.base_version_id,
            "before": profile(df), "after": profile(changed), "events": events,
            "saved": False}


@cleaning_router.post("/apply")
def save_cleaned_version(request: CleaningRequest, user_id: str = Depends(get_current_user_id)):
    if not request.rules:
        raise HTTPException(status_code=422, detail="Choose at least one cleaning rule")
    _, df = checked_load(request.analysis_id, user_id, request.base_version_id)
    display_name = request.version_name.strip() if request.version_name else None
    if request.version_name is not None and not display_name:
        raise HTTPException(status_code=422, detail="Version name cannot be blank")
    try:
        changed, events = apply_rules(df, request.rules)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
    version_id = str(uuid.uuid4())
    path = f"{user_id}/{request.analysis_id}/cleaning_versions/{version_id}.parquet"
    db = get_supabase_client()
    try:
        db.storage.from_(DATASET_BUCKET).upload(
            path=path, file=dataframe_to_parquet_bytes(changed),
            file_options={"content-type": "application/octet-stream", "upsert": "false"})
        row = {
            "id": version_id, "analysis_id": request.analysis_id, "user_id": user_id,
            "display_name": display_name,
            "parent_version_id": request.base_version_id, "storage_path": path,
            "rules": [r.model_dump(mode="json") for r in request.rules],
            "events": events, "row_count": len(changed),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        db.table("cleaning_versions").insert(row).execute()
    except Exception:
        # Remove any orphaned uploaded object if DB insert failed.
        try:
            db.storage.from_(DATASET_BUCKET).remove([path])
        except Exception:
            pass
        raise HTTPException(status_code=503, detail="Unable to save dataset version")
    return {"analysis_id": request.analysis_id, "version_id": version_id,
            "parent_version_id": request.base_version_id, "display_name": display_name, "saved": True,
            "rows": len(changed), "events": events,
            "note": "Saved as a separate dataset version. Dashboard and training sources are unchanged."}


@cleaning_router.get("/analyses/{analysis_id}/versions")
def list_cleaning_versions(analysis_id: str, user_id: str = Depends(get_current_user_id)):
    checked_load(analysis_id, user_id, None)
    db = get_supabase_client()
    res = (db.table("cleaning_versions")
           .select("id, display_name, parent_version_id, row_count, rules, events, created_at")
           .eq("analysis_id", analysis_id).eq("user_id", user_id)
           .order("created_at", desc=True).execute())
    return {"analysis_id": analysis_id, "versions": res.data}


@cleaning_router.get("/analyses/{analysis_id}/download")
def download_cleaning_csv(analysis_id: str, version_id: str,
                          user_id: str = Depends(get_current_user_id)):
    _, df = checked_load(analysis_id, user_id, version_id)
    payload = df.to_csv(index=False).encode("utf-8-sig")
    return StreamingResponse(io.BytesIO(payload), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="pulseiq_cleaned_{version_id}.csv"'})
