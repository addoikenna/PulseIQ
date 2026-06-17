from fastapi import APIRouter, HTTPException, Query
from app.services.supabase_client import get_supabase_client


router = APIRouter(prefix="/analyses", tags=["Analyses"])


@router.get("")
def list_analyses(user_id: str = Query(...)):
    try:
        supabase = get_supabase_client()

        response = (
            supabase.table("analyses")
            .select("id, filename, rows, columns, quality_score, status, created_at, updated_at")
            .eq("user_id", user_id)
            .eq("status", "active")
            .order("created_at", desc=True)
            .execute()
        )

        return {
            "status": "success",
            "analyses": response.data,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{analysis_id}")
def get_analysis(analysis_id: str, user_id: str = Query(...)):
    try:
        supabase = get_supabase_client()

        response = (
            supabase.table("analyses")
            .select("*")
            .eq("id", analysis_id)
            .eq("user_id", user_id)
            .single()
            .execute()
        )

        if not response.data:
            raise HTTPException(status_code=404, detail="Analysis not found.")

        return {
            "status": "success",
            "analysis": response.data,
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{analysis_id}/archive")
def archive_analysis(analysis_id: str, user_id: str = Query(...)):
    try:
        supabase = get_supabase_client()

        response = (
            supabase.table("analyses")
            .update({"status": "archived"})
            .eq("id", analysis_id)
            .eq("user_id", user_id)
            .execute()
        )

        return {
            "status": "success",
            "message": "Analysis archived successfully.",
            "analysis": response.data,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))