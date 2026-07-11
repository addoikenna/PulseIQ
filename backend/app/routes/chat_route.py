from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path

from app.schemas.chat import ChatAskRequest, ChatAskResponse
from app.services.auth import get_current_user_id
from app.services.conversational_analytics import ask_analysis_question
from app.services.supabase_client import get_supabase_client


router = APIRouter(
    prefix="/chat",
    tags=["Conversational Analytics"],
)


@router.post(
    "/ask",
    response_model=ChatAskResponse,
    summary="Ask a question about an analysis",
)
def ask_question(
    request: ChatAskRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    Ask a grounded question about a saved analysis.

    If session_id is omitted, a new chat session is created.
    If session_id is supplied, the existing conversation continues.
    """

    try:
        return ask_analysis_question(
            analysis_id=request.analysis_id,
            session_id=request.session_id,
            question=request.question,
            user_id=user_id,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )

    except Exception as error:
        print(f"Chat ask endpoint error: {error}")

        raise HTTPException(
            status_code=500,
            detail="Unable to process the chat question.",
        )


@router.get(
    "/sessions",
    summary="List chat sessions",
)
def list_chat_sessions(
    user_id: str = Depends(get_current_user_id),
) -> dict[str, Any]:
    """
    Return all active chat sessions owned by the authenticated user.
    """

    try:
        supabase = get_supabase_client()

        response = (
            supabase.table("chat_sessions")
            .select(
                "id, analysis_id, title, status, "
                "last_message_at, created_at, updated_at"
            )
            .eq("user_id", user_id)
            .eq("status", "active")
            .order("last_message_at", desc=True)
            .execute()
        )

        return {
            "status": "success",
            "sessions": response.data or [],
        }

    except Exception as error:
        print(f"List chat sessions error: {error}")

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve chat sessions.",
        )


@router.get(
    "/sessions/{session_id}",
    summary="Get a chat session",
)
def get_chat_session(
    session_id: str = Path(
        ...,
        description="ID of the chat session.",
    ),
    user_id: str = Depends(get_current_user_id),
) -> dict[str, Any]:
    """
    Return one chat session and its messages.

    The authenticated user must own the session.
    """

    try:
        supabase = get_supabase_client()

        session_response = (
            supabase.table("chat_sessions")
            .select("*")
            .eq("id", session_id)
            .eq("user_id", user_id)
            .neq("status", "deleted")
            .limit(1)
            .execute()
        )

        if not session_response.data:
            raise HTTPException(
                status_code=404,
                detail="Chat session not found.",
            )

        messages_response = (
            supabase.table("chat_messages")
            .select(
                "id, role, content, answer_type, confidence, "
                "evidence, model_used, provider, created_at"
            )
            .eq("session_id", session_id)
            .eq("user_id", user_id)
            .order("created_at")
            .execute()
        )

        return {
            "status": "success",
            "session": session_response.data[0],
            "messages": messages_response.data or [],
        }

    except HTTPException:
        raise

    except Exception as error:
        print(f"Get chat session error: {error}")

        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve the chat session.",
        )


@router.patch(
    "/sessions/{session_id}/archive",
    summary="Archive a chat session",
)
def archive_chat_session(
    session_id: str = Path(
        ...,
        description="ID of the chat session to archive.",
    ),
    user_id: str = Depends(get_current_user_id),
) -> dict[str, str]:
    """
    Archive a chat session without deleting its messages.
    """

    try:
        supabase = get_supabase_client()
        now = datetime.now(timezone.utc).isoformat()

        existing_response = (
            supabase.table("chat_sessions")
            .select("id")
            .eq("id", session_id)
            .eq("user_id", user_id)
            .neq("status", "deleted")
            .limit(1)
            .execute()
        )

        if not existing_response.data:
            raise HTTPException(
                status_code=404,
                detail="Chat session not found.",
            )

        (
            supabase.table("chat_sessions")
            .update({
                "status": "archived",
                "updated_at": now,
            })
            .eq("id", session_id)
            .eq("user_id", user_id)
            .execute()
        )

        return {
            "status": "success",
            "message": "Chat session archived successfully.",
        }

    except HTTPException:
        raise

    except Exception as error:
        print(f"Archive chat session error: {error}")

        raise HTTPException(
            status_code=500,
            detail="Unable to archive the chat session.",
        )