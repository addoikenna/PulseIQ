import json
import os
from datetime import datetime, timezone
from typing import Any

import requests
from dotenv import load_dotenv
from google import genai

from app.services.chat_title_generator import generate_chat_title
from app.services.supabase_client import get_supabase_client
from app.services.conversation_context import build_conversation_context


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_PRIMARY_MODEL = os.getenv(
    "OPENROUTER_PRIMARY_MODEL",
    "deepseek/deepseek-chat-v3.1",
)
OPENROUTER_FALLBACK_MODEL = os.getenv(
    "OPENROUTER_FALLBACK_MODEL",
    "deepseek/deepseek-v3.1-terminus",
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Change this in .env only if your analyses table uses a different column.
ANALYSIS_PAYLOAD_COLUMN = os.getenv(
    "ANALYSIS_PAYLOAD_COLUMN",
    "analysis_json",
)

MAX_CHAT_HISTORY_MESSAGES = 12


def parse_llm_json(content: str) -> dict[str, Any]:
    if not content:
        raise ValueError("LLM returned empty content.")

    content = content.strip()

    if content.startswith("```json"):
        content = content.replace("```json", "", 1).replace("```", "").strip()
    elif content.startswith("```"):
        content = content.replace("```", "").strip()

    start = content.find("{")
    end = content.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model response.")

    return json.loads(content[start:end + 1])


def normalize_analysis_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        try:
            parsed = json.loads(value)

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Saved analysis field '{ANALYSIS_PAYLOAD_COLUMN}' "
        "does not contain a valid JSON object."
    )


def format_chat_history(
    messages: list[dict[str, Any]],
) -> list[dict[str, str]]:
    history = []

    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if role not in {"user", "assistant"}:
            continue

        if not content:
            continue

        history.append({
            "role": role,
            "content": str(content),
        })

    return history


def build_chat_prompt(
    question: str,
    analysis_context: dict[str, Any],
    chat_history: list[dict[str, str]],
) -> str:
    return f"""
You are PulseIQ, a grounded conversational business analyst.

Answer the user's question using only the saved analysis context supplied
below and the previous conversation where relevant.

Important rules:
- Do not invent figures, categories, causes, trends, or conclusions.
- Do not claim that a calculation was performed unless its result exists in
  the supplied analysis context.
- Do not use external knowledge to answer questions about this dataset.
- If the context does not contain enough evidence, say so clearly.
- Keep the answer useful, concise, and business-friendly.
- Refer to specific evidence where possible.
- Do not mention JSON, prompts, models, or backend implementation.
- Treat correlation as association, not proof of causation.
- Do not expose personal identifiers or contact details.
- Suggested questions must also be answerable from the supplied context.

Classify the answer as:
- analysis_context: the question can be answered from the available analysis.
- unsupported: the available analysis does not contain enough evidence.

Return ONLY valid JSON with exactly this structure:

{{
  "answer": "...",
  "answer_type": "analysis_context | unsupported",
  "confidence": "high | medium | low",
  "evidence": [
    {{
      "source": "...",
      "detail": "..."
    }}
  ],
  "suggested_questions": [
    "...",
    "..."
  ]
}}

- The supplied context was selected based on the detected question intent.
- Do not assume that missing context means the underlying dataset lacks the information.
- If the selected context is insufficient, classify the response as unsupported.

Previous conversation:
{json.dumps(chat_history, default=str)}

Relevant analysis context:
{json.dumps(analysis_context, default=str)}

Current user question:
{question}
"""


def call_gemini(prompt: str) -> dict[str, Any]:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured.")

    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    return parse_llm_json(response.text or "")


def call_openrouter(
    model: str,
    prompt: str,
) -> dict[str, Any]:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not configured.")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://pulseiq-dsxe.onrender.com",
        "X-Title": "PulseIQ",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a grounded business analyst. "
                    "Return only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.1,
        "response_format": {
            "type": "json_object",
        },
    }

    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=45,
    )

    response.raise_for_status()

    result = response.json()
    content = result["choices"][0]["message"].get("content", "")

    return parse_llm_json(content)


def generate_grounded_answer(
    question: str,
    analysis_context: dict[str, Any],
    chat_history: list[dict[str, str]],
) -> dict[str, Any]:
    prompt = build_chat_prompt(
        question=question,
        analysis_context=analysis_context,
        chat_history=chat_history,
    )

    try:
        result = call_gemini(prompt)

        return {
            **result,
            "model_used": GEMINI_MODEL,
            "provider": "gemini",
        }

    except Exception as error:
        print(f"Gemini conversational analytics error: {error}")

    for model in [
        OPENROUTER_PRIMARY_MODEL,
        OPENROUTER_FALLBACK_MODEL,
    ]:
        try:
            result = call_openrouter(model, prompt)

            return {
                **result,
                "model_used": model,
                "provider": "openrouter",
            }

        except Exception as error:
            print(
                "OpenRouter conversational analytics error "
                f"for model {model}: {error}"
            )

    return {
        "answer": (
            "I could not generate a reliable answer at this time. "
            "Please try again shortly."
        ),
        "answer_type": "unsupported",
        "confidence": "low",
        "evidence": [],
        "suggested_questions": [],
        "model_used": None,
        "provider": "fallback",
    }


def get_owned_analysis(
    analysis_id: str,
    user_id: str,
) -> dict[str, Any]:
    supabase = get_supabase_client()

    response = (
        supabase.table("analyses")
        .select(f"id, user_id, filename, {ANALYSIS_PAYLOAD_COLUMN}")
        .eq("id", analysis_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )

    if not response.data:
        raise ValueError("Analysis not found or access denied.")

    return response.data[0]


def create_chat_session(
    analysis_id: str,
    user_id: str,
    question: str,
) -> dict[str, Any]:
    supabase = get_supabase_client()
    session_title = generate_chat_title(question)
    now = datetime.now(timezone.utc).isoformat()

    response = (
        supabase.table("chat_sessions")
        .insert({
            "user_id": user_id,
            "analysis_id": analysis_id,
            "title": session_title,
            "status": "active",
            "last_message_at": now,
            "updated_at": now,
        })
        .execute()
    )

    if not response.data:
        raise ValueError("Unable to create chat session.")

    return response.data[0]


def get_owned_chat_session(
    session_id: str,
    analysis_id: str,
    user_id: str,
) -> dict[str, Any]:
    supabase = get_supabase_client()

    response = (
        supabase.table("chat_sessions")
        .select("*")
        .eq("id", session_id)
        .eq("analysis_id", analysis_id)
        .eq("user_id", user_id)
        .neq("status", "deleted")
        .limit(1)
        .execute()
    )

    if not response.data:
        raise ValueError("Chat session not found or access denied.")

    return response.data[0]


def load_recent_chat_messages(
    session_id: str,
    user_id: str,
) -> list[dict[str, Any]]:
    supabase = get_supabase_client()

    response = (
        supabase.table("chat_messages")
        .select("role, content, created_at")
        .eq("session_id", session_id)
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(MAX_CHAT_HISTORY_MESSAGES)
        .execute()
    )

    messages = response.data or []

    # Query returns newest first; Gemini should see chronological order.
    messages.reverse()

    return messages


def save_chat_message(
    session_id: str,
    user_id: str,
    role: str,
    content: str,
    answer_type: str | None = None,
    confidence: str | None = None,
    evidence: list[dict[str, Any]] | None = None,
    model_used: str | None = None,
    provider: str | None = None,
) -> None:
    supabase = get_supabase_client()

    payload = {
        "session_id": session_id,
        "user_id": user_id,
        "role": role,
        "content": content,
        "answer_type": answer_type,
        "confidence": confidence,
        "evidence": evidence or [],
        "model_used": model_used,
        "provider": provider,
    }

    supabase.table("chat_messages").insert(payload).execute()


def update_session_activity(session_id: str) -> None:
    supabase = get_supabase_client()
    now = datetime.now(timezone.utc).isoformat()

    (
        supabase.table("chat_sessions")
        .update({
            "last_message_at": now,
            "updated_at": now,
        })
        .eq("id", session_id)
        .execute()
    )


def sanitize_answer_result(
    result: dict[str, Any],
) -> dict[str, Any]:
    answer_type = result.get("answer_type", "unsupported")

    if answer_type not in {
        "analysis_context",
        "unsupported",
    }:
        answer_type = "unsupported"

    confidence = str(
        result.get("confidence", "low")
    ).lower()

    if confidence not in {
        "high",
        "medium",
        "low",
    }:
        confidence = "low"

    evidence = result.get("evidence", [])

    if not isinstance(evidence, list):
        evidence = []

    valid_evidence = []

    for item in evidence[:5]:
        if not isinstance(item, dict):
            continue

        source = item.get("source")
        detail = item.get("detail")

        if source and detail:
            valid_evidence.append({
                "source": str(source),
                "detail": str(detail),
            })

    suggested_questions = result.get(
        "suggested_questions",
        [],
    )

    if not isinstance(suggested_questions, list):
        suggested_questions = []

    suggested_questions = [
        str(item)
        for item in suggested_questions[:3]
        if item
    ]

    return {
        "answer": str(
            result.get(
                "answer",
                "The available analysis does not contain enough evidence.",
            )
        ),
        "answer_type": answer_type,
        "confidence": confidence,
        "evidence": valid_evidence,
        "suggested_questions": suggested_questions,
        "model_used": result.get("model_used"),
        "provider": result.get("provider"),
    }


def ask_analysis_question(
    analysis_id: str,
    question: str,
    user_id: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    analysis_record = get_owned_analysis(
        analysis_id=analysis_id,
        user_id=user_id,
    )

    analysis_payload = normalize_analysis_payload(
        analysis_record.get(ANALYSIS_PAYLOAD_COLUMN)
    )

    if session_id:
        session = get_owned_chat_session(
            session_id=session_id,
            analysis_id=analysis_id,
            user_id=user_id,
        )
    else:
        session = create_chat_session(
            analysis_id=analysis_id,
            user_id=user_id,
            question=question,
        )

    resolved_session_id = session["id"]
    session_title = session.get(
        "title",
        "Analysis Conversation",
    )

    previous_messages = load_recent_chat_messages(
        session_id=resolved_session_id,
        user_id=user_id,
    )

    save_chat_message(
        session_id=resolved_session_id,
        user_id=user_id,
        role="user",
        content=question,
    )

    analysis_context = build_conversation_context(
        analysis=analysis_payload,
        question=question,
    )

    result = generate_grounded_answer(
        question=question,
        analysis_context=analysis_context,
        chat_history=format_chat_history(previous_messages),
    )

    sanitized_result = sanitize_answer_result(result)

    save_chat_message(
        session_id=resolved_session_id,
        user_id=user_id,
        role="assistant",
        content=sanitized_result["answer"],
        answer_type=sanitized_result["answer_type"],
        confidence=sanitized_result["confidence"],
        evidence=sanitized_result["evidence"],
        model_used=sanitized_result["model_used"],
        provider=sanitized_result["provider"],
    )

    update_session_activity(resolved_session_id)

    return {
        "status": "success",
        "session_id": resolved_session_id,
        "session_title": session_title,
        "intent": analysis_context.get("intent"),
        **sanitized_result,
    }