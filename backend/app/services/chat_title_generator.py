import os
import re

from dotenv import load_dotenv
from google import genai


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def clean_chat_title(title: str) -> str:
    title = str(title or "").strip()

    title = title.replace('"', "").replace("'", "")
    title = re.sub(r"^(title\s*:\s*)", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+", " ", title)

    if len(title) > 60:
        title = title[:57].rstrip() + "..."

    return title or "Analysis Conversation"


def generate_fallback_title(question: str) -> str:
    question = str(question or "").strip()

    if not question:
        return "Analysis Conversation"

    cleaned = re.sub(
        r"^(what|which|why|how|where|when|who|can|could|please|show|tell|explain)\s+",
        "",
        question,
        flags=re.IGNORECASE,
    )

    cleaned = cleaned.rstrip("?.!").strip()

    if not cleaned:
        return "Analysis Conversation"

    words = cleaned.split()

    if len(words) > 7:
        cleaned = " ".join(words[:7])

    return clean_chat_title(cleaned.title())


def generate_chat_title(question: str) -> str:
    if not GEMINI_API_KEY:
        return generate_fallback_title(question)

    prompt = f"""
Create a short title for an analytics chat based on the user's first question.

Rules:
- Return only the title.
- Use 2 to 6 words.
- Do not use quotation marks.
- Do not add punctuation at the end.
- Make it clear and professional.
- Do not invent details not contained in the question.

User question:
{question}
"""

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        title = response.text or ""

        return clean_chat_title(title)

    except Exception as error:
        print(f"Chat title generation error: {error}")
        return generate_fallback_title(question)