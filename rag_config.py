"""RAG 서버에서 공통으로 사용하는 환경설정 값을 관리한다."""

from __future__ import annotations

import os

from dotenv import load_dotenv


load_dotenv()

OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "https://gms.ssafy.io/gmsapi/api.openai.com/v1",
)
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_chunks")
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "12"))
DEFAULT_QUESTION_COUNT = int(os.getenv("RAG_QUESTION_COUNT", "20"))


def ensure_api_key() -> str:
    """환경변수에서 API 키를 읽고 없으면 서버 설정 오류를 발생시킨다."""

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    raise RuntimeError("OPENAI_API_KEY environment variable is required.")
