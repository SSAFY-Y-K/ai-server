"""Chroma 컬렉션과 호환되는 검색용 임베딩 함수를 만든다."""

from __future__ import annotations

import os
import sqlite3

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from rag_chroma_store import DEFAULT_EMBED_DIM, hash_embedding
from rag_config import CHROMA_DIR, EMBEDDING_MODEL, OPENAI_BASE_URL, ensure_api_key


class LocalHashEmbeddings(Embeddings):
    """Chroma DB가 로컬 해시 임베딩으로 만들어진 경우 검색 벡터를 생성한다."""

    def __init__(self, dimension: int = DEFAULT_EMBED_DIM) -> None:
        """해시 임베딩 차원을 저장한다."""

        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """여러 문서를 Chroma에 넣을 수 있는 벡터 목록으로 변환한다."""

        return [hash_embedding(text, self.dimension) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """사용자 질문을 검색용 벡터로 변환한다."""

        return hash_embedding(text, self.dimension)


def detect_collection_embedding() -> str | None:
    """Chroma 컬렉션 메타데이터에서 저장 당시 임베딩 방식을 확인한다."""

    sqlite_path = os.path.join(CHROMA_DIR, "chroma.sqlite3")
    if not os.path.exists(sqlite_path):
        return None

    conn = sqlite3.connect(sqlite_path)
    try:
        row = conn.execute(
            "select str_value from collection_metadata where key = 'embedding'"
        ).fetchone()
    finally:
        conn.close()
    return str(row[0]) if row and row[0] else None


def build_embedding_function() -> Embeddings:
    """저장된 DB와 호환되는 검색용 임베딩 함수를 만든다."""

    embedding_kind = detect_collection_embedding()
    if embedding_kind and embedding_kind.startswith("local_hash_"):
        dimension = int(embedding_kind.removeprefix("local_hash_"))
        return LocalHashEmbeddings(dimension)

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=ensure_api_key(),
        base_url=OPENAI_BASE_URL,
    )
