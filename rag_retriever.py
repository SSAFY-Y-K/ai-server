"""Chroma에서 자격증 관련 RAG 문서를 검색하고 프롬프트용 컨텍스트를 만든다."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag_config import CHROMA_COLLECTION, CHROMA_DIR, DEFAULT_TOP_K
from rag_embeddings import build_embedding_function
from rag_models import Source


def build_vector_db() -> Chroma:
    """로컬 Chroma 컬렉션에 연결한다."""

    return Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=CHROMA_COLLECTION,
        embedding_function=build_embedding_function(),
    )


async def retrieve_certification_docs(
    certification_name: str,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> list[Document]:
    """자격증 이름과 일치하는 syllabus와 기출 청크만 검색한다."""

    vector_db = build_vector_db()
    syllabus_query = f"{certification_name} 과목 출제범위 핵심 개념 학습 내용"
    exam_query = f"{certification_name} 기출문제 핵심 개념 문제 보기 정답 해설"
    try:
        syllabus_docs = await vector_db.asimilarity_search(
            syllabus_query,
            k=max(1, min(4, top_k)),
            filter={"$and": [{"category": certification_name}, {"doc_type": "syllabus"}]},
        )
        exam_docs = await vector_db.asimilarity_search(
            exam_query,
            k=top_k,
            filter={"$and": [{"category": certification_name}, {"doc_type": "exam_pdf"}]},
        )
    except Exception as error:
        raise RuntimeError(
            "Chroma 검색에 실패했습니다. chroma_db가 생성되어 있는지, "
            "벡터 DB 생성 시 사용한 임베딩과 현재 설정이 일치하는지 확인하세요."
        ) from error

    return unique_documents([*syllabus_docs, *exam_docs])


def unique_documents(docs: Sequence[Document]) -> list[Document]:
    """source/page/chunk_index 기준으로 중복 문서를 제거한다."""

    unique: list[Document] = []
    seen: set[tuple[str, str, str]] = set()
    for doc in docs:
        metadata = doc.metadata or {}
        key = (
            str(metadata.get("source_path") or metadata.get("source") or ""),
            str(metadata.get("page") or ""),
            str(metadata.get("chunk_index") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def format_context(docs: Sequence[Document]) -> str:
    """검색된 문서를 LLM 프롬프트에 넣기 좋은 컨텍스트 문자열로 바꾼다."""

    context_parts: list[str] = []
    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        source = metadata.get("source") or metadata.get("source_path") or "unknown"
        page = metadata.get("page", "unknown")
        category = metadata.get("category", "")
        doc_type = metadata.get("doc_type", "exam_pdf")
        label = f"[{index}] source={source}, page={page}"
        if category:
            label += f", category={category}"
        label += f", doc_type={doc_type}"
        context_parts.append(f"{label}\n{doc.page_content}")
    return "\n\n".join(context_parts)


def collect_sources(docs: Sequence[Document]) -> list[Source]:
    """검색된 문서들의 중복 없는 출처 목록을 만든다."""

    seen: set[tuple[str, str, str]] = set()
    sources: list[Source] = []
    for doc in docs:
        metadata = doc.metadata or {}
        source = str(metadata.get("source") or metadata.get("source_path") or "unknown")
        page = metadata.get("page", "unknown")
        category = str(metadata.get("category") or "")
        key = (source, str(page), category)
        if key in seen:
            continue
        seen.add(key)
        sources.append(Source(source=source, page=page, category=category))
    return sources
