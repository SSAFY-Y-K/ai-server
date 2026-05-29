"""RAG 검색, 문제 생성, 검수, 수정을 하나의 문제 생성 파이프라인으로 조립한다."""

from __future__ import annotations

from llm_client import request_chat_completion
from question_prompts import (
    build_generation_messages,
    build_review_messages,
    build_revision_messages,
)
from rag_config import DEFAULT_QUESTION_COUNT, DEFAULT_TOP_K
from rag_models import GeneratedQuestionSet
from rag_retriever import collect_sources, format_context, retrieve_certification_docs


async def generate_questions_for_certification(
    certification_name: str,
    *,
    question_count: int = DEFAULT_QUESTION_COUNT,
    top_k: int = DEFAULT_TOP_K,
) -> GeneratedQuestionSet:
    """자격증 이름을 입력받아 RAG 검색 후 최종 문제 세트를 생성한다."""

    certification_name = certification_name.strip()
    if not certification_name:
        raise ValueError("certification_name is required.")
    if question_count <= 0:
        raise ValueError("question_count must be greater than 0.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    docs = await retrieve_certification_docs(certification_name, top_k=top_k)
    has_context = bool(docs)
    context = format_context(docs) or "이 자격증과 일치하는 RAG 참고자료가 없습니다."
    draft = await request_chat_completion(
        build_generation_messages(
            certification_name,
            question_count,
            context,
            has_context=has_context,
        ),
        temperature=0.4,
    )
    review_feedback = await request_chat_completion(
        build_review_messages(
            certification_name,
            question_count,
            context,
            draft,
            has_context=has_context,
        ),
        temperature=0.1,
    )
    content = await request_chat_completion(
        build_revision_messages(
            certification_name,
            question_count,
            context,
            draft,
            review_feedback,
            has_context=has_context,
        ),
        temperature=0.2,
    )
    return GeneratedQuestionSet(
        certification_name=certification_name,
        question_count=question_count,
        content=content,
        review_feedback=review_feedback,
        sources=collect_sources(docs),
    )
