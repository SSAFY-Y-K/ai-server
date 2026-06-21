"""RAG 검색, 문제 생성, 검수, 수정을 하나의 문제 생성 파이프라인으로 조립한다."""

from __future__ import annotations

import json
from typing import Literal

from llm_client import request_chat_completion
from pydantic import ValidationError
from question_prompts import (
    build_generation_messages,
    build_review_messages,
    build_revision_messages,
)
from rag_config import DEFAULT_TOP_K
from rag_models import GeneratedQuestion, ProblemItem
from rag_retriever import collect_sources, format_context, retrieve_certification_docs


def parse_problem(raw_content: str, problem_type: Literal["MULTIPLE", "SHORT_ANSWER"]) -> ProblemItem:
    """LLM 응답에서 JSON 객체를 추출해 단일 문제로 검증한다."""

    content = raw_content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or start >= end:
        raise RuntimeError("LLM did not return a valid JSON object.")

    json_text = content[start : end + 1]
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"LLM returned invalid JSON: {error}") from error

    if not isinstance(payload, dict):
        raise RuntimeError("LLM returned a JSON value that is not an object.")
    payload.setdefault("problemType", problem_type)

    try:
        return ProblemItem.model_validate(payload)
    except ValidationError as error:
        raise RuntimeError(f"LLM returned an invalid problem payload: {error}") from error


async def generate_question_for_certification(
    certification_name: str,
    *,
    reference_text: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    problem_type: Literal["MULTIPLE", "SHORT_ANSWER"],
) -> GeneratedQuestion:
    """자격증 이름을 입력받아 RAG 검색 후 단일 문제를 생성한다."""

    certification_name = certification_name.strip()
    reference_text = reference_text.strip() if reference_text else None
    if not certification_name:
        raise ValueError("certification_name is required.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")
    if problem_type not in {"MULTIPLE", "SHORT_ANSWER"}:
        raise ValueError("problem_type must be MULTIPLE or SHORT_ANSWER.")

    docs = await retrieve_certification_docs(certification_name, top_k=top_k)
    has_context = bool(docs)
    context = format_context(docs) or "이 자격증과 일치하는 RAG 참고자료가 없습니다."
    draft = await request_chat_completion(
        build_generation_messages(
            certification_name,
            context,
            problem_type,
            has_context=has_context,
            reference_text=reference_text,
        ),
        temperature=0.4,
    )
    review_feedback = await request_chat_completion(
        build_review_messages(
            certification_name,
            context,
            draft,
            problem_type,
            has_context=has_context,
            reference_text=reference_text,
        ),
        temperature=0.1,
    )
    final_content = await request_chat_completion(
        build_revision_messages(
            certification_name,
            context,
            draft,
            review_feedback,
            problem_type,
            has_context=has_context,
            reference_text=reference_text,
        ),
        temperature=0.2,
    )
    problem = parse_problem(final_content, problem_type)
    if problem.problemType != problem_type:
        raise RuntimeError("LLM returned a problemType that does not match the requested type.")

    return GeneratedQuestion(
        certification_name=certification_name,
        problem=problem,
        review_feedback=review_feedback,
        sources=collect_sources(docs),
    )
