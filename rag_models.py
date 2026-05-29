"""RAG 문제 생성 파이프라인에서 공유하는 데이터 구조."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    """RAG 검색에 사용된 원본 문서 위치를 표현한다."""

    source: str
    page: int | str
    category: str


@dataclass(frozen=True)
class GeneratedQuestionSet:
    """생성된 최종 문제 세트, 검수 피드백, 참조 출처를 함께 담는다."""

    certification_name: str
    question_count: int
    content: str
    review_feedback: str
    sources: list[Source]
