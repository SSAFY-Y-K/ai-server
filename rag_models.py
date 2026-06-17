"""RAG 문제 생성 파이프라인에서 공유하는 데이터 구조."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, model_validator


@dataclass(frozen=True)
class Source:
    """RAG 검색에 사용된 원본 문서 위치를 표현한다."""

    source: str
    page: int | str
    category: str


class ProblemItem(BaseModel):
    """단일 문제를 표현한다."""

    problemType: Literal["MULTIPLE", "SHORT_ANSWER"]
    certId: int | None = None
    title: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    choice1Content: str | None = None
    choice2Content: str | None = None
    choice3Content: str | None = None
    choice4Content: str | None = None
    answerNumber: int | None = Field(default=None, ge=1, le=4)
    answer: str | None = None

    @model_validator(mode="after")
    def validate_problem_type_fields(self) -> ProblemItem:
        """문제 유형에 따라 필수/금지 필드를 검증한다."""

        if self.problemType == "MULTIPLE":
            if self.answerNumber is None:
                raise ValueError("MULTIPLE problems require answerNumber.")
            choices = [
                self.choice1Content,
                self.choice2Content,
                self.choice3Content,
                self.choice4Content,
            ]
            if any(not choice or not choice.strip() for choice in choices):
                raise ValueError("MULTIPLE problems require choice1Content through choice4Content.")
            if self.answer is not None:
                raise ValueError("MULTIPLE problems must not include answer.")
            return self

        if not self.answer or not self.answer.strip():
            raise ValueError("SHORT_ANSWER problems require answer.")
        if self.answerNumber is not None:
            raise ValueError("SHORT_ANSWER problems must not include answerNumber.")
        if any(
            choice is not None
            for choice in [
                self.choice1Content,
                self.choice2Content,
                self.choice3Content,
                self.choice4Content,
            ]
        ):
            raise ValueError("SHORT_ANSWER problems must not include choices.")
        return self


class MultipleChoiceProblemResponse(BaseModel):
    """객관식 단일 문제 응답 구조."""

    certId: int | None = None
    title: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    choice1Content: str = Field(..., min_length=1)
    choice2Content: str = Field(..., min_length=1)
    choice3Content: str = Field(..., min_length=1)
    choice4Content: str = Field(..., min_length=1)
    answerNumber: int = Field(..., ge=1, le=4)


class ShortAnswerProblemResponse(BaseModel):
    """주관식 단일 문제 응답 구조."""

    certId: int | None = None
    title: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)


@dataclass(frozen=True)
class GeneratedQuestion:
    """생성된 단일 문제, 검수 피드백, 참조 출처를 함께 담는다."""

    certification_name: str
    problem: ProblemItem
    review_feedback: str
    sources: list[Source]
