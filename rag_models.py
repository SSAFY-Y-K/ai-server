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


class ProblemChoice(BaseModel):
    """객관식 보기 한 개를 표현한다."""

    problemId: int | None = None
    choiceNumber: int = Field(..., ge=1, le=4)
    content: str = Field(..., min_length=1)


class ProblemItem(BaseModel):
    """단일 문제를 표현한다."""

    problemType: Literal["MULTIPLE", "SHORT_ANSWER"]
    question: str = Field(..., min_length=1)
    answerCorrectNumber: int | None = Field(default=None, ge=1, le=4)
    answerText: str | None = None
    problemChoices: list[ProblemChoice] | None = None

    @model_validator(mode="after")
    def validate_problem_type_fields(self) -> ProblemItem:
        """문제 유형에 따라 필수/금지 필드를 검증한다."""

        if self.problemType == "MULTIPLE":
            if self.answerCorrectNumber is None:
                raise ValueError("MULTIPLE problems require answerCorrectNumber.")
            if self.answerText is not None:
                raise ValueError("MULTIPLE problems must set answerText to null.")
            if not self.problemChoices:
                raise ValueError("MULTIPLE problems require problemChoices.")
            if len(self.problemChoices) != 4:
                raise ValueError("MULTIPLE problems must contain exactly 4 choices.")
            choice_numbers = [choice.choiceNumber for choice in self.problemChoices]
            if sorted(choice_numbers) != [1, 2, 3, 4]:
                raise ValueError("MULTIPLE problem choices must use choiceNumber 1 through 4.")
            if self.answerCorrectNumber not in choice_numbers:
                raise ValueError("answerCorrectNumber must match one of the choice numbers.")
            return self

        if self.answerCorrectNumber is not None:
            raise ValueError("SHORT_ANSWER problems must set answerCorrectNumber to null.")
        if not self.answerText or not self.answerText.strip():
            raise ValueError("SHORT_ANSWER problems require answerText.")
        if self.problemChoices is not None:
            raise ValueError("SHORT_ANSWER problems must set problemChoices to null.")
        return self


class MultipleChoiceProblemResponse(BaseModel):
    """객관식 단일 문제 응답 구조."""

    question: str = Field(..., min_length=1)
    choice1Content: str = Field(..., min_length=1)
    choice2Content: str = Field(..., min_length=1)
    choice3Content: str = Field(..., min_length=1)
    choice4Content: str = Field(..., min_length=1)
    answerNumber: int = Field(..., ge=1, le=4)


class ShortAnswerProblemResponse(BaseModel):
    """주관식 단일 문제 응답 구조."""

    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)


@dataclass(frozen=True)
class GeneratedQuestion:
    """생성된 단일 문제, 검수 피드백, 참조 출처를 함께 담는다."""

    certification_name: str
    problem: ProblemItem
    review_feedback: str
    sources: list[Source]
