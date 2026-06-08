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

    problemId: int | None = None
    problemSetId: int | None = None
    problemNumber: int = Field(..., ge=1)
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


class ProblemSetPayload(BaseModel):
    """API가 반환하는 최종 문제 세트 JSON 구조."""

    userId: int | None = None
    certId: int | None = None
    problemCount: int = Field(..., ge=1)
    problems: list[ProblemItem] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_problem_set(self) -> ProblemSetPayload:
        """문제 수와 문제 번호가 구조와 일치하는지 검증한다."""

        if self.problemCount != len(self.problems):
            raise ValueError("problemCount must match the number of problems.")
        expected_numbers = list(range(1, len(self.problems) + 1))
        actual_numbers = [problem.problemNumber for problem in self.problems]
        if actual_numbers != expected_numbers:
            raise ValueError("problemNumber must start at 1 and increase sequentially.")
        return self


@dataclass(frozen=True)
class GeneratedQuestionSet:
    """생성된 최종 문제 세트, 검수 피드백, 참조 출처를 함께 담는다."""

    certification_name: str
    question_count: int
    problem_set: ProblemSetPayload
    review_feedback: str
    sources: list[Source]
