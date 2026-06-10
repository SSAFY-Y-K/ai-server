"""FastAPI 엔드포인트를 정의해 자격증별 RAG 문제 생성 및 알고리즘 문제 생성을 제공한다."""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from chat_with_rag import generate_question_for_certification
from rag_models import (
    MultipleChoiceProblemResponse,
    ProblemItem,
    ShortAnswerProblemResponse,
)


app = FastAPI(title="Certification RAG Question Generator")


class GenerateQuestionsRequest(BaseModel):
    """문제 생성 API가 받는 요청 본문 구조."""

    certification: str = Field(..., min_length=1, examples=["정보처리기사"])
    problemType: Literal["MULTIPLE_CHOICE", "SHORT_ANSWER", "CODING"] = Field(
        ...,
        examples=["MULTIPLE_CHOICE", "SHORT_ANSWER", "CODING"],
    )


def build_multiple_choice_response(problem: ProblemItem) -> MultipleChoiceProblemResponse:
    """내부 단일 문제 구조를 외부 객관식 단일 문제 응답 구조로 변환한다."""

    if problem.problemType != "MULTIPLE" or not problem.problemChoices:
        raise RuntimeError("Expected a MULTIPLE problem with four choices.")

    choices = {choice.choiceNumber: choice.content for choice in problem.problemChoices}
    required_numbers = {1, 2, 3, 4}
    if set(choices) != required_numbers:
        raise RuntimeError("Expected MULTIPLE problem choices numbered 1 through 4.")

    return MultipleChoiceProblemResponse(
        question=problem.question,
        choice1Content=choices[1],
        choice2Content=choices[2],
        choice3Content=choices[3],
        choice4Content=choices[4],
        answerNumber=problem.answerCorrectNumber,
    )


def build_short_answer_response(problem: ProblemItem) -> ShortAnswerProblemResponse:
    """내부 단일 문제 구조를 외부 주관식 단일 문제 응답 구조로 변환한다."""

    if problem.problemType != "SHORT_ANSWER" or not problem.answerText:
        raise RuntimeError("Expected a SHORT_ANSWER problem with a non-empty answer.")

    return ShortAnswerProblemResponse(
        question=problem.question,
        answer=problem.answerText,
    )


# ── 알고리즘 문제 생성 ────────────────────────────────────────────────────────

class GenerateAlgorithmRequest(BaseModel):
    """알고리즘 문제 생성 API가 받는 요청 본문 구조."""

    difficulty: Literal["EASY", "MEDIUM", "HARD"] = Field(
        default="MEDIUM", examples=["EASY", "MEDIUM", "HARD"]
    )
    category: str = Field(default="구현", examples=["dp", "graph", "구현", "정렬", "이분탐색"])


class TestCaseResponse(BaseModel):
    """생성된 테스트케이스 한 건 (DB test_cases 테이블 대응)."""

    input_data: str
    expected_output: str
    is_sample: bool
    case_order: int


class GenerateAlgorithmResponse(BaseModel):
    """알고리즘 문제 생성 API가 반환하는 응답 본문 구조 (DB problems 테이블 대응)."""

    title: str
    description: str
    input_description: str
    output_description: str
    constraint_text: str
    time_limit: int
    memory_limit: int
    category: str
    test_cases: list[TestCaseResponse]


@app.post("/algorithm/generate", response_model=GenerateAlgorithmResponse)
async def generate_algorithm(request: GenerateAlgorithmRequest) -> GenerateAlgorithmResponse:
    """난이도와 카테고리를 받아 알고리즘 문제와 테스트케이스 세트를 생성한다."""

    try:
        problem = await generate_algorithm_problem(
            difficulty=request.difficulty,
            category=request.category,
        )
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    return GenerateAlgorithmResponse(
        title=problem.title,
        description=problem.description,
        input_description=problem.input_description,
        output_description=problem.output_description,
        constraint_text=problem.constraint_text,
        time_limit=problem.time_limit,
        memory_limit=problem.memory_limit,
        category=problem.category,
        test_cases=[
            TestCaseResponse(
                input_data=tc.input_data,
                expected_output=tc.expected_output,
                is_sample=tc.is_sample,
                case_order=tc.case_order,
            )
            for tc in problem.test_cases
        ],
    )


# ── 자격증 RAG 문제 생성 ──────────────────────────────────────────────────────

@app.get("/")
def read_root() -> dict[str, str]:
    """서버 상태 확인용 기본 응답을 반환한다."""

    return {"message": "Certification RAG Question Generator"}


@app.post(
    "/questions/generate",
    response_model=MultipleChoiceProblemResponse | ShortAnswerProblemResponse,
)
async def generate_questions(
    request: GenerateQuestionsRequest,
) -> MultipleChoiceProblemResponse | ShortAnswerProblemResponse:
    """자격증 이름을 받아 RAG 기반 문제 세트를 생성한다."""

    try:
        if request.problemType == "CODING":
            raise ValueError(
                "CODING problemType is not supported by /questions/generate. "
                "This endpoint currently supports only MULTIPLE_CHOICE and SHORT_ANSWER."
            )

        result = await generate_question_for_certification(
            request.certification,
            problem_type=(
                "MULTIPLE" if request.problemType == "MULTIPLE_CHOICE" else "SHORT_ANSWER"
            ),
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    if request.problemType == "MULTIPLE_CHOICE":
        return build_multiple_choice_response(result.problem)

    return build_short_answer_response(result.problem)
