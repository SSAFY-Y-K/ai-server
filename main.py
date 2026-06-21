"""FastAPI 엔드포인트를 정의해 자격증별 RAG 문제 생성 및 알고리즘 문제 생성을 제공한다."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from chat_with_rag import generate_question_for_certification
from llm_client import request_chat_completion, request_json_completion
from rag_models import (
    MultipleChoiceProblemResponse,
    ProblemItem,
    ShortAnswerProblemResponse,
)


app = FastAPI(title="Certification RAG Question Generator")


class GenerateQuestionsRequest(BaseModel):
    """문제 생성 API가 받는 요청 본문 구조."""

    certification: str = Field(..., min_length=1, examples=["정보처리기사"])
    referenceText: str | None = Field(
        default=None,
        examples=["운영체제의 프로세스 스케줄링과 교착 상태 개념을 참고해 문제를 생성해 주세요."],
    )
    problemType: Literal["MULTIPLE_CHOICE", "SHORT_ANSWER", "CODING"] = Field(
        ...,
        examples=["MULTIPLE_CHOICE", "SHORT_ANSWER", "CODING"],
    )


def build_multiple_choice_response(problem: ProblemItem) -> MultipleChoiceProblemResponse:
    """내부 단일 문제 구조를 외부 객관식 단일 문제 응답 구조로 변환한다."""

    if problem.problemType != "MULTIPLE" or problem.answerNumber is None:
        raise RuntimeError("Expected a MULTIPLE problem.")
    if not all(
        [
            problem.choice1Content,
            problem.choice2Content,
            problem.choice3Content,
            problem.choice4Content,
        ]
    ):
        raise RuntimeError("Expected a MULTIPLE problem with four choices.")

    return MultipleChoiceProblemResponse(
        certId=problem.certId,
        title=problem.title,
        question=problem.question,
        choice1Content=problem.choice1Content,
        choice2Content=problem.choice2Content,
        choice3Content=problem.choice3Content,
        choice4Content=problem.choice4Content,
        answerNumber=problem.answerNumber,
    )


def build_short_answer_response(problem: ProblemItem) -> ShortAnswerProblemResponse:
    """내부 단일 문제 구조를 외부 주관식 단일 문제 응답 구조로 변환한다."""

    if problem.problemType != "SHORT_ANSWER" or not problem.answer:
        raise RuntimeError("Expected a SHORT_ANSWER problem with a non-empty answer.")

    return ShortAnswerProblemResponse(
        certId=problem.certId,
        title=problem.title,
        question=problem.question,
        answer=problem.answer,
    )


# ── 알고리즘 문제 생성 ────────────────────────────────────────────────────────

@dataclass
class AlgorithmTestCase:
    input_data: str
    expected_output: str
    is_sample: bool
    case_order: int


@dataclass
class AlgorithmProblem:
    title: str
    description: str
    input_description: str
    output_description: str
    constraint_text: str
    time_limit: int
    memory_limit: int
    category: str
    test_cases: list[AlgorithmTestCase] = field(default_factory=list)


DIFFICULTY_LABEL = {"EASY": "초급", "MEDIUM": "중급", "HARD": "상급"}

# 3단계: 참조 솔루션 생성
SOLUTION_PROMPT = """너는 알고리즘 문제 풀이 전문가다.
주어진 문제 명세에 대한 정확한 Python 솔루션을 작성해라.
코드블록, 마크다운, 설명 없이 순수 Python 코드만 출력한다.
표준 입력(input())으로 읽고 print()로 결과를 출력한다."""


async def _run_solution(code: str, input_data: str, timeout: float = 5.0) -> str | None:
    """참조 솔루션을 실제로 실행해 출력값을 반환한다. 실패 시 None."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp_path = f.name

        proc = await asyncio.create_subprocess_exec(
            "python3", tmp_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(
                proc.communicate(input_data.encode()), timeout=timeout
            )
            if proc.returncode == 0:
                return stdout.decode().strip()
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
    except Exception:
        pass
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return None

# 1단계: 문제 설명만 생성 (테스트케이스 제외)
PROBLEM_SPEC_PROMPT = """너는 알고리즘 코딩 문제 출제 전문가다.
주어진 난이도와 카테고리에 맞는 알고리즘 문제 1개의 명세를 아래 JSON 형식으로 생성해라.
테스트케이스는 포함하지 않는다. 코드블록, 마크다운, 설명 없이 유효한 JSON 객체 하나만 출력한다.

{{
  "title": "문제 제목 (한국어, 간결하게)",
  "description": "문제 설명 (한국어, 상세하게. 입출력 예시 없이 순수 설명만)",
  "input_description": "입력 형식 설명 (줄 수, 각 줄의 의미, 공백 구분 등 정확히)",
  "output_description": "출력 형식 설명 (정확히 어떤 값을 어떤 순서로 출력하는지)",
  "constraint_text": "제약 조건 (예: 1 <= N <= 10000)",
  "time_limit": 1000,
  "memory_limit": 256,
  "category": "{category}"
}}

규칙:
- time_limit 단위는 밀리초(ms), memory_limit 단위는 MB
- 모든 텍스트는 한국어로 작성
- 입력/출력 형식은 구체적이고 모호하지 않게 작성"""

# 2단계: 문제 명세를 주고 테스트케이스를 단계별로 검증하면서 생성
TESTCASE_PROMPT = """너는 알고리즘 문제의 테스트케이스를 검증하며 생성하는 전문가다.
아래 문제 명세를 보고 테스트케이스 4개(샘플 2 + 히든 2)를 생성해라.

문제 명세:
{spec}

각 테스트케이스에 대해 반드시 다음 순서를 따라라:
1. 입력값을 결정한다
2. 해당 입력에 대해 알고리즘을 손으로 단계별로 실행한다 (머릿속으로만 하지 말고 trace)
3. 최종 출력값을 확정한다

그 후 아래 JSON 형식으로만 출력한다. 코드블록, 마크다운, 설명 없이 유효한 JSON 배열만:

[
  {{"input_data": "입력1", "expected_output": "출력1", "trace": "단계별 실행 과정"}},
  {{"input_data": "입력2", "expected_output": "출력2", "trace": "단계별 실행 과정"}},
  {{"input_data": "입력3", "expected_output": "출력3", "trace": "단계별 실행 과정"}},
  {{"input_data": "입력4", "expected_output": "출력4", "trace": "단계별 실행 과정"}}
]

규칙:
- 반드시 4개의 테스트케이스를 생성
- input_data와 expected_output의 끝에 불필요한 공백/개행 없이 정확하게 작성
- 각 케이스는 서로 다른 입력값 사용
- trace 필드는 출력값이 맞는지 확인용이므로 반드시 작성"""


async def generate_algorithm_problem(difficulty: str, category: str) -> AlgorithmProblem:
    """2단계로 LLM을 호출해 문제 명세 생성 → 테스트케이스 검증 생성."""

    difficulty_label = DIFFICULTY_LABEL.get(difficulty, "중급")

    # 1단계: 문제 명세 생성
    spec_messages = [
        {"role": "system", "content": PROBLEM_SPEC_PROMPT.format(category=category)},
        {
            "role": "user",
            "content": (
                f"난이도: {difficulty_label} ({difficulty})\n"
                f"카테고리: {category}\n\n"
                "위 조건에 맞는 알고리즘 문제 명세를 JSON으로 생성해라."
            ),
        },
    ]

    spec_raw = await request_json_completion(spec_messages, temperature=0.7)

    try:
        spec_data = json.loads(spec_raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"문제 명세 생성 실패 - 유효하지 않은 JSON: {e}") from e

    # 2단계: 테스트케이스 입력/출력 생성 (AI 산술 오류 가능 — 3단계에서 검증)
    spec_text = json.dumps(spec_data, ensure_ascii=False, indent=2)
    tc_messages = [
        {
            "role": "system",
            "content": TESTCASE_PROMPT.format(spec=spec_text),
        },
        {
            "role": "user",
            "content": "위 문제 명세에 맞는 테스트케이스 4개를 단계별 trace와 함께 JSON 배열로 생성해라.",
        },
    ]

    tc_raw = await request_chat_completion(tc_messages, temperature=0.2)

    tc_stripped = tc_raw.strip()
    if tc_stripped.startswith("```"):
        lines = tc_stripped.split("\n")
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        tc_stripped = "\n".join(inner).strip()

    try:
        tc_data = json.loads(tc_stripped)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"테스트케이스 생성 실패 - 유효하지 않은 JSON: {e}\n원본: {tc_raw[:200]}") from e

    if isinstance(tc_data, list):
        tc_list = tc_data
    elif isinstance(tc_data, dict):
        tc_list = next((v for v in tc_data.values() if isinstance(v, list)), [])
    else:
        tc_list = []

    # 3단계: 참조 솔루션 생성 후 실제 실행해 expected_output 검증·수정
    solution_messages = [
        {"role": "system", "content": SOLUTION_PROMPT},
        {"role": "user", "content": f"다음 문제의 Python 솔루션을 작성해라:\n\n{spec_text}"},
    ]
    solution_raw = await request_chat_completion(solution_messages, temperature=0.0)

    solution_code = solution_raw.strip()
    if solution_code.startswith("```"):
        lines = solution_code.split("\n")
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        solution_code = "\n".join(inner).strip()

    for tc in tc_list:
        actual = await _run_solution(solution_code, tc.get("input_data", ""))
        if actual is not None:
            tc["expected_output"] = actual

    # is_sample을 모델 응답에 맡기지 않고 순서로 직접 할당 (앞 2개 샘플, 나머지 히든)
    test_cases = [
        AlgorithmTestCase(
            input_data=tc.get("input_data", "").strip(),
            expected_output=tc.get("expected_output", "").strip(),
            is_sample=(i < 2),
            case_order=i + 1,
        )
        for i, tc in enumerate(tc_list)
    ]

    return AlgorithmProblem(
        title=spec_data.get("title", ""),
        description=spec_data.get("description", ""),
        input_description=spec_data.get("input_description", ""),
        output_description=spec_data.get("output_description", ""),
        constraint_text=spec_data.get("constraint_text", ""),
        time_limit=int(spec_data.get("time_limit", 1000)),
        memory_limit=int(spec_data.get("memory_limit", 256)),
        category=spec_data.get("category", category),
        test_cases=test_cases,
    )


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
            reference_text=request.referenceText,
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
