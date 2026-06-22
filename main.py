"""FastAPI app for certification question generation and algorithm problem generation."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from chat_with_rag import generate_question_for_certification
from llm_client import request_chat_completion, request_json_completion
from rag_models import (
    MultipleChoiceProblemResponse,
    ProblemItem,
    ShortAnswerProblemResponse,
)


app = FastAPI(title="Certification RAG Question Generator")
logger = logging.getLogger(__name__)


@app.middleware("http")
async def log_question_generation_raw_body(request: Request, call_next):
    """Log raw request bodies for the certification question endpoint."""

    if request.method == "POST" and request.url.path == "/questions/generate":
        body = await request.body()
        try:
            body_text = body.decode("utf-8")
        except UnicodeDecodeError:
            body_text = repr(body)
        logger.info("Incoming /questions/generate raw request body: %s", body_text or "<empty>")
    return await call_next(request)


class GenerateQuestionsRequest(BaseModel):
    """Request body for certification question generation."""

    certification: str = Field(..., min_length=1, examples=["example-certification"])
    referenceText: str | None = Field(
        default=None,
        examples=["Generate a question based on this reference text."],
    )
    problemType: Literal["MULTIPLE_CHOICE", "SHORT_ANSWER", "CODING"] = Field(
        ...,
        examples=["MULTIPLE_CHOICE", "SHORT_ANSWER", "CODING"],
    )


def build_multiple_choice_response(problem: ProblemItem) -> MultipleChoiceProblemResponse:
    """Convert a generic problem item into the multiple-choice response shape."""

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
    """Convert a generic problem item into the short-answer response shape."""

    if problem.problemType != "SHORT_ANSWER" or not problem.answer:
        raise RuntimeError("Expected a SHORT_ANSWER problem with a non-empty answer.")

    return ShortAnswerProblemResponse(
        certId=problem.certId,
        title=problem.title,
        question=problem.question,
        answer=problem.answer,
    )


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


@dataclass
class AlgorithmGenerationPlan:
    difficulty: str
    category: str
    intended_algorithm: str
    intended_time_complexity: str
    intended_memory_complexity: str
    bruteforce_time_complexity: str
    max_constraints: str
    chosen_time_limit_ms: int
    chosen_memory_limit_mb: int
    why_bruteforce_fails: str


RESOURCE_BANDS = {
    "EASY": {
        "time_min": 800,
        "time_max": 1500,
        "time_default": 1000,
        "memory_min": 128,
        "memory_max": 256,
        "memory_default": 256,
    },
    "MEDIUM": {
        "time_min": 1000,
        "time_max": 2500,
        "time_default": 2000,
        "memory_min": 128,
        "memory_max": 512,
        "memory_default": 256,
    },
    "HARD": {
        "time_min": 1500,
        "time_max": 4000,
        "time_default": 3000,
        "memory_min": 256,
        "memory_max": 1024,
        "memory_default": 512,
    },
}


ALGORITHM_PLAN_PROMPT = """
You create hidden planning metadata for one algorithm coding problem.
Input fields:
- difficulty: EASY | MEDIUM | HARD
- category: a fixed algorithm category string

Hard rules:
1. category is fixed. Never change it.
2. difficulty changes only the difficulty inside the same category.
3. Choose time and memory limits dynamically for the planned problem.
4. The chosen limits must allow the intended optimal solution.
5. If brute force is not the intended solution, choose limits and constraints so brute force does not pass.
6. Be conservative. If uncertain, choose safer constraints instead of extreme ones.
7. Return one JSON object only.

Difficulty meaning:
- EASY: the main idea is relatively direct, with lighter implementation burden and simpler edge cases.
- MEDIUM: the standard solution must be identified and implemented correctly.
- HARD: the idea, optimization, or edge-case handling is meaningfully harder.

Resource bands:
- EASY: time 800..1500 ms, memory 128..256 MB
- MEDIUM: time 1000..2500 ms, memory 128..512 MB
- HARD: time 1500..4000 ms, memory 256..1024 MB

Output schema:
{
  "category": "copy the input category exactly",
  "difficulty": "copy the input difficulty exactly",
  "intended_algorithm": "short phrase",
  "intended_time_complexity": "e.g. O(N log N)",
  "intended_memory_complexity": "e.g. O(N)",
  "bruteforce_time_complexity": "e.g. O(N^2)",
  "max_constraints": "concise constraint summary such as N up to 200000",
  "chosen_time_limit_ms": 2000,
  "chosen_memory_limit_mb": 256,
  "why_bruteforce_fails": "short explanation"
}
"""


PROBLEM_SPEC_PROMPT = """
You are an algorithm coding problem writer.
You will receive a locked planning JSON. Follow it exactly.

Hard rules:
1. Keep category exactly the same as the plan.
2. Keep time_limit exactly equal to chosen_time_limit_ms from the plan.
3. Keep memory_limit exactly equal to chosen_memory_limit_mb from the plan.
4. The core solving idea must match intended_algorithm and intended_time_complexity from the plan.
5. Do not mention the solution algorithm name directly in the problem statement.
6. All natural-language fields must be written in Korean.
7. Do not include sample input or sample output.
8. Return one JSON object only. No markdown, no code block, no extra text.

Output schema:
{
  "title": "problem title",
  "description": "problem statement",
  "input_description": "input format",
  "output_description": "output format",
  "constraint_text": "constraints only",
  "time_limit": 1000,
  "memory_limit": 256,
  "category": "category string"
}
"""


TESTCASE_PROMPT = """
You generate exact test cases for an algorithm problem.
You will receive final problem spec JSON.

Hard rules:
1. Return one JSON object only with the single key "test_cases".
2. Create exactly 4 test cases.
3. Each test case must have this shape:
   {
     "input_data": "exact input string",
     "expected_output": "exact output string",
     "design_note": "short note"
   }
4. input_data and expected_output must not contain extra leading or trailing blank lines.
5. All four inputs must be different.
6. Include at least one boundary case.
7. Include at least one case that would reject a naive or inefficient approach.
8. Keep the cases valid for the stated constraints.
9. Return JSON only. No markdown, no code block, no extra text.
"""


SOLUTION_PROMPT = """
You write a correct Python reference solution for the given algorithm problem.
Return plain Python code only. No markdown, no code block, no extra explanation.
Read from standard input and write to standard output.
"""


def _parse_json_object(raw: str, label: str) -> dict[str, Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Failed to generate {label}: invalid JSON: {error}") from error
    if not isinstance(data, dict):
        raise RuntimeError(f"Failed to generate {label}: expected a JSON object.")
    return data


def _get_required_text(data: dict[str, Any], field_name: str, label: str) -> str:
    value = str(data.get(field_name, "")).strip()
    if not value:
        raise RuntimeError(f"Failed to generate {label}: missing {field_name}.")
    return value


def _clamp_int(value: Any, minimum: int, maximum: int, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _strip_code_block(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.split("\n")
    inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
    return "\n".join(inner).strip()


def _normalize_generation_plan(
    difficulty: str,
    category: str,
    plan_data: dict[str, Any],
) -> AlgorithmGenerationPlan:
    bands = RESOURCE_BANDS.get(difficulty, RESOURCE_BANDS["MEDIUM"])
    return AlgorithmGenerationPlan(
        difficulty=difficulty,
        category=category,
        intended_algorithm=_get_required_text(plan_data, "intended_algorithm", "algorithm plan"),
        intended_time_complexity=_get_required_text(plan_data, "intended_time_complexity", "algorithm plan"),
        intended_memory_complexity=_get_required_text(plan_data, "intended_memory_complexity", "algorithm plan"),
        bruteforce_time_complexity=_get_required_text(plan_data, "bruteforce_time_complexity", "algorithm plan"),
        max_constraints=_get_required_text(plan_data, "max_constraints", "algorithm plan"),
        chosen_time_limit_ms=_clamp_int(
            plan_data.get("chosen_time_limit_ms"),
            bands["time_min"],
            bands["time_max"],
            bands["time_default"],
        ),
        chosen_memory_limit_mb=_clamp_int(
            plan_data.get("chosen_memory_limit_mb"),
            bands["memory_min"],
            bands["memory_max"],
            bands["memory_default"],
        ),
        why_bruteforce_fails=_get_required_text(plan_data, "why_bruteforce_fails", "algorithm plan"),
    )


def _normalize_problem_spec(
    spec_data: dict[str, Any],
    plan: AlgorithmGenerationPlan,
) -> dict[str, Any]:
    return {
        "title": _get_required_text(spec_data, "title", "problem spec"),
        "description": _get_required_text(spec_data, "description", "problem spec"),
        "input_description": _get_required_text(spec_data, "input_description", "problem spec"),
        "output_description": _get_required_text(spec_data, "output_description", "problem spec"),
        "constraint_text": _get_required_text(spec_data, "constraint_text", "problem spec"),
        "time_limit": plan.chosen_time_limit_ms,
        "memory_limit": plan.chosen_memory_limit_mb,
        "category": plan.category,
    }


def _extract_test_case_list(testcase_data: dict[str, Any]) -> list[dict[str, Any]]:
    test_cases = testcase_data.get("test_cases")
    if not isinstance(test_cases, list):
        test_cases = next((value for value in testcase_data.values() if isinstance(value, list)), None)
    if not isinstance(test_cases, list):
        raise RuntimeError("Failed to generate test cases: missing test_cases list.")
    if len(test_cases) < 4:
        raise RuntimeError("Failed to generate test cases: expected at least 4 cases.")

    normalized_cases: list[dict[str, Any]] = []
    for case in test_cases[:4]:
        if not isinstance(case, dict):
            raise RuntimeError("Failed to generate test cases: each case must be a JSON object.")
        normalized_cases.append(
            {
                "input_data": str(case.get("input_data", "")).strip(),
                "expected_output": str(case.get("expected_output", "")).strip(),
            }
        )

    unique_inputs = {case["input_data"] for case in normalized_cases}
    if len(unique_inputs) != len(normalized_cases):
        raise RuntimeError("Failed to generate test cases: duplicate inputs were returned.")

    return normalized_cases


async def _run_solution(code: str, input_data: str, timeout: float = 5.0) -> str | None:
    """Run a Python reference solution and return its stdout."""

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as file:
            file.write(code)
            tmp_path = file.name

        proc = await asyncio.create_subprocess_exec(
            "python3",
            tmp_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(
                proc.communicate(input_data.encode()),
                timeout=timeout,
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


async def generate_algorithm_problem(difficulty: str, category: str) -> AlgorithmProblem:
    """Generate one algorithm problem through plan, spec, and testcase steps."""

    plan_messages = [
        {"role": "system", "content": ALGORITHM_PLAN_PROMPT},
        {
            "role": "user",
            "content": (
                f"difficulty={difficulty}\n"
                f"category={category}\n\n"
                "Keep category fixed. Choose dynamic limits and constraints that fit the same category."
            ),
        },
    ]
    plan_raw = await request_json_completion(plan_messages, temperature=0.2)
    plan_data = _parse_json_object(plan_raw, "algorithm plan")
    plan = _normalize_generation_plan(difficulty, category, plan_data)

    spec_messages = [
        {"role": "system", "content": PROBLEM_SPEC_PROMPT},
        {
            "role": "user",
            "content": (
                "Locked planning JSON:\n"
                f"{json.dumps(plan.__dict__, ensure_ascii=False, indent=2)}\n\n"
                "Generate the final problem spec JSON. "
                "Keep category, time_limit, and memory_limit exactly aligned with the plan."
            ),
        },
    ]
    spec_raw = await request_json_completion(spec_messages, temperature=0.3)
    spec_data = _parse_json_object(spec_raw, "problem spec")
    normalized_spec = _normalize_problem_spec(spec_data, plan)
    spec_text = json.dumps(normalized_spec, ensure_ascii=False, indent=2)

    tc_messages = [
        {"role": "system", "content": TESTCASE_PROMPT},
        {
            "role": "user",
            "content": (
                "Problem spec JSON:\n"
                f"{spec_text}\n\n"
                "Create exactly 4 test cases. Include one boundary case and one anti-naive case."
            ),
        },
    ]
    tc_raw = await request_json_completion(tc_messages, temperature=0.1)
    tc_data = _parse_json_object(tc_raw, "test cases")
    tc_list = _extract_test_case_list(tc_data)

    solution_messages = [
        {"role": "system", "content": SOLUTION_PROMPT},
        {"role": "user", "content": f"Write a correct Python solution for this problem:\n\n{spec_text}"},
    ]
    solution_raw = await request_chat_completion(solution_messages, temperature=0.0)
    solution_code = _strip_code_block(solution_raw)

    for test_case in tc_list:
        actual = await _run_solution(solution_code, test_case.get("input_data", ""))
        if actual is not None:
            test_case["expected_output"] = actual

    test_cases = [
        AlgorithmTestCase(
            input_data=test_case.get("input_data", "").strip(),
            expected_output=test_case.get("expected_output", "").strip(),
            is_sample=(index < 2),
            case_order=index + 1,
        )
        for index, test_case in enumerate(tc_list)
    ]

    return AlgorithmProblem(
        title=normalized_spec["title"],
        description=normalized_spec["description"],
        input_description=normalized_spec["input_description"],
        output_description=normalized_spec["output_description"],
        constraint_text=normalized_spec["constraint_text"],
        time_limit=normalized_spec["time_limit"],
        memory_limit=normalized_spec["memory_limit"],
        category=normalized_spec["category"],
        test_cases=test_cases,
    )


class GenerateAlgorithmRequest(BaseModel):
    """Request body for algorithm problem generation."""

    difficulty: Literal["EASY", "MEDIUM", "HARD"] = Field(
        default="MEDIUM",
        examples=["EASY", "MEDIUM", "HARD"],
    )
    category: str = Field(
        default="구현",
        examples=["구현", "그래프", "정렬", "이분탐색", "그리디", "문자열"],
    )


class TestCaseResponse(BaseModel):
    """Response shape for one generated test case."""

    input_data: str
    expected_output: str
    is_sample: bool
    case_order: int


class GenerateAlgorithmResponse(BaseModel):
    """Response body for algorithm problem generation."""

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
    """Generate an algorithm problem from difficulty and category."""

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
                input_data=test_case.input_data,
                expected_output=test_case.expected_output,
                is_sample=test_case.is_sample,
                case_order=test_case.case_order,
            )
            for test_case in problem.test_cases
        ],
    )


@app.get("/")
def read_root() -> dict[str, str]:
    """Return a basic health response."""

    return {"message": "Certification RAG Question Generator"}


@app.post(
    "/questions/generate",
    response_model=MultipleChoiceProblemResponse | ShortAnswerProblemResponse,
)
async def generate_questions(
    request: GenerateQuestionsRequest,
) -> MultipleChoiceProblemResponse | ShortAnswerProblemResponse:
    """Generate certification questions through the RAG pipeline."""

    logger.info("Parsed /questions/generate request body: %s", request.model_dump_json())

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
