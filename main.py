"""FastAPI app for certification question generation and algorithm problem generation."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import time
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
    input_model: str
    intended_algorithm: str
    intended_time_complexity: str
    intended_memory_complexity: str
    opt_tc: str
    bf_tc: str
    max_constraints: str
    max_n: int | None
    max_q: int | None
    max_v: int | None
    max_e: int | None
    max_r: int | None
    max_c: int | None
    peak_state_items: int | None
    chosen_time_limit_ms: int
    chosen_memory_limit_mb: int
    why_bruteforce_fails: str


VALID_CATEGORIES = {"구현", "dp", "graph", "정렬", "이분탐색", "greedy", "bfs", "string"}
VALID_INPUT_MODELS = {"ARRAY", "GRID", "GRAPH", "STRING", "INTERVAL", "ANSWER_SEARCH"}
VALID_COMPLEXITY_CODES = {
    "LINEAR",
    "LOG_N",
    "N_LOG_N",
    "N2",
    "Q_LOG_N",
    "V_PLUS_E",
    "E_LOG_V",
    "RC",
    "RC_LOG_RC",
}
PLAN_RETRY_LIMIT = 3
TESTCASE_RETRY_LIMIT = 5
OPERATIONS_PER_MS = 5_000.0
MEMORY_BYTES_PER_ITEM = 32
TESTCASE_NEAR_MAX_RATIO = {"EASY": 0.4, "MEDIUM": 0.6, "HARD": 0.75}
TESTCASE_ANTI_NAIVE_RATIO = {"EASY": 0.2, "MEDIUM": 0.3, "HARD": 0.4}
LARGE_ANTI_NAIVE_CATEGORIES = {"dp", "graph", "정렬", "이분탐색", "greedy", "bfs", "string"}
LITERAL_INPUT_CHAR_LIMIT = 1500

CATEGORY_ALIASES = {
    "구현": "구현",
    "IMPLEMENTATION": "구현",
    "IMPL": "구현",
    "DP": "dp",
    "DYNAMICPROGRAMMING": "dp",
    "GRAPH": "graph",
    "GRAPHS": "graph",
    "정렬": "정렬",
    "SORT": "정렬",
    "SORTING": "정렬",
    "이분탐색": "이분탐색",
    "BINARYSEARCH": "이분탐색",
    "PARAMETRICSEARCH": "이분탐색",
    "GREEDY": "greedy",
    "BFS": "bfs",
    "BFSDFS": "bfs",
    "DFS": "bfs",
    "STRING": "string",
    "STRINGS": "string",
}

INPUT_MODEL_ALIASES = {
    "ARRAY": "ARRAY",
    "LIST": "ARRAY",
    "SEQUENCE": "ARRAY",
    "GRID": "GRID",
    "MATRIX": "GRID",
    "BOARD": "GRID",
    "GRAPH": "GRAPH",
    "STRING": "STRING",
    "TEXT": "STRING",
    "INTERVAL": "INTERVAL",
    "RANGE": "INTERVAL",
    "ANSWERSEARCH": "ANSWER_SEARCH",
    "PARAMETRICSEARCH": "ANSWER_SEARCH",
    "BINARYSEARCHONANSWER": "ANSWER_SEARCH",
}

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

CATEGORY_PROFILES = {
    "구현": {
        "input_models": {"ARRAY", "GRID", "STRING"},
        "opt_tc": {"LINEAR", "N_LOG_N", "RC"},
        "required_any": (("max_n",), ("max_r", "max_c")),
    },
    "dp": {
        "input_models": {"ARRAY", "GRID"},
        "opt_tc": {"LINEAR", "N_LOG_N", "N2", "RC"},
        "required_any": (("max_n",), ("max_r", "max_c")),
    },
    "graph": {
        "input_models": {"GRAPH"},
        "opt_tc": {"V_PLUS_E", "E_LOG_V"},
        "required_any": (("max_v", "max_e"),),
    },
    "정렬": {
        "input_models": {"ARRAY"},
        "opt_tc": {"N_LOG_N"},
        "required_any": (("max_n",),),
    },
    "이분탐색": {
        "input_models": {"ARRAY", "ANSWER_SEARCH"},
        "opt_tc": {"LOG_N", "N_LOG_N", "Q_LOG_N"},
        "required_any": (("max_n",),),
    },
    "greedy": {
        "input_models": {"ARRAY", "INTERVAL"},
        "opt_tc": {"LINEAR", "N_LOG_N"},
        "required_any": (("max_n",),),
    },
    "bfs": {
        "input_models": {"GRAPH", "GRID"},
        "opt_tc": {"V_PLUS_E", "RC"},
        "required_any": (("max_v", "max_e"), ("max_r", "max_c")),
    },
    "string": {
        "input_models": {"STRING"},
        "opt_tc": {"LINEAR", "N_LOG_N"},
        "required_any": (("max_n",),),
    },
}


ALGORITHM_PLAN_PROMPT = """
You create hidden planning metadata for one algorithm coding problem.
The incoming category is fixed and must stay exact.
Allowed category strings are:
- 구현
- dp
- graph
- 정렬
- 이분탐색
- greedy
- bfs
- string

Hard rules:
1. Copy category exactly. Do not translate or rename it.
2. Copy difficulty exactly.
3. difficulty changes only the difficulty inside the same category.
4. Choose time and memory limits dynamically inside the allowed resource band.
5. The intended optimal solution must pass within the chosen limits.
6. The brute-force approach must not be comfortably feasible.
7. Return one JSON object only.
8. Use null for unused numeric fields.

Difficulty meaning:
- EASY: direct main idea, lighter implementation, simpler edge cases.
- MEDIUM: standard solution must be identified and implemented correctly.
- HARD: harder idea, optimization, or edge-case handling.

Resource bands:
- EASY: time 800..1500 ms, memory 128..256 MB
- MEDIUM: time 1000..2500 ms, memory 128..512 MB
- HARD: time 1500..4000 ms, memory 256..1024 MB

Allowed input_model values:
- ARRAY        # 배열/수열
- GRID         # 2차원 격자/행렬
- GRAPH        # 그래프 (정점·간선)
- STRING       # 문자열
- INTERVAL     # 구간
- ANSWER_SEARCH  # 이분탐색으로 정답을 탐색하는 형태

Allowed complexity code values for opt_tc and bf_tc:
- LINEAR      # O(N) 또는 O(V+E) 선형
- LOG_N       # O(log N)
- N_LOG_N     # O(N log N)
- N2          # O(N²)
- Q_LOG_N     # O(Q log N) — 쿼리 기반
- V_PLUS_E    # O(V+E) — 그래프 순회
- E_LOG_V     # O(E log V) — 다익스트라 등
- RC          # O(R×C) — 격자 전체 순회
- RC_LOG_RC   # O(R×C × log(R×C))

Category-specific requirements:
- 구현: use ARRAY, GRID, or STRING. Fill max_n for ARRAY/STRING or max_r and max_c for GRID.
  # ARRAY/STRING이면 max_n, GRID이면 max_r·max_c를 채운다.
- dp: use ARRAY or GRID. Fill max_n for ARRAY or max_r and max_c for GRID.
  # ARRAY면 max_n, GRID면 max_r·max_c를 채운다.
  N2 dp: max_n must be ≤ 5000 (N²=25M fits in time). LINEAR/N_LOG_N dp: max_n up to 200000.
  # N2 DP는 max_n ≤ 5000 (N²=2500만 연산이 제한 내 통과). 선형·N log N DP는 최대 200000.
  RC dp: max_r * max_c must fit within the time limit (e.g. 1000*1000=1M is fine).
  # RC DP는 R×C가 시간 제한 내에 들어와야 한다 (예: 1000×1000=100만 정도).
- graph: use GRAPH. 반드시 max_v and max_e를 둘 다 채운다.
  # 그래프: max_v(정점 수)와 max_e(간선 수) 둘 다 필수.
- 정렬: use ARRAY. 반드시 max_n을 채운다.
  # 정렬: 배열 사용, max_n 필수.
- 이분탐색: use ARRAY or ANSWER_SEARCH. 반드시 max_n을 채운다. Q가 중요하면 max_q도 채운다.
  # 이분탐색: max_n 필수, 쿼리가 중요한 문제면 max_q도 채운다.
- greedy: use ARRAY or INTERVAL. 반드시 max_n을 채운다.
  # 그리디: max_n 필수.
- bfs: use GRAPH or GRID. GRAPH면 max_v and max_e, GRID면 max_r and max_c를 채운다.
  # BFS: GRAPH면 max_v·max_e, GRID면 max_r·max_c 필수.
  GRAPH bfs: opt_tc=V_PLUS_E, bf_tc는 반드시 다른 값(예: N2)을 써야 한다. bf_tc=V_PLUS_E는 금지.
  # GRAPH BFS: opt_tc는 V_PLUS_E 고정, bf_tc는 반드시 다른 값(예: N2). 같은 값 금지.
  GRID bfs: opt_tc=RC, bf_tc는 반드시 다른 값(예: N2)을 써야 한다. bf_tc=RC는 금지.
  # GRID BFS: opt_tc는 RC 고정, bf_tc는 반드시 다른 값. 같은 값 금지.
- string: use STRING. 반드시 max_n을 채운다.
  # 문자열: max_n 필수.

Output schema:
{
  "category": "copy the input category exactly",          # 입력 category 그대로
  "difficulty": "copy the input difficulty exactly",      # 입력 difficulty 그대로
  "input_model": "one allowed value",                     # 허용된 값 중 하나
  "intended_algorithm": "short phrase",                   # 의도한 알고리즘 (짧은 설명)
  "intended_time_complexity": "short phrase such as O(N log N)",   # 최적 시간복잡도
  "intended_memory_complexity": "short phrase such as O(N)",       # 공간복잡도
  "opt_tc": "one allowed complexity code",                # 최적 복잡도 코드
  "bf_tc": "one allowed complexity code",                 # 브루트포스 복잡도 코드
  "max_constraints": "short Korean summary of constraints",  # 제약 조건 한국어 요약
  "max_n": 200000,       # 배열/문자열 최대 크기 (해당 없으면 null)
  "max_q": null,         # 쿼리 최대 수 (해당 없으면 null)
  "max_v": null,         # 정점 최대 수 (해당 없으면 null)
  "max_e": null,         # 간선 최대 수 (해당 없으면 null)
  "max_r": null,         # 격자 행 최대 수 (해당 없으면 null)
  "max_c": null,         # 격자 열 최대 수 (해당 없으면 null)
  "peak_state_items": 200000,       # 메모리 추정용 최대 상태 항목 수
  "chosen_time_limit_ms": 2000,     # 선택한 시간 제한 (ms)
  "chosen_memory_limit_mb": 256,    # 선택한 메모리 제한 (MB)
  "why_bruteforce_fails": "short explanation"  # 브루트포스가 통과 못하는 이유
}
"""

# ALGORITHM_PLAN_PROMPT 해설
# - 알고리즘 문제 1개에 대한 내부 planning JSON을 만든다.
# - category와 difficulty는 입력값을 그대로 유지하고, 번역하거나 다른 값으로 바꾸면 안 된다.
# - 난이도는 같은 category 안에서만 조정하며, 선택한 시간/메모리 제한 안에서 최적해는 통과하고 브루트포스는 통과하기 어렵게 잡아야 한다.
# - 사용하지 않는 숫자 필드는 null로 채운다.
# - input_model은 ARRAY / GRID / GRAPH / STRING / INTERVAL / ANSWER_SEARCH 중 하나만 쓴다.
# - input_model 의미:
#   ARRAY는 배열/수열, GRID는 2차원 격자, GRAPH는 정점·간선 그래프, STRING은 문자열, INTERVAL은 구간형 입력, ANSWER_SEARCH는 정답 공간 이분탐색형이다.
# - opt_tc와 bf_tc는 허용된 복잡도 코드 중 하나를 사용한다.
# - 복잡도 코드 의미:
#   LINEAR, LOG_N, N_LOG_N, N2, Q_LOG_N, V_PLUS_E, E_LOG_V, RC, RC_LOG_RC 중 하나를 사용한다.
# - 카테고리별로 max_n, max_q, max_v, max_e, max_r, max_c 같은 제약 필드를 맞게 채운다.
# - 최종 출력은 JSON 객체 하나이며, 출력 스키마의 각 필드는 그대로 맞춰야 한다.


PROBLEM_SPEC_PROMPT = """
You are an algorithm coding problem writer.
You will receive a locked planning JSON. Follow it exactly.

Hard rules:
1. Keep category exactly the same as the plan.
2. Keep time_limit exactly equal to chosen_time_limit_ms from the plan.
3. Keep memory_limit exactly equal to chosen_memory_limit_mb from the plan.
4. The core solving idea must match the plan.
5. The constraints must be consistent with the plan numbers and complexity.
6. Do not mention the solution algorithm name directly in the problem statement.
7. All natural-language fields must be written in Korean.
8. Do not include sample input or sample output.
9. Return one JSON object only. No markdown, no code block, no extra text.

Output schema:
{
  "title": "problem title",           # 문제 제목 (한국어)
  "description": "problem statement", # 문제 설명 (한국어)
  "input_description": "input format",  # 입력 형식 설명 (한국어)
  "output_description": "output format", # 출력 형식 설명 (한국어)
  "constraint_text": "constraints only", # 제약 조건만 (한국어)
  "time_limit": 1000,    # 시간 제한 (ms) — 플랜의 chosen_time_limit_ms와 동일
  "memory_limit": 256,   # 메모리 제한 (MB) — 플랜의 chosen_memory_limit_mb와 동일
  "category": "category string"  # 카테고리 — 플랜과 동일
}
"""

# PROBLEM_SPEC_PROMPT 해설
# - locked planning JSON을 기반으로 최종 문제 스펙 JSON을 작성한다.
# - category, time_limit, memory_limit은 plan 값과 정확히 일치해야 한다.
# - 문제의 핵심 풀이 방향과 제약은 plan의 알고리즘/복잡도/수치와 맞아야 한다.
# - 문제 본문에서 풀이 알고리즘 이름을 직접 드러내면 안 된다.
# - 제목, 설명, 입력 형식, 출력 형식, 제약 조건은 모두 한국어로 작성한다.
# - 샘플 입력/출력은 넣지 않는다.
# - JSON 객체 하나만 반환하고, 마크다운이나 추가 설명은 금지한다.


TESTCASE_PROMPT = """
You generate exact test cases for an algorithm problem.
You will receive a locked planning JSON and a final problem spec JSON.

Each test case uses EITHER input_data OR input_generator (never both, never neither):
- input_data: the literal input string. Use ONLY when the total text fits in ~50 lines / ~2000 chars.
- input_generator: Python 3 code that prints the input to stdout. Use for all large inputs.
  The generator must be deterministic (no random), import only stdlib, and match input_description exactly.

Hard rules:
1. Return one JSON object only with the single key "test_cases".
2. Create exactly 4 test cases.
3. Each test case must have this shape:
   {
     "input_data": "literal input OR null",       # 직접 입력 문자열 또는 null
     "input_generator": "python code OR null",    # 입력 생성 Python 코드 또는 null
     "expected_output": "leave as empty string — filled automatically",  # 빈 문자열로 두면 서버가 자동으로 채운다
     "primary_size": 100,       # 이 케이스의 dominant 입력 크기 (아래 10번 규칙 참조)
     "is_boundary": true,       # 경계 케이스 여부
     "is_anti_naive": false,    # 브루트포스를 걸러내는 케이스 여부
     "design_note": "short English note"  # 이 케이스의 설계 의도 (짧게)
   }
4. input_data (or the output of input_generator) must print EVERY field in input_description.
   Count every value, every line, every token — a missing field causes IndexError.
   GRAPH generators: ALL vertex indices must be in [1, N]. Loop EXACTLY M times for M edges.
   If input_description mentions extra fields after edges (starting node S, target T, etc.)
   you MUST print them — they are not optional.
   ARRAY/INTERVAL generators: if there are Q queries after the array, print Q and all queries.
5. input_generator imports: write ALL needed imports at the top (e.g. `import string`, `import math`).
   Do NOT use random — generate data with arithmetic patterns (cycling, sequences, formulas).
6. All four inputs must represent different scenarios.
7. Include at least one boundary case.
8. Include at least one anti-naive case.
9. Cases 3 and 4 (the 3rd and 4th in the array) are HIDDEN. At least one of them MUST
   have is_anti_naive: true. Never leave BOTH hidden cases with is_anti_naive: false.
   Anti-naive cases must have LARGE primary_size (close to the required threshold).
10. primary_size must be the dominant input scale for the case:
    - ARRAY / STRING / INTERVAL / ANSWER_SEARCH: use N (array length)
    - GRAPH: use max(V, E) — the LARGER of vertex count and edge count; never use V alone
    - GRID: use R*C (total cell count)
11. Keep all test cases valid for the stated constraints.
12. Return JSON only. No markdown, no code block, no extra text.
"""

# TESTCASE_PROMPT 해설
# - locked planning JSON과 최종 문제 스펙 JSON을 바탕으로 정확한 테스트케이스 4개를 만든다.
# - 각 테스트케이스는 input_data와 input_generator 중 정확히 하나만 사용해야 한다.
# - 작은 입력만 literal input_data를 쓰고, 큰 입력은 deterministic한 Python generator를 사용한다.
# - generator는 random 없이 stdlib만 사용하고, input_description의 모든 필드를 정확히 출력해야 한다.
# - expected_output은 빈 문자열로 두고 서버가 자동으로 채운다.
# - 네 케이스는 서로 다른 시나리오여야 하며, 최소 1개 boundary, 최소 1개 anti-naive가 필요하다.
# - 3번/4번 숨김 케이스 중 최소 하나는 반드시 anti-naive여야 한다.
# - primary_size는 모델별로 dominant scale을 넣어야 한다.
#   ARRAY / STRING / INTERVAL / ANSWER_SEARCH는 N, GRAPH는 max(V, E), GRID는 R*C를 사용한다.
# - 모든 테스트케이스는 제약을 만족해야 하며, 반환 형식은 JSON 객체 하나만 허용된다.


SOLUTION_PROMPT = """
You write a correct Python reference solution for the given algorithm problem.
Return plain Python code only. No markdown, no code block, no extra explanation.
Read from standard input and write to standard output.

Hard rules:
1. Parse input EXACTLY as described in input_description — same order, same line structure.
2. Handle ALL edge cases including minimum/maximum constraints and empty inputs.
3. Never assume extra fields that are not described; never skip fields that are described.
4. Always print at least one line of output. Problems with empty output are not supported.
"""

# SOLUTION_PROMPT 해설
# - 주어진 알고리즘 문제에 대한 정답 Python 레퍼런스 코드를 작성한다.
# - 반환은 순수 Python 코드만 허용되며, 마크다운/코드블록/설명 텍스트는 금지한다.
# - 입력은 input_description의 순서와 줄 구조를 정확히 따라 파싱해야 한다.
# - 최소/최대 제약과 빈 입력을 포함한 모든 엣지케이스를 처리해야 한다.
# - 문제 설명에 없는 필드를 임의로 가정하면 안 되고, 설명된 필드는 절대 생략하면 안 된다.
# - 출력이 완전히 비어 있는 문제는 지원하지 않으므로 최소 한 줄은 반드시 출력해야 한다.


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


def _parse_optional_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def _strip_code_block(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.split("\n")
    inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
    return "\n".join(inner).strip()


def _compact_token(value: str) -> str:
    return "".join(ch for ch in value.upper() if ch.isalnum())


def _normalize_category_value(value: str, fallback: str) -> str:
    token = _compact_token(value)
    return CATEGORY_ALIASES.get(token, fallback)


def _normalize_input_model_value(value: str, category: str) -> str:
    token = _compact_token(value)
    normalized = INPUT_MODEL_ALIASES.get(token)
    if normalized:
        return normalized
    if category == "graph":
        return "GRAPH"
    if category == "bfs":
        return "GRAPH"
    if category == "string":
        return "STRING"
    if category == "정렬":
        return "ARRAY"
    if category == "이분탐색":
        return "ARRAY"
    if category == "greedy":
        return "ARRAY"
    if category == "dp":
        return "ARRAY"
    return "ARRAY"


def _normalize_complexity_code(value: str, fallback: str) -> str:
    token = _compact_token(value)
    if token in {"LINEAR", "ON"}:
        return "LINEAR"
    if token in {"LOGN", "OLOGN"}:
        return "LOG_N"
    if token in {"NLOGN", "ONLOGN"}:
        return "N_LOG_N"
    if token in {"N2", "ON2", "NN", "NPOWER2"}:
        return "N2"
    if token in {"QLOGN", "OQLOGN"}:
        return "Q_LOG_N"
    if token in {"VPLUSE", "OVE", "OVPLUSE"}:
        return "V_PLUS_E"
    if token in {"ELOGV", "OELOGV"}:
        return "E_LOG_V"
    if token in {"RC", "ORC", "RTIMESC"}:
        return "RC"
    if token in {"RCLOGRC", "ORCLOGRC"}:
        return "RC_LOG_RC"
    return fallback


def _extract_upper_bound(text: str, variable_name: str) -> int | None:
    patterns = [
        rf"1\s*<=\s*{variable_name}\s*<=\s*([0-9][0-9_,]*)",
        rf"0\s*<=\s*{variable_name}\s*<=\s*([0-9][0-9_,]*)",
        rf"\b{variable_name}\b[^0-9]{{0,20}}<=\s*([0-9][0-9_,]*)",
        rf"\b{variable_name}\b[^0-9]{{0,20}}([0-9][0-9_,]*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _parse_optional_positive_int(match.group(1).replace(",", "").replace("_", ""))
    return None


def _infer_size_fields(
    category: str,
    input_model: str,
    opt_tc: str,
    max_constraints: str,
    current_fields: dict[str, int | None],
) -> dict[str, int | None]:
    inferred = dict(current_fields)
    text = max_constraints or ""

    for variable_name, field_name in (
        ("N", "max_n"),
        ("Q", "max_q"),
        ("V", "max_v"),
        ("E", "max_e"),
        ("R", "max_r"),
        ("C", "max_c"),
    ):
        if inferred[field_name] is None:
            inferred[field_name] = _extract_upper_bound(text, variable_name)

    if category in {"graph", "bfs"} and input_model == "GRAPH":
        inferred["max_v"] = inferred["max_v"] or 100000
        inferred["max_e"] = inferred["max_e"] or max(200000, (inferred["max_v"] or 0) * 2)
    if category in {"bfs", "구현", "dp"} and input_model == "GRID":
        inferred["max_r"] = inferred["max_r"] or 500
        inferred["max_c"] = inferred["max_c"] or 500
    if input_model in {"ARRAY", "STRING", "INTERVAL", "ANSWER_SEARCH"}:
        if inferred["max_n"] is None:
            if opt_tc == "N2":
                inferred["max_n"] = 5000
            else:
                inferred["max_n"] = 200000
        elif opt_tc == "N2" and inferred["max_n"] > 5000:
            pass  # leave as-is; _validate_plan will reject and request a smaller max_n
    if input_model == "STRING" and inferred["max_n"] is None:
        inferred["max_n"] = 200000
    if category == "이분탐색" and inferred["max_q"] is None and opt_tc == "Q_LOG_N":
        inferred["max_q"] = inferred["max_n"] or 100000

    return inferred


def _extract_key_error(error_info: str) -> str:
    """Extract the actionable part of a Python traceback: failing line + error type."""
    text = error_info.replace("\\r\\n", "\n").replace("\\r", "\n")
    lines = text.splitlines()
    code_lines = [l.strip() for l in lines if l.startswith("    ") and not l.strip().startswith(("File", "^", "~"))]
    error_line = next((l.strip() for l in reversed(lines) if l.strip() and not l.startswith(" ")), "")
    parts = []
    if code_lines:
        parts.append(f"failed at: `{code_lines[-1]}`")
    if error_line and "Error" in error_line:
        parts.append(error_line)
    return "; ".join(parts) if parts else error_info[:200]


def _format_issues(issues: list[str]) -> str:
    return "\n".join(f"- {issue}" for issue in issues[:6])


def _build_plan_feedback(plan: AlgorithmGenerationPlan, issues: list[str]) -> str:
    feedback = _format_issues(issues)
    hints: list[str] = []

    if any("bf_tc" in issue for issue in issues):
        hints.append("Choose a MUCH weaker bf_tc than opt_tc.")
        hints.append("bf_tc must exceed the time budget and be at least 8x more expensive than opt_tc.")
        if plan.input_model == "GRID":
            hints.append("For GRID, use a clearly superlinear brute force over total cells and keep R*C large.")
            hints.append("Prefer bf_tc=RC_LOG_RC or bf_tc=N2 only when it is truly over total cells, not equivalent to RC.")
        elif plan.input_model in {"ARRAY", "STRING", "INTERVAL", "ANSWER_SEARCH"}:
            hints.append("For ARRAY/STRING-like inputs, prefer bf_tc=N2 and keep max_n near the upper bound.")
        elif plan.input_model == "GRAPH":
            hints.append("For GRAPH, prefer a clearly slower bf_tc such as E_LOG_V with large enough V and E.")

    if hints:
        feedback += "\n" + "\n".join(f"- {hint}" for hint in hints)

    return feedback


def _build_testcase_feedback(plan: AlgorithmGenerationPlan, issues: list[str]) -> str:
    feedback = _format_issues(issues)
    hints: list[str] = []

    if any(issue.startswith("input generator failed on case") for issue in issues):
        hints.append("Regenerate the test cases. The previous input_generator crashed.")
        hints.append("Keep every declared count consistent with the emitted data: N items, M edges, Q operations.")
        hints.append("Never index beyond a generated list. If you print Q lines, build exactly Q values and loop over range(q).")
        hints.append("Print every required field from input_description in the exact order.")

    if any("timed out after" in issue for issue in issues):
        hints.append("The previous reference execution timed out on generated inputs.")
        hints.append("Keep anti-naive cases large but still valid for the intended optimal solution.")
        hints.append("Do not accidentally create oversized output-heavy or malformed cases that distort runtime.")

    if hints:
        feedback += "\n" + "\n".join(f"- {hint}" for hint in hints)

    return feedback


def _plan_brief(plan: AlgorithmGenerationPlan) -> str:
    return (
        f"category={plan.category}, difficulty={plan.difficulty}, input_model={plan.input_model}, "
        f"opt_tc={plan.opt_tc}, bf_tc={plan.bf_tc}, tl_ms={plan.chosen_time_limit_ms}, "
        f"ml_mb={plan.chosen_memory_limit_mb}"
    )


def _normalize_generation_plan(
    difficulty: str,
    category: str,
    plan_data: dict[str, Any],
) -> AlgorithmGenerationPlan:
    bands = RESOURCE_BANDS.get(difficulty, RESOURCE_BANDS["MEDIUM"])
    normalized_category = _normalize_category_value(str(plan_data.get("category", "")), category)
    normalized_category = category if category in VALID_CATEGORIES else normalized_category
    normalized_input_model = _normalize_input_model_value(
        str(plan_data.get("input_model", "")),
        normalized_category,
    )
    normalized_opt_tc = _normalize_complexity_code(
        str(plan_data.get("opt_tc", "")),
        "LINEAR",
    )
    normalized_bf_tc = _normalize_complexity_code(
        str(plan_data.get("bf_tc", "")),
        "N2",
    )
    max_constraints = _get_required_text(plan_data, "max_constraints", "algorithm plan")
    size_fields = _infer_size_fields(
        normalized_category,
        normalized_input_model,
        normalized_opt_tc,
        max_constraints,
        {
            "max_n": _parse_optional_positive_int(plan_data.get("max_n")),
            "max_q": _parse_optional_positive_int(plan_data.get("max_q")),
            "max_v": _parse_optional_positive_int(plan_data.get("max_v")),
            "max_e": _parse_optional_positive_int(plan_data.get("max_e")),
            "max_r": _parse_optional_positive_int(plan_data.get("max_r")),
            "max_c": _parse_optional_positive_int(plan_data.get("max_c")),
        },
    )
    peak_state_items = _parse_optional_positive_int(plan_data.get("peak_state_items"))
    if peak_state_items is None:
        if normalized_input_model == "GRAPH":
            peak_state_items = max((size_fields["max_v"] or 0) + (size_fields["max_e"] or 0), size_fields["max_e"] or 0)
        elif normalized_input_model == "GRID":
            peak_state_items = (size_fields["max_r"] or 0) * (size_fields["max_c"] or 0)
        else:
            peak_state_items = size_fields["max_n"] or size_fields["max_q"]

    return AlgorithmGenerationPlan(
        difficulty=difficulty,
        category=normalized_category,
        input_model=normalized_input_model,
        intended_algorithm=_get_required_text(plan_data, "intended_algorithm", "algorithm plan"),
        intended_time_complexity=_get_required_text(plan_data, "intended_time_complexity", "algorithm plan"),
        intended_memory_complexity=_get_required_text(plan_data, "intended_memory_complexity", "algorithm plan"),
        opt_tc=normalized_opt_tc,
        bf_tc=normalized_bf_tc,
        max_constraints=max_constraints,
        max_n=size_fields["max_n"],
        max_q=size_fields["max_q"],
        max_v=size_fields["max_v"],
        max_e=size_fields["max_e"],
        max_r=size_fields["max_r"],
        max_c=size_fields["max_c"],
        peak_state_items=peak_state_items,
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
        raise RuntimeError("Failed to generate test cases: missing test_cases list.")
    if len(test_cases) != 4:
        raise RuntimeError("Failed to generate test cases: expected exactly 4 cases.")

    normalized_cases: list[dict[str, Any]] = []
    for case in test_cases:
        if not isinstance(case, dict):
            raise RuntimeError("Failed to generate test cases: each case must be a JSON object.")
        normalized_cases.append(
            {
                "input_data": str(case.get("input_data") or "").strip(),
                "input_generator": str(case.get("input_generator") or "").strip(),
                "expected_output": str(case.get("expected_output", "")).strip(),
                "primary_size": _parse_optional_positive_int(case.get("primary_size")) or 0,
                "is_boundary": _parse_bool(case.get("is_boundary")),
                "is_anti_naive": _parse_bool(case.get("is_anti_naive")),
                "design_note": str(case.get("design_note", "")).strip(),
            }
        )
    return normalized_cases


def _has_required_shape(plan: AlgorithmGenerationPlan, fields: tuple[str, ...]) -> bool:
    return all(getattr(plan, field_name) is not None for field_name in fields)


def _estimate_ops(plan: AlgorithmGenerationPlan, complexity_code: str) -> float | None:
    n = float(plan.max_n or 0)
    q = float(plan.max_q or 0)
    v = float(plan.max_v or 0)
    e = float(plan.max_e or 0)
    rc = float((plan.max_r or 0) * (plan.max_c or 0))

    if complexity_code == "LINEAR":
        if plan.input_model == "GRAPH":
            return v + e if (v or e) else None
        if plan.input_model == "GRID":
            return rc or None
        return n or q or None
    if complexity_code == "LOG_N":
        if n <= 0:
            return None
        return math.log2(max(n, 2.0))
    if complexity_code == "N_LOG_N":
        if n <= 0:
            return None
        return n * math.log2(max(n, 2.0))
    if complexity_code == "N2":
        if plan.input_model == "GRID":
            if rc <= 0:
                return None
            return rc * rc
        scale = n or v or (math.sqrt(rc) if rc > 0 else 0)
        if scale <= 0:
            return None
        return scale * scale
    if complexity_code == "Q_LOG_N":
        effective_q = q or n
        if effective_q <= 0 or n <= 0:
            return None
        return effective_q * math.log2(max(n, 2.0))
    if complexity_code == "V_PLUS_E":
        total = v + e
        return total or None
    if complexity_code == "E_LOG_V":
        if e <= 0 or v <= 0:
            return None
        return e * math.log2(max(v, 2.0))
    if complexity_code == "RC":
        return rc or None
    if complexity_code == "RC_LOG_RC":
        if rc <= 0:
            return None
        return rc * math.log2(max(rc, 2.0))
    return None


def _ops_budget(time_limit_ms: int) -> float:
    return max(1_500_000.0, time_limit_ms * OPERATIONS_PER_MS)


def _estimate_peak_state_items(plan: AlgorithmGenerationPlan) -> int:
    if plan.peak_state_items:
        return plan.peak_state_items
    if plan.input_model == "GRAPH":
        return max((plan.max_v or 0) + (plan.max_e or 0), plan.max_e or 0)
    if plan.input_model == "GRID":
        return (plan.max_r or 0) * (plan.max_c or 0)
    return plan.max_n or plan.max_q or 0


def _target_primary_size(plan: AlgorithmGenerationPlan) -> int:
    if plan.input_model == "GRAPH":
        return max(plan.max_v or 0, plan.max_e or 0)
    if plan.input_model == "GRID":
        return (plan.max_r or 0) * (plan.max_c or 0)
    return max(plan.max_n or 0, plan.max_q or 0)


def _reference_solution_timeout_seconds(plan: "AlgorithmGenerationPlan | None") -> float:
    if plan is None:
        return 5.0
    base_seconds = plan.chosen_time_limit_ms / 1000.0
    return max(5.0, min(15.0, base_seconds * 3.0))


def _compute_actual_primary_size(input_data: str, input_model: str, max_q: int | None = None) -> int | None:
    """Server-side primary_size from actual input text — not LLM self-report.

    Returns:
      None  — parse failed or input is ambiguous (caller should reject)
      0     — parsed successfully, dominant size is zero (N=0, no elements)
      >0    — valid dominant size

    N=0 is returned as 0 (not None) so callers can distinguish "genuinely empty"
    from "could not parse". Both should be rejected by current policy, but for
    different reasons.

    Q heuristic is applied only for ARRAY and ANSWER_SEARCH (conservative).
    Format detection order:
      A) Header has ≥2 tokens (N Q on same line) → max(N, Q)
      B) Header has 1 token; array is one space-separated line → Q is lines[2]
      C) Header has 1 token; array is one element per line → Q is lines[N+1]
    Ambiguous → return N (no Q), never guess aggressively.
    """
    if not input_data:
        return None
    lines = [line for line in input_data.strip().splitlines() if line.strip()]
    if not lines:
        return None

    def _single_int(line: str) -> int | None:
        tokens = line.split()
        if len(tokens) == 1:
            try:
                return int(tokens[0])
            except ValueError:
                pass
        return None

    try:
        header_tokens = lines[0].split()

        if input_model == "GRAPH":
            if len(header_tokens) >= 2:
                return max(int(header_tokens[0]), int(header_tokens[1]))
            return int(header_tokens[0])

        if input_model == "GRID":
            if len(header_tokens) >= 2:
                return int(header_tokens[0]) * int(header_tokens[1])
            return int(header_tokens[0])

        # INTERVAL / STRING: no Q heuristic, just N
        if input_model not in {"ARRAY", "ANSWER_SEARCH"}:
            return int(header_tokens[0])

        # ── ARRAY / ANSWER_SEARCH ─────────────────────────────────────────────
        # Format A: "N Q" on the header line
        if len(header_tokens) >= 2 and max_q is not None and max_q > 0:
            try:
                n_a, q_a = int(header_tokens[0]), int(header_tokens[1])
                return max(n_a, q_a)
            except ValueError:
                pass

        n = int(header_tokens[0])

        if max_q is None or max_q <= 0:
            return n

        # N=0: no array elements — Q is on the next line (return 0 if missing)
        if n == 0:
            if len(lines) > 1:
                q_zero = _single_int(lines[1])
                if q_zero is not None and q_zero >= 0:
                    return q_zero
            logger.debug("_compute_actual_primary_size: N=0, Q not found, returning 0")
            return 0

        if len(lines) < 3:
            return n

        # Format B: array is all on one line (most common in competitive programming)
        second_line_tokens = lines[1].split()
        if len(second_line_tokens) >= n:
            q_b = _single_int(lines[2]) if len(lines) > 2 else None
            if q_b is not None and q_b > 0:
                return max(n, q_b)
            logger.debug("_compute_actual_primary_size: Q not found (Format B), returning N=%s", n)
            return n

        # Format C: array is one element per line
        q_line_idx = n + 1  # header=line0, array=lines[1..n], Q=line[n+1]
        if q_line_idx < len(lines):
            q_c = _single_int(lines[q_line_idx])
            if q_c is not None and q_c > 0:
                return max(n, q_c)

        logger.debug("_compute_actual_primary_size: Q ambiguous (Format C fallback), returning N=%s", n)
        return n

    except (ValueError, IndexError):
        logger.debug("_compute_actual_primary_size: parse failed for input_model=%s", input_model)
        return None


def _anti_naive_ratio(plan: AlgorithmGenerationPlan) -> float:
    if plan.bf_tc == "N2":
        return {"EASY": 0.08, "MEDIUM": 0.12, "HARD": 0.18}[plan.difficulty]
    if plan.bf_tc in {"RC", "RC_LOG_RC"}:
        return {"EASY": 0.12, "MEDIUM": 0.18, "HARD": 0.25}[plan.difficulty]
    if plan.bf_tc in {"N_LOG_N", "Q_LOG_N", "E_LOG_V"}:
        return {"EASY": 0.18, "MEDIUM": 0.25, "HARD": 0.35}[plan.difficulty]
    return TESTCASE_ANTI_NAIVE_RATIO[plan.difficulty]


def _requires_large_anti_naive(plan: AlgorithmGenerationPlan) -> bool:
    if plan.category not in LARGE_ANTI_NAIVE_CATEGORIES:
        return False
    return plan.bf_tc not in {"LINEAR", "LOG_N"}


def _plan_needs_generator(plan: AlgorithmGenerationPlan) -> bool:
    """Return True when near-max test case input would exceed LITERAL_INPUT_CHAR_LIMIT chars.

    Estimates the character count of a near-max input for the given plan.
    GRAPH always needs a generator (edge lists are inherently large).
    """
    near_max_ratio = TESTCASE_NEAR_MAX_RATIO.get(plan.difficulty, 0.6)

    if plan.input_model == "GRAPH":
        dominant = max(plan.max_v or 0, plan.max_e or 0)
        near_max = int(dominant * near_max_ratio)
        return near_max * 10 > LITERAL_INPUT_CHAR_LIMIT  # ~10 chars per edge ("u v\n")

    if plan.input_model == "GRID":
        rc = (plan.max_r or 0) * (plan.max_c or 0)
        near_max_cells = int(rc * near_max_ratio)
        return near_max_cells * 3 > LITERAL_INPUT_CHAR_LIMIT  # ~3 chars per cell

    if plan.input_model == "INTERVAL":
        dominant = max(plan.max_n or 0, plan.max_q or 0)
        near_max = int(dominant * near_max_ratio)
        return near_max * 14 > LITERAL_INPUT_CHAR_LIMIT  # ~14 chars per interval ("l r\n")

    # ARRAY / STRING / ANSWER_SEARCH
    dominant = max(plan.max_n or 0, plan.max_q or 0)
    near_max = int(dominant * near_max_ratio)
    return near_max * 7 > LITERAL_INPUT_CHAR_LIMIT  # ~7 chars per integer ("12345 \n")


def _validate_plan(
    requested_difficulty: str,
    requested_category: str,
    plan: AlgorithmGenerationPlan,
) -> list[str]:
    issues: list[str] = []

    if plan.category != requested_category:
        issues.append("category mismatch")
    if plan.difficulty != requested_difficulty:
        issues.append("difficulty mismatch")
    if plan.category not in VALID_CATEGORIES:
        issues.append("unknown category")
        return issues
    if plan.input_model not in VALID_INPUT_MODELS:
        issues.append("invalid input_model")
    if plan.opt_tc not in VALID_COMPLEXITY_CODES:
        issues.append("invalid opt_tc")
    if plan.bf_tc not in VALID_COMPLEXITY_CODES:
        issues.append("invalid bf_tc")

    profile = CATEGORY_PROFILES.get(requested_category)
    if profile is None:
        issues.append("missing category profile")
        return issues

    if plan.input_model not in profile["input_models"]:
        issues.append(f"{requested_category} category requires a compatible input_model")
    if plan.opt_tc not in profile["opt_tc"]:
        issues.append(f"{requested_category} category requires a compatible opt_tc")
    if not any(_has_required_shape(plan, required_fields) for required_fields in profile["required_any"]):
        issues.append(f"{requested_category} category is missing required size fields")

    if plan.opt_tc == plan.bf_tc:
        issues.append("bf_tc must be weaker than opt_tc")

    if plan.opt_tc == "N2" and (plan.max_n or 0) > 5000:
        issues.append(
            f"N2 opt_tc requires max_n <= 5000 (N²=25M fits budget), "
            f"but max_n={plan.max_n}. Reduce max_n or choose a different opt_tc."
        )

    opt_ops = _estimate_ops(plan, plan.opt_tc)
    bf_ops = _estimate_ops(plan, plan.bf_tc)
    ops_budget = _ops_budget(plan.chosen_time_limit_ms)

    if opt_ops is None:
        issues.append("could not estimate opt_tc operations")
    elif opt_ops > ops_budget:
        issues.append("opt_tc too expensive for chosen_time_limit_ms")

    if bf_ops is None:
        issues.append("could not estimate bf_tc operations")
    else:
        if bf_ops <= ops_budget * 1.25:
            issues.append("bf_tc too weak; brute force may still pass")
        if opt_ops is not None and bf_ops < opt_ops * 8:
            issues.append("bf_tc is not sufficiently worse than opt_tc")

    estimated_memory_bytes = _estimate_peak_state_items(plan) * MEMORY_BYTES_PER_ITEM
    memory_budget_bytes = plan.chosen_memory_limit_mb * 1024 * 1024 * 0.6
    if estimated_memory_bytes > memory_budget_bytes:
        issues.append("peak_state_items too large for chosen_memory_limit_mb")

    return issues


def _validate_test_cases(
    plan: AlgorithmGenerationPlan,
    test_cases: list[dict[str, Any]],
) -> list[str]:
    issues: list[str] = []

    if len(test_cases) != 4:
        issues.append("expected exactly 4 test cases")
        return issues

    # P1-2: override LLM-reported primary_size with server-calculated value for literal cases
    # Always override — None means "parse failed", 0 means "zero-size". Both get rejected below.
    for test_case in test_cases:
        if test_case.get("input_data") and not test_case.get("input_generator"):
            test_case["primary_size"] = _compute_actual_primary_size(
                test_case["input_data"], plan.input_model, plan.max_q
            )

    must_use_generator = _plan_needs_generator(plan)
    seen_inputs: set[str] = set()
    for index, test_case in enumerate(test_cases, start=1):
        has_generator = bool(test_case.get("input_generator"))
        has_literal = bool(test_case.get("input_data"))
        if has_generator and has_literal:
            issues.append(
                f"case {index} has both input_data and input_generator — "
                "set exactly one; the other must be null"
            )
        elif must_use_generator and has_literal and not has_generator:
            issues.append(
                f"case {index} uses literal input_data for a {plan.input_model} problem — "
                "set input_data to null and use input_generator instead"
            )
        elif not has_generator and not has_literal:
            issues.append(f"case {index} has empty input_data and no input_generator")
        if not has_generator:
            if test_case["input_data"] in seen_inputs:
                issues.append(f"case {index} duplicates another input")
            seen_inputs.add(test_case["input_data"])
        ps = test_case["primary_size"]
        if ps is None:
            issues.append(
                f"case {index} dominant size could not be inferred from actual input — "
                "check that the input follows input_description exactly"
            )
        elif ps == 0:
            issues.append(
                f"case {index} has zero dominant size — "
                "generated test cases must have at least one element/edge/cell/query"
            )
        elif ps < 0:
            issues.append(f"case {index} has invalid primary_size ({ps})")

    if not any(test_case["is_boundary"] for test_case in test_cases):
        issues.append("missing boundary case")
    if not any(test_case["is_anti_naive"] for test_case in test_cases):
        issues.append("missing anti-naive case")
    if not any(test_case["is_anti_naive"] for test_case in test_cases[2:]):
        hidden_anti = [i + 1 for i, tc in enumerate(test_cases[2:], start=2) if tc["is_anti_naive"]]
        issues.append(
            "at least one hidden case must be anti-naive "
            "(cases 3 and 4 are the hidden cases — "
            "set is_anti_naive: true for case 3 or case 4, currently both are false)"
        )

    target_size = _target_primary_size(plan)
    if target_size > 0:
        near_max_threshold = max(2, int(target_size * TESTCASE_NEAR_MAX_RATIO[plan.difficulty]))
        anti_naive_threshold = max(2, int(target_size * _anti_naive_ratio(plan)))

        size_issues = _validate_size_constraints(plan, test_cases, target_size, near_max_threshold, anti_naive_threshold)
        issues.extend(size_issues)

    return issues


def _validate_size_constraints(
    plan: AlgorithmGenerationPlan,
    test_cases: list[dict[str, Any]],
    target_size: int,
    near_max_threshold: int,
    anti_naive_threshold: int,
) -> list[str]:
    """Re-runnable size checks — called after initial validation and after materialization."""
    issues: list[str] = []
    if target_size <= 0:
        return issues

    def _safe_size(tc: dict) -> int:
        ps = tc.get("primary_size")
        return ps if isinstance(ps, int) and ps > 0 else 0

    if not any(_safe_size(tc) >= near_max_threshold for tc in test_cases):
        issues.append(
            f"no test case is close enough to the upper constraint "
            f"(need at least one with primary_size >= {near_max_threshold}; max is {target_size})"
        )

    anti_naive_cases = [tc for tc in test_cases if tc["is_anti_naive"]]
    if _requires_large_anti_naive(plan) and anti_naive_cases and not any(
        _safe_size(tc) >= anti_naive_threshold for tc in anti_naive_cases
    ):
        anti_naive_sizes = [tc.get("primary_size") for tc in anti_naive_cases]
        issues.append(
            "anti-naive cases are too small for the planned constraints "
            f"(threshold={anti_naive_threshold}, sizes={anti_naive_sizes})"
        )

    for index, tc in enumerate(test_cases, start=1):
        ps = tc.get("primary_size")
        if isinstance(ps, int) and ps > target_size:
            issues.append(f"case {index} exceeds the planned dominant size")

    return issues


def _run_solution_sync(code: str, input_data: str, timeout: float) -> tuple[str | None, str]:
    """Blocking subprocess execution — called from a thread pool executor.

    Returns (stdout, error_info). error_info is empty on success.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as file:
            file.write(code)
            tmp_path = file.name
        result = subprocess.run(
            [sys.executable, tmp_path],
            input=input_data.encode(),
            capture_output=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            output = result.stdout.decode().strip()
            if not output:
                return None, "solution produced empty output (empty output is not supported)"
            return output, ""
        stderr_text = result.stderr.decode(errors="replace").strip()[:500]
        logger.warning("Reference solution exited with code %s. stderr: %s", result.returncode, stderr_text)
        return None, f"exit {result.returncode}: {stderr_text}"
    except subprocess.TimeoutExpired:
        logger.warning("Reference solution timed out after %ss.", timeout)
        return None, f"timed out after {timeout}s"
    except Exception as exc:
        logger.exception("Reference solution runner raised an unexpected exception.")
        return None, f"runner exception: {exc}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


async def _run_solution(code: str, input_data: str, timeout: float = 5.0) -> tuple[str | None, str]:
    """Run a Python reference solution and return (stdout, error_info).

    Uses a thread pool executor so it works on both ProactorEventLoop and
    SelectorEventLoop (asyncio.create_subprocess_exec requires ProactorEventLoop
    on Windows and raises NotImplementedError under SelectorEventLoop).
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        functools.partial(_run_solution_sync, code, input_data, timeout),
    )


def _run_generator_sync(code: str, timeout: float = 15.0) -> tuple[str | None, str]:
    """Run a Python input-generator script and return (stdout, error_info)."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as file:
            file.write(code)
            tmp_path = file.name
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            output = result.stdout.decode()
            if output.strip():
                return output, ""
            return None, "generator produced empty output"
        stderr_text = result.stderr.decode(errors="replace").strip()[:500]
        logger.warning("Input generator exited with code %s. stderr: %s", result.returncode, stderr_text)
        return None, f"exit {result.returncode}: {stderr_text}"
    except subprocess.TimeoutExpired:
        logger.warning("Input generator timed out after %ss.", timeout)
        return None, f"generator timed out after {timeout}s"
    except Exception as exc:
        logger.exception("Input generator raised an unexpected exception.")
        return None, f"generator exception: {exc}"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


async def _run_generator(code: str, timeout: float = 15.0) -> tuple[str | None, str]:
    """Async wrapper for _run_generator_sync."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        functools.partial(_run_generator_sync, code, timeout),
    )


async def _generate_validated_plan(difficulty: str, category: str) -> AlgorithmGenerationPlan:
    feedback = ""
    last_issues: list[str] = []
    logger.info("Algorithm generation stage started: plan generation. category=%s difficulty=%s", category, difficulty)

    for attempt in range(1, PLAN_RETRY_LIMIT + 1):
        attempt_started_at = time.perf_counter()
        logger.info(
            "Algorithm plan attempt %s/%s started. category=%s difficulty=%s",
            attempt,
            PLAN_RETRY_LIMIT,
            category,
            difficulty,
        )
        user_content = (
            f"difficulty={difficulty}\n"
            f"category={category}\n\n"
            "Keep category fixed. Choose dynamic limits and constraints that fit the same category."
        )
        if feedback:
            user_content += f"\n\nPrevious validation issues:\n{feedback}\nFix every issue."

        plan_messages = [
            {"role": "system", "content": ALGORITHM_PLAN_PROMPT},
            {"role": "user", "content": user_content},
        ]
        plan_raw = await request_json_completion(plan_messages, temperature=0.2)
        plan_data = _parse_json_object(plan_raw, "algorithm plan")
        plan = _normalize_generation_plan(difficulty, category, plan_data)
        issues = _validate_plan(difficulty, category, plan)
        if not issues:
            elapsed_ms = int((time.perf_counter() - attempt_started_at) * 1000)
            logger.info(
                "Algorithm plan attempt %s/%s succeeded in %sms. %s",
                attempt,
                PLAN_RETRY_LIMIT,
                elapsed_ms,
                _plan_brief(plan),
            )
            return plan
        logger.warning(
            "Algorithm plan validation failed for category=%s difficulty=%s: %s | %s",
            category,
            difficulty,
            issues,
            _plan_brief(plan),
        )
        last_issues = issues
        feedback = _build_plan_feedback(plan, issues)

    raise RuntimeError(f"Failed to generate a valid algorithm plan: {'; '.join(last_issues)}")


async def _generate_reference_solution(spec_text: str, error_feedback: str = "") -> str:
    started_at = time.perf_counter()
    logger.info("Algorithm generation stage started: reference solution generation.")
    user_content = f"Write a correct Python solution for this problem:\n\n{spec_text}"
    if error_feedback:
        user_content += (
            f"\n\nThe previous solution failed with this error:\n{error_feedback}\n"
            "Fix the bug. Common causes:\n"
            "(1) input parsing does not match input_description — re-read it carefully;\n"
            "(2) edge cases not handled: N=0, empty array, single element, disconnected graph;\n"
            "(3) using multiple input() calls instead of sys.stdin.read().split() — "
            "prefer reading all data at once to avoid EOFError on edge cases.\n"
        )
        if "timed out after" in error_feedback:
            user_content += (
                "(4) the previous solution timed out — complexity is too slow. "
                "Use the intended optimal algorithm only; avoid quadratic rescans, "
                "full recomputation per query, and one-print-per-line output on huge results.\n"
            )
        user_content += (
            "Recommended input pattern:\n"
            "  import sys\n"
            "  data = sys.stdin.read().split()\n"
            "  idx = 0\n"
            "  n = int(data[idx]); idx += 1\n"
            "  arr = [int(data[idx+i]) for i in range(n)]; idx += n\n"
        )
    solution_messages = [
        {"role": "system", "content": SOLUTION_PROMPT},
        {"role": "user", "content": user_content},
    ]
    solution_raw = await request_chat_completion(solution_messages, temperature=0.0)
    solution_code = _strip_code_block(solution_raw)
    logger.info(
        "Algorithm generation stage finished: reference solution generation in %sms. code_length=%s",
        int((time.perf_counter() - started_at) * 1000),
        len(solution_code),
    )
    return solution_code


_INPUT_FORMAT_ERRORS = ("EOFError", "StopIteration")
_INPUT_PARSE_KEYWORDS = ("data[", "edges_data[", "input(", "stdin", ".split()", "readline", "int(line", "map(int", "_data[", "lines[")


async def _materialize_reference_outputs(
    solution_code: str,
    test_cases: list[dict[str, Any]],
    plan: "AlgorithmGenerationPlan | None" = None,
) -> tuple[list[str], str]:
    """Execute test cases against the reference solution.

    Returns (issues, first_solution_error). Distinguishes between generator bugs
    (test case content problem) and solution bugs (solution logic problem).
    Generators and solution runs execute in parallel for speed.
    """
    issues: list[str] = []
    generator_failure_count = 0
    solution_failure_count = 0
    first_solution_error: str = ""
    logger.info("Algorithm generation stage started: reference execution for %s test cases.", len(test_cases))

    # ── 1. Run all generators in parallel ────────────────────────────────────
    gen_tasks = {
        index: _run_generator(test_case["input_generator"])
        for index, test_case in enumerate(test_cases, start=1)
        if test_case.get("input_generator") and not test_case["input_data"]
    }
    if gen_tasks:
        gen_started_at = time.perf_counter()
        logger.info("Running %s input generators in parallel.", len(gen_tasks))
        gen_results = await asyncio.gather(*gen_tasks.values())
        for (index, _), (generated, gen_error) in zip(gen_tasks.items(), gen_results):
            test_case = test_cases[index - 1]
            if generated is None:
                generator_failure_count += 1
                issues.append(f"input generator failed on case {index}: {gen_error}")
                logger.warning("Input generator failed for testcase %s.", index)
            else:
                test_case["input_data"] = generated
                # P1-1: override LLM-reported primary_size with server-calculated value
                if plan is not None:
                    actual_size = _compute_actual_primary_size(generated, plan.input_model, plan.max_q)
                    if actual_size != test_case.get("primary_size"):
                        logger.info(
                            "case %s: primary_size overridden %s → %s (server-calculated)",
                            index, test_case.get("primary_size"), actual_size,
                        )
                    test_case["primary_size"] = actual_size
        logger.info("All generators finished in %sms.", int((time.perf_counter() - gen_started_at) * 1000))

    # P1-1: re-run size constraints with actual primary_size values after generator materialization
    if plan is not None and generator_failure_count == 0:
        for index, tc in enumerate(test_cases, start=1):
            ps = tc.get("primary_size")
            if ps is None:
                issues.append(
                    f"case {index} dominant size could not be inferred from generated input"
                )
            elif ps == 0:
                issues.append(
                    f"case {index} generator produced zero-size input — "
                    "test cases must have at least one element/edge/cell/query"
                )
        target_size = _target_primary_size(plan)
        near_max_threshold = max(2, int(target_size * TESTCASE_NEAR_MAX_RATIO[plan.difficulty])) if target_size > 0 else 0
        anti_naive_threshold = max(2, int(target_size * _anti_naive_ratio(plan))) if target_size > 0 else 0
        size_issues = _validate_size_constraints(plan, test_cases, target_size, near_max_threshold, anti_naive_threshold)
        if size_issues:
            logger.warning("Size re-validation failed after materialization: %s", size_issues)
            issues.extend(size_issues)

    # P2-1: duplicate check on actual generated inputs
    seen_inputs: set[str] = set()
    for index, test_case in enumerate(test_cases, start=1):
        inp = test_case.get("input_data", "")
        if inp and inp in seen_inputs:
            issues.append(f"case {index} has the same input as a previous case after generation")
            logger.warning("Duplicate input detected for testcase %s after generator materialization.", index)
        if inp:
            seen_inputs.add(inp)

    if issues:
        if generator_failure_count > 0:
            issues.append("generator_broken")
        logger.warning(
            "Skipping reference executions because testcase materialization already has issues: %s",
            issues,
        )
        return issues, first_solution_error

    # ── 2. Run all solution executions in parallel ────────────────────────────
    runnable = [(index, tc) for index, tc in enumerate(test_cases, start=1) if tc["input_data"]]
    if runnable:
        sol_started_at = time.perf_counter()
        solution_timeout = _reference_solution_timeout_seconds(plan)
        logger.info(
            "Running %s reference executions in parallel. timeout=%ss",
            len(runnable),
            solution_timeout,
        )
        sol_results = await asyncio.gather(
            *(_run_solution(solution_code, tc["input_data"], timeout=solution_timeout) for _, tc in runnable)
        )
        logger.info("All reference executions finished in %sms.", int((time.perf_counter() - sol_started_at) * 1000))
    else:
        sol_results = []

    for (index, test_case), (actual, error_info) in zip(runnable, sol_results):
        logger.info(
            "Reference result testcase %s/%s. primary_size=%s boundary=%s anti_naive=%s",
            index, len(test_cases),
            test_case.get("primary_size"),
            test_case.get("is_boundary"),
            test_case.get("is_anti_naive"),
        )
        if actual is None:
            key_error = _extract_key_error(error_info)
            failing_line = (
                key_error.split("failed at: `")[1].split("`")[0]
                if "failed at: `" in key_error else ""
            )
            is_input_format_error = (
                any(e in error_info for e in _INPUT_FORMAT_ERRORS)
                or (
                    ("IndexError" in error_info or "ValueError" in error_info)
                    and any(kw in failing_line for kw in _INPUT_PARSE_KEYWORDS)
                )
            )
            if is_input_format_error:
                extra_hint = ""
                if "edges_data" in failing_line or ("data[" in failing_line and plan is not None and plan.input_model == "GRAPH"):
                    extra_hint = (
                        " For GRAPH generators: the loop generating edges MUST run "
                        "EXACTLY M times (M = the value printed in the header). "
                        "Use `for i in range(m): print(u, v)` not `for i in range(n)`."
                    )
                issues.append(
                    f"case {index} input is missing a required field or has wrong format "
                    f"({key_error}) — check input_description and ensure every field is present "
                    f"in the correct order.{extra_hint}"
                )
            else:
                solution_failure_count += 1
                issues.append(f"reference solution logic error on case {index}: {key_error}")
                if not first_solution_error:
                    first_solution_error = error_info
            logger.warning("Reference execution failed for testcase %s/%s. error=%s", index, len(test_cases), key_error)
        else:
            test_case["expected_output"] = actual
            logger.info("Reference execution succeeded for testcase %s/%s.", index, len(test_cases))

    if solution_failure_count > 0:
        issues.append("solution_broken")
    elif generator_failure_count > 0:
        issues.append("generator_broken")
    return issues, first_solution_error


async def _generate_validated_test_cases(
    plan: AlgorithmGenerationPlan,
    spec_text: str,
    solution_code: str,
) -> list[dict[str, Any]]:
    feedback = ""
    last_issues: list[str] = []
    consecutive_format_errors = 0
    last_format_error = ""
    logger.info("Algorithm generation stage started: testcase generation. %s", _plan_brief(plan))

    target_size = _target_primary_size(plan)
    near_max_threshold = max(2, int(target_size * TESTCASE_NEAR_MAX_RATIO[plan.difficulty])) if target_size > 0 else 0
    anti_naive_threshold = max(2, int(target_size * _anti_naive_ratio(plan))) if target_size > 0 else 0

    for attempt in range(1, TESTCASE_RETRY_LIMIT + 1):
        attempt_started_at = time.perf_counter()
        logger.info(
            "Algorithm testcase attempt %s/%s started. category=%s difficulty=%s",
            attempt,
            TESTCASE_RETRY_LIMIT,
            plan.category,
            plan.difficulty,
        )
        size_hint = ""
        if target_size > 0:
            size_hint = (
                f"\n\nSize requirements (must be satisfied):\n"
                f"- At least one test case must have primary_size >= {near_max_threshold} "
                f"(that is >= {int(TESTCASE_NEAR_MAX_RATIO[plan.difficulty] * 100)}% of max={target_size}).\n"
            )
            if _requires_large_anti_naive(plan):
                size_hint += (
                    f"- At least one anti-naive case must have primary_size >= {anti_naive_threshold}.\n"
                )
        needs_generator = _plan_needs_generator(plan)

        generator_mandate = ""
        if needs_generator:
            if plan.input_model == "GRAPH":
                example = (
                    "  n, m = 5, 6\n"
                    "  print(n, m)  # header — add extra fields here if input_description requires (e.g. S)\n"
                    "  # CRITICAL: loop must run EXACTLY m times; all vertices in [1, n]\n"
                    "  for i in range(m):\n"
                    "      u = (i % n) + 1\n"
                    "      v = (u % n) + 1\n"
                    "      print(u, v)  # add weight w here if edges are weighted\n"
                    "  # If input_description has fields AFTER edges (start S, target T), print them here\n"
                    "  # e.g.: print(1)  # starting node S\n"
                )
            elif plan.input_model == "GRID":
                example = (
                    "  r, c = 3, 4\n"
                    "  print(r, c)\n"
                    "  for i in range(r):\n"
                    "      print(' '.join(['1'] * c))\n"
                )
            elif plan.input_model == "INTERVAL":
                example = (
                    "  n = 5\n"
                    "  print(n)\n"
                    "  intervals = [(1,3),(2,5),(4,7),(6,9),(8,10)]\n"
                    "  for l, r in intervals:\n"
                    "      print(l, r)\n"
                )
            elif plan.input_model == "STRING":
                example = (
                    "  # No random — use arithmetic pattern\n"
                    f"  n = {min(plan.max_n or 10, 20)}\n"
                    "  print(n)\n"
                    "  print(''.join(chr(ord('a') + (i % 26)) for i in range(n)))\n"
                )
            else:  # ARRAY / ANSWER_SEARCH
                q_hint = ""
                if plan.max_q:
                    q_hint = (
                        f"  q = {min(plan.max_q, 5)}\n"
                        "  print(q)\n"
                        "  # print each query — check input_description for format\n"
                        "  for i in range(q):\n"
                        "      print(i + 1)\n"
                    )
                example = (
                    f"  n = {min(plan.max_n or 10, 20)}\n"
                    "  print(n)\n"
                    "  print(' '.join(str(i + 1) for i in range(n)))\n"
                    + q_hint
                )
            anti_naive_reminder = ""
            if _requires_large_anti_naive(plan) and anti_naive_threshold > 0:
                anti_naive_reminder = (
                    f"Anti-naive generators MUST produce primary_size >= {anti_naive_threshold}. "
                    "Small anti-naive cases (primary_size < threshold) fail validation.\n"
                )
            generator_mandate = (
                f"\n\nMANDATORY — {plan.input_model} INPUT MODEL:\n"
                "ALL 4 test cases MUST use input_generator. Set input_data to null for every case.\n"
                "NEVER write literal data in input_data — it will exceed the response size limit.\n"
                "Each input_generator must be short deterministic Python 3 code (< 30 lines) "
                "that prints the complete input to stdout, following input_description exactly.\n"
                "Write ALL necessary imports at the top (import math, import string, etc.).\n"
                "Do NOT use random — use arithmetic/formula-based patterns instead.\n"
                f"{anti_naive_reminder}"
                f"Example:\n{example}"
            )

        user_content = (
            "Locked planning JSON:\n"
            f"{json.dumps(plan.__dict__, ensure_ascii=False, indent=2)}\n\n"
            "Problem spec JSON:\n"
            f"{spec_text}\n\n"
            "Create exactly 4 test cases with valid metadata."
            f"{size_hint}"
            f"{generator_mandate}"
        )
        if feedback:
            user_content += f"\n\nPrevious validation issues:\n{feedback}\nFix every issue."

        tc_messages = [
            {"role": "system", "content": TESTCASE_PROMPT},
            {"role": "user", "content": user_content},
        ]
        tc_raw = await request_json_completion(tc_messages, temperature=0.1)
        try:
            tc_data = _parse_json_object(tc_raw, "test cases")
        except RuntimeError:
            logger.warning(
                "Algorithm testcase attempt %s/%s: JSON parse failed (likely response truncated). "
                "category=%s difficulty=%s",
                attempt, TESTCASE_RETRY_LIMIT, plan.category, plan.difficulty,
            )
            last_issues = ["JSON response was truncated because input_data contained too much literal text"]
            feedback = (
                "CRITICAL: Your previous JSON response was CUT OFF because input_data was too large.\n"
                f"For this {plan.input_model} problem, ALL 4 test cases MUST use input_generator.\n"
                "Set input_data to null for every case.\n"
                "input_generator must be short Python 3 code (< 30 lines) that prints the input.\n"
                + (generator_mandate.strip() if generator_mandate else "")
            )
            continue
        test_cases = _extract_test_case_list(tc_data)

        issues = _validate_test_cases(plan, test_cases)
        exec_error = ""
        if not issues:
            issues, exec_error = await _materialize_reference_outputs(solution_code, test_cases, plan)
        if not issues:
            logger.info(
                "Algorithm testcase attempt %s/%s succeeded in %sms.",
                attempt,
                TESTCASE_RETRY_LIMIT,
                int((time.perf_counter() - attempt_started_at) * 1000),
            )
            return test_cases

        logger.warning(
            "Algorithm testcase validation failed for category=%s difficulty=%s: %s",
            plan.category,
            plan.difficulty,
            issues,
        )
        last_issues = issues
        if "solution_broken" in issues:
            has_input_format_errors = any("missing a required field" in i for i in issues)
            if has_input_format_errors:
                # Mixed: some cases have format errors + some have solution crashes.
                # The crashes are likely caused by bad generator output (e.g. vertex index > N),
                # not a real solution bug. Retry test cases instead of regenerating solution.
                logger.warning(
                    "Mixed solution+format failures detected — treating as generator issue, retrying testcases."
                )
                issues = [i for i in issues if i != "solution_broken"]
                if "generator_broken" not in issues:
                    issues.append("generator_broken")
                last_issues = issues
            else:
                raise RuntimeError(
                    f"solution_broken\n{exec_error}\n"
                    f"cases: {'; '.join(i for i in issues if i not in {'solution_broken', 'generator_broken'})}"
                )

        has_format_errors = any("missing a required field" in i for i in issues)
        if has_format_errors:
            consecutive_format_errors += 1
            format_error_sample = next(
                (i for i in issues if "missing a required field" in i), ""
            )
            last_format_error = format_error_sample
        else:
            consecutive_format_errors = 0

        if consecutive_format_errors >= 3:
            raise RuntimeError(
                f"solution_broken\n"
                f"Solution does not handle edge cases robustly — "
                f"same input format error after {consecutive_format_errors} retries: {last_format_error}\n"
                f"cases: {last_format_error}"
            )

        display_issues = [i for i in issues if i != "generator_broken"]
        feedback = _build_testcase_feedback(plan, display_issues)

    raise RuntimeError(f"Failed to generate valid test cases: {'; '.join(last_issues)}")


async def generate_algorithm_problem(difficulty: str, category: str) -> AlgorithmProblem:
    """Generate one algorithm problem through plan, spec, testcase, and reference checks."""
    started_at = time.perf_counter()
    logger.info("Algorithm generation request started. category=%s difficulty=%s", category, difficulty)

    plan = await _generate_validated_plan(difficulty, category)

    spec_started_at = time.perf_counter()
    logger.info("Algorithm generation stage started: problem spec generation. %s", _plan_brief(plan))
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
    logger.info(
        "Algorithm generation stage finished: problem spec generation in %sms. title=%s",
        int((time.perf_counter() - spec_started_at) * 1000),
        normalized_spec["title"],
    )

    solution_code = await _generate_reference_solution(spec_text)

    solution_retry_limit = 3
    test_case_payloads: list[dict[str, Any]] | None = None
    for solution_attempt in range(1, solution_retry_limit + 1):
        try:
            test_case_payloads = await _generate_validated_test_cases(plan, spec_text, solution_code)
            break
        except RuntimeError as error:
            error_str = str(error)
            if "solution_broken" in error_str and solution_attempt < solution_retry_limit:
                error_detail = error_str.replace("solution_broken\n", "", 1)
                logger.warning(
                    "Reference solution appears broken (attempt %s/%s). Regenerating. error=%s",
                    solution_attempt,
                    solution_retry_limit,
                    error_detail[:200],
                )
                solution_code = await _generate_reference_solution(spec_text, error_feedback=error_detail)
            else:
                raise

    if test_case_payloads is None:
        raise RuntimeError("Failed to generate test cases: no solution attempt succeeded.")

    test_cases = [
        AlgorithmTestCase(
            input_data=test_case["input_data"],
            expected_output=test_case["expected_output"],
            is_sample=(index < 2),
            case_order=index + 1,
        )
        for index, test_case in enumerate(test_case_payloads)
    ]

    problem = AlgorithmProblem(
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
    logger.info(
        "Algorithm generation request finished in %sms. category=%s difficulty=%s title=%s testcase_count=%s",
        int((time.perf_counter() - started_at) * 1000),
        category,
        difficulty,
        problem.title,
        len(problem.test_cases),
    )
    return problem


class GenerateAlgorithmRequest(BaseModel):
    """Request body for algorithm problem generation."""

    difficulty: Literal["EASY", "MEDIUM", "HARD"] = Field(
        default="MEDIUM",
        examples=["EASY", "MEDIUM", "HARD"],
    )
    category: str = Field(
        default="구현",
        examples=["구현", "dp", "graph", "정렬", "이분탐색", "greedy", "bfs", "string"],
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
    endpoint_started_at = time.perf_counter()
    logger.info(
        "Incoming /algorithm/generate request. category=%s difficulty=%s",
        request.category,
        request.difficulty,
    )

    try:
        problem = await generate_algorithm_problem(
            difficulty=request.difficulty,
            category=request.category,
        )
    except RuntimeError as error:
        logger.exception(
            "Algorithm generation failed after %sms. category=%s difficulty=%s",
            int((time.perf_counter() - endpoint_started_at) * 1000),
            request.category,
            request.difficulty,
        )
        raise HTTPException(status_code=500, detail=str(error)) from error

    logger.info(
        "/algorithm/generate request succeeded in %sms. category=%s difficulty=%s problem_title=%s",
        int((time.perf_counter() - endpoint_started_at) * 1000),
        request.category,
        request.difficulty,
        problem.title,
    )
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
