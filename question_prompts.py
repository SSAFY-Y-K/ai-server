"""문제 생성, 검수, 수정 단계에 사용할 LLM 메시지를 만든다."""

from __future__ import annotations

from typing import Literal


SINGLE_PROBLEM_FORMAT_GUIDE = """반드시 요청된 problemType에 맞는 JSON 객체 형식으로만 답하라. 설명, 코드블록, 마크다운을 추가하지 마라.

problemType이 MULTIPLE이면 반드시 MultipleChoiceProblemCreateRequest 구조로 답한다:
{
  "certId": null,
  "title": "문제 제목",
  "question": "문제 내용",
  "choice1Content": "1번 선택지",
  "choice2Content": "2번 선택지",
  "choice3Content": "3번 선택지",
  "choice4Content": "4번 선택지",
  "answerNumber": <정답 선택지 번호>
}

problemType이 SHORT_ANSWER이면 반드시 ShortAnswerProblemCreateRequest 구조로 답한다:
{
  "certId": null,
  "title": "문제 제목",
  "question": "문제",
  "answer": "정답"
}

규칙:
- certId는 현재 입력에 없으므로 반드시 null로 작성
- title은 문제 내용을 짧게 요약한 한국어 제목으로 작성
- MULTIPLE이면 choice1Content~choice4Content를 모두 비어 있지 않은 문자열로 작성하고 answerNumber는 반드시 1~4 숫자
- SHORT_ANSWER이면 answer를 비어 있지 않은 짧고 명확한 문자열로 작성
- 정답은 반드시 명확하게 1개만 존재해야 하며, 여러 답이 가능하거나 해석이 갈리는 문제는 만들지 말 것
- 정답을 영문으로 작성해야 하면 question에 "영문으로 쓰시오"를 명시하고, 한글로 작성해야 하면 question에 "한글로 쓰시오"를 명시
- 요청된 problemType 구조에 없는 필드는 절대 포함하지 말 것
- 이번 호출에서는 문제를 정확히 1개만 생성
- 모든 문제와 보기 텍스트는 한국어로 작성
- JSON 문법이 완전히 유효해야 함"""


def context_policy(has_context: bool) -> str:
    """RAG 검색 결과 유무에 따라 모델이 따라야 할 근거 사용 방침을 만든다."""

    if has_context:
        return (
            "RAG 검색 컨텍스트는 같은 자격증의 참고자료다. "
            "doc_type=syllabus 자료는 과목 범위와 주제 균형을 잡는 데 우선 활용하고, "
            "doc_type=exam_pdf 자료는 난이도와 출제 스타일 참고에만 활용하라. "
            "컨텍스트의 개념, 난이도, 출제 스타일만 참고하고 원문 문제를 그대로 복사하지 마라. "
            "반드시 새 문제를 만들어라."
        )
    return (
        "RAG 검색 컨텍스트가 없다. 다른 자격증 자료를 참고하거나 언급하지 마라. "
        "자격증 이름과 일반적으로 알려진 시험 범위에 맞춰 새 문제를 만들어라. "
        "출처가 없으므로 특정 기출 원문처럼 단정하지 마라."
    )


def problem_type_policy(problem_type: Literal["MULTIPLE", "SHORT_ANSWER"]) -> str:
    """요청된 문제 유형에 맞는 단일 유형 강제 지침을 만든다."""

    if problem_type == "MULTIPLE":
        return (
            '이번 호출에서는 problemType을 반드시 "MULTIPLE"로 작성하라. '
            "객관식 4지선다 1문제만 생성하고 SHORT_ANSWER는 사용하지 마라."
        )
    return (
        '이번 호출에서는 problemType을 반드시 "SHORT_ANSWER"로 작성하라. '
        "주관식 1문제만 생성하고 MULTIPLE은 사용하지 마라."
    )


def build_generation_messages(
    certification_name: str,
    context: str,
    problem_type: Literal["MULTIPLE", "SHORT_ANSWER"],
    *,
    has_context: bool,
) -> list[dict[str, str]]:
    """문제 1개 초안 생성을 위한 system/user 메시지를 구성한다."""

    return [
        {
            "role": "system",
            "content": (
                "너는 자격증 문제 출제 도우미다. "
                f"{context_policy(has_context)} "
                f"{problem_type_policy(problem_type)} "
                "사용자가 요청한 자격증에 맞는 문제 1개를 JSON으로 생성한다. "
                "출력은 반드시 유효한 JSON 객체 하나만 반환한다."
            ),
        },
        {
            "role": "user",
            "content": (
                f"자격증 이름: {certification_name}\n"
                "이번 호출에서 생성할 문제 수: 1개\n"
                f"RAG 참고자료 존재 여부: {'있음' if has_context else '없음'}\n\n"
                "생성 요구사항:\n"
                "- 이번 호출에서는 문제를 정확히 1개만 생성\n"
                "- 자격증 범위와 무관한 다른 자격증 내용을 섞지 말 것\n"
                "- RAG 참고자료가 있더라도 원문을 그대로 가져오지 말고 변형/신규 작성할 것\n"
                f"- 요청된 문제 유형만 사용: {problem_type}\n"
                "- 정답은 반드시 명확하게 1개만 존재하도록 작성\n"
                "- 정답이 영문이면 문제에 영문으로 쓰라고 명시하고, 정답이 한글이면 문제에 한글로 쓰라고 명시\n"
                "- 객관식은 choice1Content~choice4Content를 사용하는 4지선다만 허용\n"
                "- 주관식은 짧고 명확한 정답 문자열을 사용\n"
                f"{SINGLE_PROBLEM_FORMAT_GUIDE}\n\n"
                f"RAG 검색 컨텍스트:\n{context}"
            ),
        },
    ]


def build_review_messages(
    certification_name: str,
    context: str,
    draft: str,
    problem_type: Literal["MULTIPLE", "SHORT_ANSWER"],
    *,
    has_context: bool,
) -> list[dict[str, str]]:
    """생성된 문제 1개 초안이 요구사항과 RAG 사용 방침을 잘 따르는지 검수한다."""

    return [
        {
            "role": "system",
            "content": (
                "너는 자격증 문제 검수 담당 AI다. "
                f"{context_policy(has_context)} "
                f"{problem_type_policy(problem_type)} "
                "초안 JSON의 구조, 문제 수, 문제 유형별 필드 규칙, 자격증 적합성, 중복 여부를 검수한다. "
                "직접 최종 JSON을 다시 쓰지 말고, 개선해야 할 피드백만 간결하게 제시한다."
            ),
        },
        {
            "role": "user",
            "content": (
                f"자격증 이름: {certification_name}\n"
                "이번 호출의 요구 문제 수: 1개\n"
                f"RAG 참고자료 존재 여부: {'있음' if has_context else '없음'}\n\n"
                f"{SINGLE_PROBLEM_FORMAT_GUIDE}\n\n"
                f"RAG 검색 컨텍스트:\n{context}\n\n"
                f"문제 초안 JSON:\n{draft}\n\n"
                "검수 기준:\n"
                "- 실제 문제를 정확히 1개만 담은 JSON 객체인가\n"
                f"- problemType이 요청된 유형({problem_type})과 정확히 일치하는가\n"
                "- 정답이 명확하게 1개만 존재하는가\n"
                "- 정답 언어가 영문이면 문제에 영문으로 쓰라고 명시했고, 한글이면 한글로 쓰라고 명시했는가\n"
                "- MULTIPLE 문제는 MultipleChoiceProblemCreateRequest 구조와 answerNumber 규칙을 지키는가\n"
                "- SHORT_ANSWER 문제는 ShortAnswerProblemCreateRequest 구조와 answer 규칙을 지키는가\n"
                "- 요청한 자격증과 무관한 다른 자격증 내용이 섞였는가\n"
                "- RAG 참고자료 원문을 그대로 복사한 문제가 있는가\n"
                "- JSON으로 바로 파싱하기 어려운 표현이 있는가"
            ),
        },
    ]


def build_revision_messages(
    certification_name: str,
    context: str,
    draft: str,
    review_feedback: str,
    problem_type: Literal["MULTIPLE", "SHORT_ANSWER"],
    *,
    has_context: bool,
) -> list[dict[str, str]]:
    """검수 피드백을 반영해 문제 1개 완성본 JSON을 다시 작성하는 메시지를 만든다."""

    return [
        {
            "role": "system",
            "content": (
                "너는 자격증 문제 JSON을 최종 편집하는 AI다. "
                f"{context_policy(has_context)} "
                f"{problem_type_policy(problem_type)} "
                "검수 피드백을 반영해 완성본 JSON만 작성한다. "
                "설명, 코드블록, 마크다운 없이 유효한 JSON 객체 하나만 출력한다."
            ),
        },
        {
            "role": "user",
            "content": (
                f"자격증 이름: {certification_name}\n"
                "이번 호출의 최종 문제 수: 1개\n"
                f"RAG 참고자료 존재 여부: {'있음' if has_context else '없음'}\n\n"
                f"{SINGLE_PROBLEM_FORMAT_GUIDE}\n\n"
                f"RAG 검색 컨텍스트:\n{context}\n\n"
                f"문제 초안 JSON:\n{draft}\n\n"
                f"검수 피드백:\n{review_feedback}\n\n"
                "최종 작성 조건:\n"
                "- 이번 호출에서는 문제를 정확히 1개만 작성\n"
                f"- problemType은 반드시 {problem_type}\n"
                "- 정답은 반드시 명확하게 1개만 존재하도록 작성\n"
                "- 정답이 영문이면 문제에 영문으로 쓰라고 명시하고, 정답이 한글이면 문제에 한글로 쓰라고 명시\n"
                "- 다른 자격증명을 근거로 들거나 섞지 말 것\n"
                "- RAG 참고자료가 있어도 원문을 그대로 복사하지 말 것\n"
                "- 최종 답변은 JSON 객체 하나만 출력"
            ),
        },
    ]
