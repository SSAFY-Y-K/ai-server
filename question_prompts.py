"""문제 생성, 검수, 수정 단계에 사용할 LLM 메시지를 만든다."""

from __future__ import annotations


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


def build_generation_messages(
    certification_name: str,
    question_count: int,
    context: str,
    *,
    has_context: bool,
) -> list[dict[str, str]]:
    """문제 초안 생성을 위한 system/user 메시지를 구성한다."""

    return [
        {
            "role": "system",
            "content": (
                "너는 자격증 객관식 문제 출제 도우미다. "
                f"{context_policy(has_context)} "
                "문제는 한국어로 작성하고, 객관식 4지선다로 만든다. "
                "문제, 보기, 정답, 해설은 서로 논리적으로 일치해야 한다."
            ),
        },
        {
            "role": "user",
            "content": (
                f"자격증 이름: {certification_name}\n"
                f"생성할 문제 수: {question_count}개\n"
                f"RAG 참고자료 존재 여부: {'있음' if has_context else '없음'}\n\n"
                "요구사항:\n"
                "- 번호는 1번부터 순서대로 작성\n"
                "- 각 문제는 보기 ①, ②, ③, ④를 포함\n"
                "- 각 문제마다 정답과 1~2문장 해설 포함\n"
                "- 답변은 마크다운으로 작성\n"
                "- 같은 문제나 보기를 반복하지 말 것\n"
                "- RAG 참고자료가 있더라도 원문을 그대로 가져오지 말고 변형/신규 작성할 것\n\n"
                f"RAG 검색 컨텍스트:\n{context}"
            ),
        },
    ]


def build_review_messages(
    certification_name: str,
    question_count: int,
    context: str,
    draft: str,
    *,
    has_context: bool,
) -> list[dict[str, str]]:
    """생성된 초안이 요구사항과 RAG 사용 방침을 잘 따르는지 검수하는 메시지를 만든다."""

    return [
        {
            "role": "system",
            "content": (
                "너는 자격증 문제 검수 담당 AI다. "
                f"{context_policy(has_context)} "
                "초안의 문제 수, 4지선다 형식, 정답/해설, 자격증 적합성, 중복 여부를 검수한다. "
                "직접 최종 문제를 다시 쓰지 말고, 개선해야 할 피드백만 간결하게 제시한다."
            ),
        },
        {
            "role": "user",
            "content": (
                f"자격증 이름: {certification_name}\n"
                f"요구 문제 수: {question_count}개\n"
                f"RAG 참고자료 존재 여부: {'있음' if has_context else '없음'}\n\n"
                f"RAG 검색 컨텍스트:\n{context}\n\n"
                f"문제 초안:\n{draft}\n\n"
                "검수 기준:\n"
                "- 문제 수가 요구사항과 일치하는가\n"
                "- 각 문제에 보기 ①, ②, ③, ④가 있는가\n"
                "- 각 문제에 정답과 해설이 있는가\n"
                "- 요청한 자격증과 무관한 다른 자격증 내용이 섞였는가\n"
                "- RAG 참고자료 원문을 그대로 복사한 문제가 있는가\n"
                "- 중복되거나 지나치게 유사한 문제가 있는가"
            ),
        },
    ]


def build_revision_messages(
    certification_name: str,
    question_count: int,
    context: str,
    draft: str,
    review_feedback: str,
    *,
    has_context: bool,
) -> list[dict[str, str]]:
    """검수 피드백을 반영해 최종 문제 세트를 다시 작성하는 메시지를 만든다."""

    return [
        {
            "role": "system",
            "content": (
                "너는 자격증 객관식 문제를 최종 편집하는 AI다. "
                f"{context_policy(has_context)} "
                "검수 피드백을 반영해 완성본만 작성한다. "
                "불필요한 자기 설명 없이 최종 문제 세트만 마크다운으로 출력한다."
            ),
        },
        {
            "role": "user",
            "content": (
                f"자격증 이름: {certification_name}\n"
                f"최종 문제 수: {question_count}개\n"
                f"RAG 참고자료 존재 여부: {'있음' if has_context else '없음'}\n\n"
                f"RAG 검색 컨텍스트:\n{context}\n\n"
                f"문제 초안:\n{draft}\n\n"
                f"검수 피드백:\n{review_feedback}\n\n"
                "최종 작성 조건:\n"
                "- 번호는 1번부터 순서대로 작성\n"
                "- 정확히 요청한 문제 수만 작성\n"
                "- 각 문제는 보기 ①, ②, ③, ④를 포함\n"
                "- 각 문제마다 정답과 1~2문장 해설 포함\n"
                "- 다른 자격증명을 근거로 들거나 섞지 말 것\n"
                "- RAG 참고자료가 있어도 원문을 그대로 복사하지 말 것\n"
                "- 답변은 마크다운으로 작성"
            ),
        },
    ]
