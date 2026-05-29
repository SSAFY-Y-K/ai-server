"""문제 생성, 검수, 수정 단계에 사용할 LLM 메시지를 만든다."""

from __future__ import annotations


def build_generation_messages(
    certification_name: str,
    question_count: int,
    context: str,
) -> list[dict[str, str]]:
    """문제 초안 생성을 위한 system/user 메시지를 구성한다."""

    return [
        {
            "role": "system",
            "content": (
                "너는 자격증 기출문제 출제 도우미다. "
                "반드시 제공된 RAG 검색 컨텍스트를 우선 근거로 사용한다. "
                "문제는 한국어로 작성하고, 실제 기출 스타일에 맞춘 객관식 4지선다로 만든다. "
                "컨텍스트에 없는 사실을 과도하게 만들지 말고, 부족한 부분은 일반화된 개념 문제로 처리한다."
            ),
        },
        {
            "role": "user",
            "content": (
                f"자격증 이름: {certification_name}\n"
                f"생성할 문제 수: {question_count}개\n\n"
                "요구사항:\n"
                "- 번호는 1번부터 순서대로 작성\n"
                "- 각 문제는 보기 ①, ②, ③, ④를 포함\n"
                "- 각 문제마다 정답과 1~2문장 해설 포함\n"
                "- 답변은 마크다운으로 작성\n"
                "- 문제 주제는 검색 컨텍스트 전체에서 다양하게 분산\n\n"
                f"RAG 검색 컨텍스트:\n{context}"
            ),
        },
    ]


def build_review_messages(
    certification_name: str,
    question_count: int,
    context: str,
    draft: str,
) -> list[dict[str, str]]:
    """생성된 초안이 요구사항과 RAG 근거를 잘 따르는지 검수하는 메시지를 만든다."""

    return [
        {
            "role": "system",
            "content": (
                "너는 자격증 문제 검수 담당 AI다. "
                "문제 초안이 RAG 컨텍스트에 근거하는지, 문제 수가 맞는지, "
                "각 문제가 4지선다/정답/해설을 갖추었는지 확인하고 수정 지시를 작성한다. "
                "직접 최종 문제를 다시 쓰지 말고, 개선해야 할 피드백만 간결하게 제시한다."
            ),
        },
        {
            "role": "user",
            "content": (
                f"자격증 이름: {certification_name}\n"
                f"요구 문제 수: {question_count}개\n\n"
                f"RAG 검색 컨텍스트:\n{context}\n\n"
                f"문제 초안:\n{draft}\n\n"
                "검수 기준:\n"
                "- 문제 수가 요구사항과 일치하는가\n"
                "- 각 문제에 보기 ①, ②, ③, ④가 있는가\n"
                "- 각 문제에 정답과 해설이 있는가\n"
                "- 컨텍스트와 동떨어진 내용이나 과도한 추측이 있는가\n"
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
) -> list[dict[str, str]]:
    """검수 피드백을 반영해 최종 문제 세트를 다시 작성하는 메시지를 만든다."""

    return [
        {
            "role": "system",
            "content": (
                "너는 자격증 기출 스타일 문제를 최종 편집하는 AI다. "
                "RAG 컨텍스트와 검수 피드백을 반영해 완성본만 작성한다. "
                "불필요한 자기 설명 없이 최종 문제 세트만 마크다운으로 출력한다."
            ),
        },
        {
            "role": "user",
            "content": (
                f"자격증 이름: {certification_name}\n"
                f"최종 문제 수: {question_count}개\n\n"
                f"RAG 검색 컨텍스트:\n{context}\n\n"
                f"문제 초안:\n{draft}\n\n"
                f"검수 피드백:\n{review_feedback}\n\n"
                "최종 작성 조건:\n"
                "- 번호는 1번부터 순서대로 작성\n"
                "- 정확히 요청한 문제 수만 작성\n"
                "- 각 문제는 보기 ①, ②, ③, ④를 포함\n"
                "- 각 문제마다 정답과 1~2문장 해설 포함\n"
                "- 답변은 마크다운으로 작성"
            ),
        },
    ]
