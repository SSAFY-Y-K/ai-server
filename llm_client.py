"""OpenAI 호환 채팅 모델 호출을 담당한다."""

from __future__ import annotations

from openai import AsyncOpenAI

from rag_config import CHAT_MODEL, OPENAI_BASE_URL, ensure_api_key


def build_client() -> AsyncOpenAI:
    """문제 생성을 호출할 OpenAI 호환 비동기 클라이언트를 만든다."""

    return AsyncOpenAI(
        api_key=ensure_api_key(),
        base_url=OPENAI_BASE_URL,
    )


_SUPPORTS_TEMPERATURE = not CHAT_MODEL.startswith("gpt-5")


async def request_chat_completion(
    messages: list[dict[str, str]],
    *,
    temperature: float,
) -> str:
    """OpenAI 호환 채팅 모델에 메시지를 보내고 텍스트 응답을 반환한다."""

    kwargs: dict = {"model": CHAT_MODEL, "messages": messages}
    if _SUPPORTS_TEMPERATURE:
        kwargs["temperature"] = temperature
    response = await build_client().chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


async def request_json_completion(
    messages: list[dict[str, str]],
    *,
    temperature: float,
) -> str:
    """JSON 객체 반환을 강제한 채팅 모델 호출로 문자열 JSON을 반환한다."""

    kwargs: dict = {
        "model": CHAT_MODEL,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    if _SUPPORTS_TEMPERATURE:
        kwargs["temperature"] = temperature
    response = await build_client().chat.completions.create(**kwargs)
    return response.choices[0].message.content or "{}"
