from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from openai import AsyncOpenAI

from rag_chroma_store import DEFAULT_EMBED_DIM, hash_embedding


load_dotenv()

OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    "https://gms.ssafy.io/gmsapi/api.openai.com/v1",
)
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1.mini")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_chunks")
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "12"))
DEFAULT_QUESTION_COUNT = int(os.getenv("RAG_QUESTION_COUNT", "20"))


@dataclass(frozen=True)
class Source:
    source: str
    page: int | str
    category: str


@dataclass(frozen=True)
class GeneratedQuestionSet:
    certification_name: str
    question_count: int
    content: str
    sources: list[Source]


class LocalHashEmbeddings(Embeddings):
    def __init__(self, dimension: int = DEFAULT_EMBED_DIM) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [hash_embedding(text, self.dimension) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return hash_embedding(text, self.dimension)


def ensure_api_key(*, prompt: bool = False) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    if prompt:
        api_key = getpass.getpass("GMS KEY를 입력하세요: ")
        os.environ["OPENAI_API_KEY"] = api_key
        return api_key
    raise RuntimeError("OPENAI_API_KEY environment variable is required.")


def detect_collection_embedding() -> str | None:
    sqlite_path = os.path.join(CHROMA_DIR, "chroma.sqlite3")
    if not os.path.exists(sqlite_path):
        return None

    conn = sqlite3.connect(sqlite_path)
    try:
        row = conn.execute(
            "select str_value from collection_metadata where key = 'embedding'"
        ).fetchone()
    finally:
        conn.close()
    return str(row[0]) if row and row[0] else None


def build_embedding_function() -> Embeddings:
    embedding_kind = detect_collection_embedding()
    if embedding_kind and embedding_kind.startswith("local_hash_"):
        dimension = int(embedding_kind.removeprefix("local_hash_"))
        return LocalHashEmbeddings(dimension)

    api_key = ensure_api_key()
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=api_key,
        base_url=OPENAI_BASE_URL,
    )


def build_vector_db() -> Chroma:
    return Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=CHROMA_COLLECTION,
        embedding_function=build_embedding_function(),
    )


def build_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=ensure_api_key(),
        base_url=OPENAI_BASE_URL,
    )


async def retrieve_certification_docs(
    certification_name: str,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> list[Document]:
    vector_db = build_vector_db()
    query = f"{certification_name} 기출문제 핵심 개념 문제 보기 정답 해설"
    try:
        docs = await vector_db.asimilarity_search(
            query,
            k=top_k,
            filter={"category": certification_name},
        )
    except Exception as error:
        raise RuntimeError(
            "Chroma 검색에 실패했습니다. chroma_db가 생성되어 있는지, "
            "벡터 DB 생성 시 사용한 임베딩과 현재 설정이 일치하는지 확인하세요."
        ) from error

    if docs:
        return docs

    return await vector_db.asimilarity_search(query, k=top_k)


def format_context(docs: Sequence[Document]) -> str:
    context_parts: list[str] = []
    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        source = metadata.get("source") or metadata.get("source_path") or "unknown"
        page = metadata.get("page", "unknown")
        category = metadata.get("category", "")
        label = f"[{index}] source={source}, page={page}"
        if category:
            label += f", category={category}"
        context_parts.append(f"{label}\n{doc.page_content}")
    return "\n\n".join(context_parts)


def collect_sources(docs: Sequence[Document]) -> list[Source]:
    seen: set[tuple[str, str, str]] = set()
    sources: list[Source] = []
    for doc in docs:
        metadata = doc.metadata or {}
        source = str(metadata.get("source") or metadata.get("source_path") or "unknown")
        page = metadata.get("page", "unknown")
        category = str(metadata.get("category") or "")
        key = (source, str(page), category)
        if key in seen:
            continue
        seen.add(key)
        sources.append(Source(source=source, page=page, category=category))
    return sources


def build_generation_messages(
    certification_name: str,
    question_count: int,
    context: str,
) -> list[dict[str, str]]:
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


async def generate_questions_for_certification(
    certification_name: str,
    *,
    question_count: int = DEFAULT_QUESTION_COUNT,
    top_k: int = DEFAULT_TOP_K,
) -> GeneratedQuestionSet:
    certification_name = certification_name.strip()
    if not certification_name:
        raise ValueError("certification_name is required.")
    if question_count <= 0:
        raise ValueError("question_count must be greater than 0.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    docs = await retrieve_certification_docs(certification_name, top_k=top_k)
    context = format_context(docs) or "검색된 문서가 없습니다."
    messages = build_generation_messages(certification_name, question_count, context)

    response = await build_client().chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.4,
    )
    content = response.choices[0].message.content or ""
    return GeneratedQuestionSet(
        certification_name=certification_name,
        question_count=question_count,
        content=content,
        sources=collect_sources(docs),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate certification questions using RAG.")
    parser.add_argument("certification_name", help="Certification name, e.g. 정보처리기사")
    parser.add_argument("--count", type=int, default=DEFAULT_QUESTION_COUNT)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    return parser


async def main() -> None:
    args = build_arg_parser().parse_args()
    result = await generate_questions_for_certification(
        args.certification_name,
        question_count=args.count,
        top_k=args.top_k,
    )
    print(result.content)
    if result.sources:
        print("\n검색 출처:")
        for source in result.sources:
            print(f"- {source.source} p.{source.page}")


if __name__ == "__main__":
    asyncio.run(main())
