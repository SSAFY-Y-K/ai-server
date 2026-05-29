"""rag_chunks.jsonl을 읽어 파일 기반 Chroma 벡터 DB로 저장한다."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, TypeVar


DEFAULT_INPUT = Path("rag_chunks.jsonl")
DEFAULT_DB_DIR = Path("chroma_db")
DEFAULT_COLLECTION = "rag_chunks"
DEFAULT_EMBED_DIM = 768
DEFAULT_BATCH_SIZE = 128
T = TypeVar("T")


def load_chunks(input_path: Path) -> list[dict[str, Any]]:
    """JSONL 청크 파일을 읽고 필수 필드가 있는지 검증한다."""

    chunks: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {input_path}:{line_number}: {error}") from error

            if not chunk.get("id"):
                raise ValueError(f"Missing id at {input_path}:{line_number}")
            if not chunk.get("text"):
                raise ValueError(f"Missing text at {input_path}:{line_number}")
            chunks.append(chunk)
    return chunks


def tokenize(text: str) -> list[str]:
    """해시 임베딩에 사용할 한글/영문/숫자 토큰을 추출한다."""

    words = re.findall(r"[가-힣A-Za-z0-9]+", text.lower())
    chars = [char for char in text if "\uac00" <= char <= "\ud7a3"]
    return words + chars


def hash_embedding(text: str, dimension: int) -> list[float]:
    """외부 모델 없이 텍스트를 고정 길이 정규화 벡터로 변환한다."""

    vector = [0.0] * dimension
    for token in tokenize(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % dimension
        sign = 1.0 if value & 1 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def flatten_metadata(chunk: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Chroma에 저장 가능한 단순 타입 metadata로 청크 정보를 펼친다."""

    metadata = dict(chunk.get("metadata") or {})
    metadata.update(
        {
            "source": chunk.get("source"),
            "source_path": chunk.get("source_path"),
            "category": chunk.get("category"),
            "page": chunk.get("page"),
            "chunk_index": chunk.get("chunk_index"),
            "char_count": chunk.get("char_count"),
        }
    )

    flattened: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, str | int | float | bool):
            flattened[key] = value
        else:
            flattened[key] = json.dumps(value, ensure_ascii=False)
    return flattened


def batched(items: list[T], batch_size: int) -> Iterable[list[T]]:
    """긴 목록을 Chroma upsert용 배치로 나눈다."""

    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_chroma_db(
    input_path: Path,
    db_dir: Path,
    collection_name: str,
    embed_dim: int,
    batch_size: int,
    reset: bool,
) -> int:
    """청크를 임베딩하고 Chroma persistent collection에 저장한다."""

    import chromadb

    chunks = load_chunks(input_path)
    db_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_dir))
    if reset:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "description": "RAG chunks generated from rag_chunks.jsonl",
            "embedding": f"local_hash_{embed_dim}",
            "hnsw:space": "cosine",
        },
    )

    for index, batch in enumerate(batched(chunks, batch_size), start=1):
        ids = [str(chunk["id"]) for chunk in batch]
        documents = [str(chunk["text"]) for chunk in batch]
        embeddings = [hash_embedding(document, embed_dim) for document in documents]
        metadatas = [flatten_metadata(chunk) for chunk in batch]

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f"[{index}] upserted {len(batch)} chunk(s)")

    return len(chunks)


def build_arg_parser() -> argparse.ArgumentParser:
    """벡터 DB 생성 옵션을 받는 CLI 파서를 만든다."""

    parser = argparse.ArgumentParser(
        description="Store rag_chunks.jsonl chunks in a persistent local Chroma DB."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embed-dim", type=int, default=DEFAULT_EMBED_DIM)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing collection before inserting chunks.",
    )
    return parser


def main() -> None:
    """CLI 인자를 읽어 Chroma DB 저장 작업을 실행한다."""

    args = build_arg_parser().parse_args()
    if args.embed_dim <= 0:
        raise SystemExit("--embed-dim must be greater than 0")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than 0")

    count = build_chroma_db(
        input_path=args.input,
        db_dir=args.db_dir,
        collection_name=args.collection,
        embed_dim=args.embed_dim,
        batch_size=args.batch_size,
        reset=args.reset,
    )
    print(f"Done. Stored {count} chunk(s) in {args.db_dir.resolve()}")


if __name__ == "__main__":
    main()
