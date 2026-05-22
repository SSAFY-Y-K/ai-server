from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Protocol


DEFAULT_DOCS_DIR = Path("docs")
DEFAULT_OUTPUT = Path("rag_chunks.jsonl")
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 150


@dataclass(frozen=True)
class PageText:
    page: int
    text: str


@dataclass(frozen=True)
class RagChunk:
    id: str
    source: str
    source_path: str
    category: str
    page: int
    chunk_index: int
    text: str
    char_count: int
    metadata: dict[str, str | int | None]


class PdfBackend(Protocol):
    name: str

    def extract_pages(self, pdf_path: Path) -> list[PageText]:
        ...


class PyMuPdfBackend:
    name = "pymupdf"

    def __init__(self) -> None:
        import fitz

        self._fitz = fitz

    def extract_pages(self, pdf_path: Path) -> list[PageText]:
        pages: list[PageText] = []
        with self._fitz.open(pdf_path) as document:
            for page_index, page in enumerate(document, start=1):
                pages.append(PageText(page=page_index, text=page.get_text("text")))
        return pages


class PyPdfBackend:
    name = "pypdf"

    def __init__(self) -> None:
        from pypdf import PdfReader

        self._pdf_reader = PdfReader

    def extract_pages(self, pdf_path: Path) -> list[PageText]:
        reader = self._pdf_reader(str(pdf_path))
        pages: list[PageText] = []
        for page_index, page in enumerate(reader.pages, start=1):
            pages.append(PageText(page=page_index, text=page.extract_text() or ""))
        return pages


def load_pdf_backend(preferred: str = "auto") -> PdfBackend:
    backends = ["pymupdf", "pypdf"] if preferred == "auto" else [preferred]

    errors: list[str] = []
    for backend in backends:
        try:
            if backend == "pymupdf":
                return PyMuPdfBackend()
            if backend == "pypdf":
                return PyPdfBackend()
        except ImportError as error:
            errors.append(f"{backend}: {error}")

    choices = ", ".join(backends)
    details = "; ".join(errors) if errors else "unknown import error"
    raise RuntimeError(
        f"No usable PDF parser found for {choices}. "
        "Install one with: pip install pymupdf  (recommended) or pip install pypdf. "
        f"Details: {details}"
    )


def iter_pdf_files(docs_dir: Path) -> list[Path]:
    return sorted(path for path in docs_dir.rglob("*.pdf") if path.is_file())


def is_boilerplate_line(line: str) -> bool:
    if not line:
        return False

    boilerplate_patterns = (
        "전자문제집 CBT",
        "www.comcbt.com",
        "최강 자격증 기출문제",
    )
    return any(pattern in line for pattern in boilerplate_patterns)


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = []
    for line in text.split("\n"):
        cleaned = re.sub(r"[ \t]+", " ", line).strip()
        if is_boilerplate_line(cleaned):
            continue
        if cleaned:
            lines.append(cleaned)

    text = "\n".join(lines)
    hangul_range = "\uac00-\ud7a3"
    choice_marks = "\u2460\u2461\u2462\u2463\u2464\u2776\u2777\u2778\u2779\u277a"
    text = re.sub(
        rf"([{choice_marks}][^\n]{{0,140}}?)((?:[1-9]\d?|100)\.\s*[{hangul_range}A-Za-z])",
        r"\1\n\n\2",
        text,
    )
    text = re.sub(
        rf"(?<!\d)(?<=[{hangul_range}A-Za-z).?!])\s*(?=(?:[1-9]\d?|100)\.\s*[{hangul_range}A-Za-z])",
        "\n\n",
        text,
    )
    text = re.sub(rf"(?<=[{hangul_range}A-Za-z0-9).])\s*(?=[{choice_marks}])", " ", text)
    text = re.sub(rf"(?<=[{choice_marks}])\s+", " ", text)
    text = re.sub(r"(?<![.!?。！？])\n(?!\s*(?:\d+[.)]|[가-힣A-Za-z ]{1,30}:))", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    overlap_separator_size = 2 if chunk_overlap else 0
    content_size = chunk_size - chunk_overlap - overlap_separator_size
    if content_size <= 0:
        raise ValueError("chunk_size must leave room for chunk_overlap")
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > content_size:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(split_long_text(paragraph, content_size, 0))
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= content_size:
            current = candidate
            continue

        chunks.append(current.strip())
        current = paragraph

    if current:
        chunks.append(current.strip())

    if chunk_overlap == 0 or len(chunks) <= 1:
        return chunks

    overlapped: list[str] = []
    previous_tail = ""
    for chunk in chunks:
        if previous_tail:
            chunk = f"{previous_tail}\n\n{chunk}"
        overlapped.append(chunk.strip())
        previous_tail = tail_for_overlap(chunk, chunk_overlap)

    return overlapped


def split_long_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(text.rfind(" ", start, end), text.rfind("\n", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)

    return chunks


def tail_for_overlap(text: str, overlap: int) -> str:
    if len(text) <= overlap:
        return text

    tail = text[-overlap:]
    boundary = min(
        [index for index in (tail.find(" "), tail.find("\n")) if index >= 0],
        default=-1,
    )
    if boundary >= 0 and boundary + 1 < len(tail):
        return tail[boundary + 1 :].strip()
    return tail.strip()


def build_chunk_id(pdf_path: Path, page: int, chunk_index: int, text: str) -> str:
    digest = hashlib.sha1(
        f"{pdf_path.as_posix()}:{page}:{chunk_index}:{text[:80]}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{pdf_path.stem}-p{page}-c{chunk_index}-{digest}"


def extract_metadata(pdf_path: Path, docs_dir: Path) -> dict[str, str | int | None]:
    relative = pdf_path.relative_to(docs_dir)
    category = relative.parts[0] if len(relative.parts) > 1 else ""
    stem = pdf_path.stem

    date_match = re.search(r"(19|20)\d{6}", stem)
    document_type_match = re.search(r"\(([^)]+)\)$", stem)

    return {
        "category": category,
        "exam_name": re.sub(r"(19|20)\d{6}.*$", "", stem).strip() or None,
        "exam_date": date_match.group(0) if date_match else None,
        "document_type": document_type_match.group(1) if document_type_match else None,
    }


def parse_pdf(
    pdf_path: Path,
    docs_dir: Path,
    backend: PdfBackend,
    chunk_size: int,
    chunk_overlap: int,
) -> list[RagChunk]:
    metadata = extract_metadata(pdf_path, docs_dir)
    category = str(metadata.get("category") or "")
    relative_path = pdf_path.relative_to(docs_dir).as_posix()
    chunks: list[RagChunk] = []

    for page_text in backend.extract_pages(pdf_path):
        text = normalize_text(page_text.text)
        if not text:
            continue

        for chunk_index, chunk_text in enumerate(
            split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            start=1,
        ):
            chunks.append(
                RagChunk(
                    id=build_chunk_id(pdf_path, page_text.page, chunk_index, chunk_text),
                    source=pdf_path.name,
                    source_path=relative_path,
                    category=category,
                    page=page_text.page,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    char_count=len(chunk_text),
                    metadata=metadata,
                )
            )

    return chunks


def parse_docs(
    docs_dir: Path,
    output_path: Path,
    backend: PdfBackend,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[int, int, list[tuple[Path, str]]]:
    pdf_files = iter_pdf_files(docs_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    failed: list[tuple[Path, str]] = []

    with output_path.open("w", encoding="utf-8") as output_file:
        for index, pdf_path in enumerate(pdf_files, start=1):
            print(f"[{index}/{len(pdf_files)}] parsing {pdf_path}")
            try:
                chunks = parse_pdf(
                    pdf_path=pdf_path,
                    docs_dir=docs_dir,
                    backend=backend,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            except Exception as error:
                failed.append((pdf_path, str(error)))
                print(f"  [failed] {error}", file=sys.stderr)
                continue

            for chunk in chunks:
                output_file.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
            total_chunks += len(chunks)
            print(f"  [ok] {len(chunks)} chunk(s)")

    return len(pdf_files), total_chunks, failed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse PDFs under docs/ into RAG JSONL chunks.")
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument(
        "--backend",
        choices=["auto", "pymupdf", "pypdf"],
        default="auto",
        help="PDF parser backend. auto tries PyMuPDF first, then pypdf.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    docs_dir = args.docs_dir.resolve()
    output_path = args.output.resolve()

    if not docs_dir.exists():
        raise SystemExit(f"Docs directory does not exist: {docs_dir}")

    backend = load_pdf_backend(args.backend)
    print(f"Using PDF backend: {backend.name}")

    pdf_count, chunk_count, failed = parse_docs(
        docs_dir=docs_dir,
        output_path=output_path,
        backend=backend,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print(f"Done. Parsed {pdf_count} PDF(s), wrote {chunk_count} chunk(s) to {output_path}")
    if failed:
        print(f"Failed {len(failed)} PDF(s):", file=sys.stderr)
        for pdf_path, error in failed:
            print(f"- {pdf_path}: {error}", file=sys.stderr)


if __name__ == "__main__":
    main()
