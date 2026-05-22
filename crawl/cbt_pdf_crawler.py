from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse
from urllib.request import HTTPCookieProcessor, Request, build_opener


DEFAULT_START_URL = "https://www.comcbt.com/xe/r2"
DEFAULT_OUTPUT_DIR = Path("docs")
USER_AGENT = "Mozilla/5.0 (compatible; ai_server-rag-crawler/1.0)"


@dataclass(frozen=True)
class Link:
    url: str
    text: str


class AnchorParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.links: list[Link] = []
        self._href: str | None = None
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return

        attrs_dict = dict(attrs)
        href = attrs_dict.get("href")
        if href:
            self._href = urljoin(self.base_url, href)
            self._text_parts = []

    def handle_data(self, data: str) -> None:
        if self._href:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or not self._href:
            return

        text = normalize_space("".join(self._text_parts))
        self.links.append(Link(url=self._href, text=text))
        self._href = None
        self._text_parts = []


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def safe_filename(filename: str) -> str:
    filename = filename.strip().replace("\u00a0", " ")
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename)
    filename = re.sub(r"\s+", " ", filename)
    return filename[:180].strip(" ._")


def is_detail_url(url: str) -> bool:
    return re.search(r"/xe/[^/?#]+/\d+(?:[?#].*)?$", url) is not None


def with_page(url: str, page: int) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if page > 1:
        query["page"] = [str(page)]
    else:
        query.pop("page", None)
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


class CbtPdfCrawler:
    def __init__(
        self,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        delay: float = 1.0,
        overwrite: bool = False,
        dry_run: bool = False,
    ) -> None:
        self.output_dir = output_dir
        self.delay = delay
        self.overwrite = overwrite
        self.dry_run = dry_run
        self.opener = build_opener(HTTPCookieProcessor())

    def fetch_text(self, url: str) -> str:
        request = Request(url, headers={"User-Agent": USER_AGENT})
        with self.opener.open(request, timeout=20) as response:
            body = response.read()
            charset = response.headers.get_content_charset() or "utf-8"
        return body.decode(charset, errors="replace")

    def fetch_bytes(self, url: str, referer: str | None = None) -> bytes:
        headers = {"User-Agent": USER_AGENT}
        if referer:
            headers["Referer"] = referer

        request = Request(url, headers=headers)
        with self.opener.open(request, timeout=30) as response:
            return response.read()

    def parse_links(self, url: str) -> list[Link]:
        parser = AnchorParser(url)
        parser.feed(self.fetch_text(url))
        return parser.links

    def find_article_links(self, list_url: str) -> list[Link]:
        board_path = urlparse(list_url).path.rstrip("/")
        links = []

        for link in self.parse_links(list_url):
            parsed = urlparse(link.url)
            if parsed.netloc != urlparse(list_url).netloc:
                continue
            if not re.fullmatch(rf"{re.escape(board_path)}/\d+", parsed.path.rstrip("/")):
                continue
            if "공지" in link.text:
                continue
            links.append(link)

        return unique_links(links)

    def find_target_pdf_links(self, detail_url: str) -> list[Link]:
        pdf_links = []

        for link in self.parse_links(detail_url):
            text = link.text
            parsed_url = urlparse(link.url)
            looks_like_pdf = text.lower().endswith(".pdf") or ".pdf" in parsed_url.query.lower()

            if looks_like_pdf:
                pdf_links.append(link)

        explanation_links = [link for link in pdf_links if "해설집" in link.text]
        if explanation_links:
            print("  [type] 해설집")
            return unique_links(explanation_links)

        teacher_links = [link for link in pdf_links if "교사용" in link.text]
        if teacher_links:
            print("  [type] 교사용 fallback")
            return unique_links(teacher_links)

        print("  [skip] no 해설집 or 교사용 PDF found")
        return []

    def crawl(
        self,
        start_url: str,
        max_list_pages: int = 1,
        max_articles: int | None = None,
    ) -> list[Path]:
        detail_links = (
            [Link(url=start_url, text=start_url)]
            if is_detail_url(start_url)
            else self.collect_detail_links(start_url, max_list_pages=max_list_pages)
        )

        if max_articles is not None:
            detail_links = detail_links[:max_articles]

        downloaded: list[Path] = []
        for detail in detail_links:
            print(f"[article] {detail.text or detail.url}")
            for pdf_link in self.find_target_pdf_links(detail.url):
                target = self.download_pdf(pdf_link, referer=detail.url)
                if target:
                    downloaded.append(target)
                self.sleep()
            self.sleep()

        return downloaded

    def collect_detail_links(self, start_url: str, max_list_pages: int) -> list[Link]:
        detail_links: list[Link] = []

        for page in range(1, max_list_pages + 1):
            list_url = with_page(start_url, page)
            print(f"[list] {list_url}")
            detail_links.extend(self.find_article_links(list_url))
            self.sleep()

        return unique_links(detail_links)

    def download_pdf(self, pdf_link: Link, referer: str | None = None) -> Path | None:
        filename = safe_filename(pdf_link.text)
        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"

        target = self.output_dir / filename
        print(f"  [pdf] {filename}")

        if self.dry_run:
            print(f"  [dry-run] {pdf_link.url}")
            return target

        if target.exists() and not self.overwrite:
            print("  [skip] already exists")
            return target

        self.output_dir.mkdir(parents=True, exist_ok=True)
        content = self.fetch_bytes(pdf_link.url, referer=referer)
        if not content.startswith(b"%PDF"):
            print("  [skip] response is not a PDF")
            return None

        target.write_bytes(content)
        print(f"  [saved] {target}")
        return target

    def sleep(self) -> None:
        if self.delay > 0:
            time.sleep(self.delay)


def unique_links(links: Iterable[Link]) -> list[Link]:
    seen: set[str] = set()
    unique: list[Link] = []

    for link in links:
        if link.url in seen:
            continue
        seen.add(link.url)
        unique.append(link)

    return unique


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download 해설집 PDFs, or 교사용 PDFs when 해설집 is unavailable."
    )
    parser.add_argument("--start-url", default=DEFAULT_START_URL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-list-pages", type=int, default=1)
    parser.add_argument("--max-articles", type=int)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    crawler = CbtPdfCrawler(
        output_dir=args.output_dir,
        delay=args.delay,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )

    try:
        downloaded = crawler.crawl(
            start_url=args.start_url,
            max_list_pages=args.max_list_pages,
            max_articles=args.max_articles,
        )
    except (HTTPError, URLError, TimeoutError) as error:
        raise SystemExit(f"Crawl failed: {error}") from error

    print(f"Done. {len(downloaded)} PDF(s) matched.")


if __name__ == "__main__":
    main()
