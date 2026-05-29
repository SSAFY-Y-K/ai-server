"""COMCBT 게시글에서 해설집/교사용 PDF를 찾아 내려받는다."""

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
    """HTML 링크의 URL과 표시 텍스트를 담는다."""

    url: str
    text: str


class AnchorParser(HTMLParser):
    """HTML 문서에서 a 태그 링크를 수집한다."""

    def __init__(self, base_url: str) -> None:
        """상대 URL을 절대 URL로 바꾸기 위한 기준 URL을 저장한다."""

        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.links: list[Link] = []
        self._href: str | None = None
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """a 태그가 시작되면 href를 기록한다."""

        if tag.lower() != "a":
            return

        attrs_dict = dict(attrs)
        href = attrs_dict.get("href")
        if href:
            self._href = urljoin(self.base_url, href)
            self._text_parts = []

    def handle_data(self, data: str) -> None:
        """현재 a 태그 안의 텍스트 조각을 모은다."""

        if self._href:
            self._text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        """a 태그가 끝나면 Link 객체로 저장한다."""

        if tag.lower() != "a" or not self._href:
            return

        text = normalize_space("".join(self._text_parts))
        self.links.append(Link(url=self._href, text=text))
        self._href = None
        self._text_parts = []


def normalize_space(text: str) -> str:
    """여러 공백 문자를 하나의 공백으로 정리한다."""

    return re.sub(r"\s+", " ", text).strip()


def safe_filename(filename: str) -> str:
    """윈도우 파일명에 쓸 수 없는 문자를 제거한다."""

    filename = filename.strip().replace("\u00a0", " ")
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename)
    filename = re.sub(r"\s+", " ", filename)
    return filename[:180].strip(" ._")


def is_detail_url(url: str) -> bool:
    """URL이 COMCBT 상세 게시글 형식인지 판별한다."""

    return re.search(r"/xe/[^/?#]+/\d+(?:[?#].*)?$", url) is not None


def with_page(url: str, page: int) -> str:
    """목록 URL에 page 쿼리 파라미터를 적용한다."""

    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if page > 1:
        query["page"] = [str(page)]
    else:
        query.pop("page", None)
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


class CbtPdfCrawler:
    """COMCBT 목록/상세 페이지를 순회하며 대상 PDF를 다운로드한다."""

    def __init__(
        self,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        delay: float = 1.0,
        overwrite: bool = False,
        dry_run: bool = False,
    ) -> None:
        """출력 경로, 요청 간격, 덮어쓰기 여부 등 크롤링 옵션을 저장한다."""

        self.output_dir = output_dir
        self.delay = delay
        self.overwrite = overwrite
        self.dry_run = dry_run
        self.opener = build_opener(HTTPCookieProcessor())

    def fetch_text(self, url: str) -> str:
        """URL의 HTML 본문을 문자열로 가져온다."""

        request = Request(url, headers={"User-Agent": USER_AGENT})
        with self.opener.open(request, timeout=20) as response:
            body = response.read()
            charset = response.headers.get_content_charset() or "utf-8"
        return body.decode(charset, errors="replace")

    def fetch_bytes(self, url: str, referer: str | None = None) -> bytes:
        """URL의 바이너리 응답을 가져온다."""

        headers = {"User-Agent": USER_AGENT}
        if referer:
            headers["Referer"] = referer

        request = Request(url, headers=headers)
        with self.opener.open(request, timeout=30) as response:
            return response.read()

    def parse_links(self, url: str) -> list[Link]:
        """HTML 페이지에서 모든 링크를 추출한다."""

        parser = AnchorParser(url)
        parser.feed(self.fetch_text(url))
        return parser.links

    def find_article_links(self, list_url: str) -> list[Link]:
        """게시판 목록 페이지에서 상세 게시글 링크만 골라낸다."""

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
        """상세 글에서 해설집 PDF를 우선 찾고 없으면 교사용 PDF를 찾는다."""

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
        """시작 URL에서 게시글을 순회하고 대상 PDF를 다운로드한다."""

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
        """여러 목록 페이지를 순회하며 상세 게시글 링크를 모은다."""

        detail_links: list[Link] = []

        for page in range(1, max_list_pages + 1):
            list_url = with_page(start_url, page)
            print(f"[list] {list_url}")
            detail_links.extend(self.find_article_links(list_url))
            self.sleep()

        return unique_links(detail_links)

    def download_pdf(self, pdf_link: Link, referer: str | None = None) -> Path | None:
        """PDF 링크를 파일로 저장하고 저장 경로를 반환한다."""

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
        """서버에 부담을 주지 않도록 설정된 시간만큼 대기한다."""

        if self.delay > 0:
            time.sleep(self.delay)


def unique_links(links: Iterable[Link]) -> list[Link]:
    """URL 기준으로 중복 링크를 제거한다."""

    seen: set[str] = set()
    unique: list[Link] = []

    for link in links:
        if link.url in seen:
            continue
        seen.add(link.url)
        unique.append(link)

    return unique


def build_arg_parser() -> argparse.ArgumentParser:
    """단일 크롤링 실행에 필요한 CLI 옵션을 정의한다."""

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
    """CLI 인자를 읽어 COMCBT PDF 크롤링을 실행한다."""

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
