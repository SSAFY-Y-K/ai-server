"""CSV에 정의된 여러 COMCBT PDF 크롤링 작업을 순차 실행한다."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError

from cbt_pdf_crawler import CbtPdfCrawler


DEFAULT_JOBS_PATH = Path("crawl_jobs.csv")


@dataclass(frozen=True)
class CrawlJob:
    """CSV 한 줄에서 읽은 크롤링 작업 설정."""

    start_url: str
    output_dir: Path
    max_articles: int | None
    max_list_pages: int
    delay: float
    overwrite: bool


def parse_optional_int(value: str | None) -> int | None:
    """빈 문자열은 None으로, 값이 있으면 정수로 변환한다."""

    if value is None or value.strip() == "":
        return None
    return int(value)


def parse_int(value: str | None, default: int) -> int:
    """빈 문자열이면 기본값을 쓰고 아니면 정수로 변환한다."""

    if value is None or value.strip() == "":
        return default
    return int(value)


def parse_float(value: str | None, default: float) -> float:
    """빈 문자열이면 기본값을 쓰고 아니면 실수로 변환한다."""

    if value is None or value.strip() == "":
        return default
    return float(value)


def parse_bool(value: str | None, default: bool = False) -> bool:
    """CSV의 true/false 계열 문자열을 bool 값으로 변환한다."""

    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def read_jobs(path: Path) -> list[CrawlJob]:
    """CSV 파일에서 실행할 크롤링 작업 목록을 읽는다."""

    jobs: list[CrawlJob] = []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for line_number, row in enumerate(reader, start=2):
            start_url = (row.get("start_url") or "").strip()
            if not start_url or start_url.startswith("#"):
                continue

            output_dir = Path((row.get("output_dir") or "docs").strip())
            jobs.append(
                CrawlJob(
                    start_url=start_url,
                    output_dir=output_dir,
                    max_articles=parse_optional_int(row.get("max_articles")),
                    max_list_pages=parse_int(row.get("max_list_pages"), default=1),
                    delay=parse_float(row.get("delay"), default=1.0),
                    overwrite=parse_bool(row.get("overwrite"), default=False),
                )
            )

    return jobs


def run_jobs(jobs: list[CrawlJob], dry_run: bool = False) -> int:
    """여러 크롤링 작업을 순서대로 실행하고 매칭된 PDF 수를 합산한다."""

    total = 0

    for index, job in enumerate(jobs, start=1):
        print(f"\n[job {index}/{len(jobs)}] {job.start_url}")
        print(f"  output_dir={job.output_dir}")
        print(
            "  "
            f"max_articles={job.max_articles}, "
            f"max_list_pages={job.max_list_pages}, "
            f"delay={job.delay}, "
            f"overwrite={job.overwrite}, "
            f"dry_run={dry_run}"
        )

        crawler = CbtPdfCrawler(
            output_dir=job.output_dir,
            delay=job.delay,
            overwrite=job.overwrite,
            dry_run=dry_run,
        )

        try:
            downloaded = crawler.crawl(
                start_url=job.start_url,
                max_list_pages=job.max_list_pages,
                max_articles=job.max_articles,
            )
        except (HTTPError, URLError, TimeoutError) as error:
            print(f"  [error] {error}")
            continue

        total += len(downloaded)
        print(f"  [done] {len(downloaded)} PDF(s) matched")

    return total


def build_arg_parser() -> argparse.ArgumentParser:
    """배치 크롤링 CLI 옵션을 정의한다."""

    parser = argparse.ArgumentParser(description="Run multiple COMCBT PDF crawl jobs from CSV.")
    parser.add_argument("--jobs", type=Path, default=DEFAULT_JOBS_PATH)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    """CLI 인자를 읽어 CSV 기반 배치 크롤링을 실행한다."""

    args = build_arg_parser().parse_args()
    if not args.jobs.exists():
        raise SystemExit(f"Jobs file not found: {args.jobs}")

    jobs = read_jobs(args.jobs)
    if not jobs:
        raise SystemExit(f"No jobs found in: {args.jobs}")

    total = run_jobs(jobs, dry_run=args.dry_run)
    print(f"\nAll jobs done. {total} PDF(s) matched.")


if __name__ == "__main__":
    main()
