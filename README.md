# AI Server RAG Utilities

`comcbt.com` 게시판에서 기출문제 해설집 PDF를 다운로드하고, RAG에 넣기 좋은 JSONL 청크로 변환하는 유틸리티입니다.

## 1. PDF 크롤러

`cbt_pdf_crawler.py`는 COMCBT 게시판 목록 또는 상세 글에서 PDF 첨부 파일을 찾아 `docs/` 폴더에 저장합니다. `해설집` PDF를 우선 다운로드하고, 해설집이 없으면 `교사용` PDF를 대신 다운로드합니다.

### 상세 글 하나 다운로드

```powershell
python cbt_pdf_crawler.py --start-url https://www.comcbt.com/xe/r2/7097908
```

### 게시판 목록에서 여러 글 다운로드

```powershell
python cbt_pdf_crawler.py --start-url https://www.comcbt.com/xe/r2 --max-articles 10
```

### 여러 목록 페이지 순회

```powershell
python cbt_pdf_crawler.py --start-url https://www.comcbt.com/xe/r2 --max-list-pages 3
```

### 다운로드 없이 찾은 PDF만 확인

```powershell
python cbt_pdf_crawler.py --start-url https://www.comcbt.com/xe/r2 --max-articles 5 --dry-run
```

## 2. 주요 옵션

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--start-url` | `https://www.comcbt.com/xe/r2` | 게시판 목록 URL 또는 상세 글 URL |
| `--output-dir` | `docs` | PDF 저장 폴더 |
| `--max-list-pages` | `1` | 게시판 목록을 몇 페이지까지 순회할지 설정 |
| `--max-articles` | 없음 | 상세 글 최대 처리 개수 |
| `--delay` | `1.0` | 요청 사이 대기 시간(초) |
| `--overwrite` | 꺼짐 | 이미 저장된 PDF를 다시 다운로드 |
| `--dry-run` | 꺼짐 | 실제 저장 없이 대상 PDF 링크만 출력 |

## 3. 다른 시험 게시판에 적용하기

같은 COMCBT 구조라면 `--start-url`만 바꾸면 됩니다.

```powershell
python cbt_pdf_crawler.py --start-url https://www.comcbt.com/xe/c2 --max-articles 10
python cbt_pdf_crawler.py --start-url https://www.comcbt.com/xe/j1 --max-list-pages 2
```

현재 크롤러는 상세 글에서 PDF 첨부를 찾은 뒤 `해설집`을 우선 다운로드합니다. `해설집`이 없으면 `교사용` PDF를 fallback으로 다운로드하고, 둘 다 없으면 해당 게시글을 건너뜁니다.

## 4. RAG 청크 생성

다운로드한 PDF는 `docs/` 폴더에 저장됩니다. 이후 PDF 파서를 실행하면 `rag_chunks.jsonl`이 생성됩니다.

```powershell
python rag_pdf_parser.py --docs-dir docs --output rag_chunks.jsonl
```

생성되는 JSONL 필드:

| 필드 | 설명 |
| --- | --- |
| `id` | 문서명, 페이지, 청크 번호를 조합한 고유 ID |
| `source` | 원본 PDF 파일명 |
| `page` | PDF 페이지 번호 |
| `chunk_index` | 페이지 안의 청크 번호 |
| `text` | RAG 검색에 사용할 본문 |

## 5. 여러 크롤링 작업 한 번에 실행

여러 게시판 URL, 최대 처리 개수, 저장 폴더를 CSV 파일에 적어두고 한 번에 실행할 수 있습니다.

먼저 예시 파일을 복사해서 작업 파일을 만듭니다.

```powershell
Copy-Item crawl_jobs.example.csv crawl_jobs.csv
```

`crawl_jobs.csv` 형식:

| 컬럼 | 설명 |
| --- | --- |
| `start_url` | 게시판 목록 URL 또는 상세 글 URL |
| `output_dir` | PDF 저장 폴더 |
| `max_articles` | 상세 글 최대 처리 개수. 비워두면 제한 없음 |
| `max_list_pages` | 목록 페이지 순회 개수 |
| `delay` | 요청 사이 대기 시간(초) |
| `overwrite` | 기존 파일을 덮어쓸지 여부. `true` 또는 `false` |

예시:

```csv
start_url,output_dir,max_articles,max_list_pages,delay,overwrite
https://www.comcbt.com/xe/r2,docs/linux_master_2,5,1,1.0,false
https://www.comcbt.com/xe/r2/7097908,docs/linux_master_2,1,1,1.0,false
```

다운로드 대상만 확인:

```powershell
python batch_crawl.py --jobs crawl_jobs.csv --dry-run
```

실제 다운로드:

```powershell
python batch_crawl.py --jobs crawl_jobs.csv
```

## 6. 권장 실행 순서

```powershell
python cbt_pdf_crawler.py --start-url https://www.comcbt.com/xe/r2 --max-articles 10 --dry-run
python cbt_pdf_crawler.py --start-url https://www.comcbt.com/xe/r2 --max-articles 10
python rag_pdf_parser.py --docs-dir docs --output rag_chunks.jsonl
```

먼저 `--dry-run`으로 대상 PDF를 확인한 뒤 실제 다운로드하는 흐름을 권장합니다.
