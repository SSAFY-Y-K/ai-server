# AI Server RAG Utilities

`comcbt.com` 게시판에서 기출문제 해설집 PDF를 다운로드하고, RAG에 넣기 좋은 JSONL 청크로 변환하는 유틸리티입니다.

## 1. PDF 수집

`cbt_pdf_crawler.py`는 COMCBT 게시글에서 PDF 첨부를 찾아 `docs/`에 저장하는 보조 스크립트입니다. `해설집`을 우선 받고, 없으면 `교사용` PDF를 받습니다.

```powershell
python cbt_pdf_crawler.py --start-url https://www.comcbt.com/xe/r2 --max-articles 10 --dry-run
python cbt_pdf_crawler.py --start-url https://www.comcbt.com/xe/r2 --max-articles 10
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--start-url` | `https://www.comcbt.com/xe/r2` | 게시판 목록 URL 또는 상세 글 URL |
| `--output-dir` | `docs` | PDF 저장 폴더 |
| `--max-list-pages` | `1` | 게시판 목록을 몇 페이지까지 순회할지 설정 |
| `--max-articles` | 없음 | 상세 글 최대 처리 개수 |
| `--delay` | `1.0` | 요청 사이 대기 시간(초) |
| `--overwrite` | 꺼짐 | 이미 저장된 PDF를 다시 다운로드 |
| `--dry-run` | 꺼짐 | 실제 저장 없이 대상 PDF 링크만 출력 |

다른 시험 게시판도 같은 구조라면 `--start-url`만 바꾸면 됩니다.

## 2. RAG 청크 생성

다운로드한 PDF는 `docs/` 폴더에 저장됩니다. 이후 PDF 파서를 실행하면 `rag_chunks.jsonl`이 생성됩니다.
이 파서는 PDF 본문을 추출한 뒤 COMCBT 반복 문구를 제거하고, 문항 번호와 보기 기호 경계를 보정한 다음 페이지 단위로 RAG에 넣기 쉬운 청크를 만듭니다.

```powershell
python rag_pdf_parser.py --docs-dir docs --output rag_chunks.jsonl
```

필요하면 청크 크기와 overlap을 조정할 수 있습니다.

```powershell
python rag_pdf_parser.py --docs-dir docs --output rag_chunks.jsonl --chunk-size 1200 --chunk-overlap 150
```

생성되는 JSONL 필드:

| 필드 | 설명 |
| --- | --- |
| `id` | 문서명, 페이지, 청크 번호를 조합한 고유 ID |
| `source` | 원본 PDF 파일명 |
| `source_path` | `docs/` 기준 원본 PDF 경로 |
| `category` | `docs/` 하위 폴더명 |
| `page` | PDF 페이지 번호 |
| `chunk_index` | 페이지 안의 청크 번호 |
| `text` | RAG 검색에 사용할 본문 |
| `metadata` | 시험명, 시험일, 문서 유형 등 보조 정보 |

## 3. 벡터 DB 저장

`rag_chroma_store.py`는 `rag_chunks.jsonl`을 읽어 Chroma 로컬 벡터 DB로 저장합니다. 기본 출력 폴더는 `chroma_db/`입니다.

```powershell
python rag_chroma_store.py --input rag_chunks.jsonl --db-dir chroma_db --reset
```

## 4. 권장 실행 순서

```powershell
python cbt_pdf_crawler.py --start-url https://www.comcbt.com/xe/r2 --max-articles 10 --dry-run
python cbt_pdf_crawler.py --start-url https://www.comcbt.com/xe/r2 --max-articles 10
python rag_pdf_parser.py --docs-dir docs --output rag_chunks.jsonl
python rag_chroma_store.py --input rag_chunks.jsonl --db-dir chroma_db --reset
```

먼저 `--dry-run`으로 대상 PDF를 확인한 뒤 실제 다운로드하는 흐름을 권장합니다.

---

### [PDF 및 파싱 텍스트 다운로드](https://drive.google.com/drive/folders/1BZeehYWFoxrqVpYpuIM7dGBwnBV7onOp?usp=drive_link)
