# AI Server RAG Utilities

`comcbt.com` 게시판에서 기출문제 해설집 PDF를 다운로드하고, RAG에 넣기 좋은 JSONL 청크로 변환하는 유틸리티입니다.

## 1. PDF 수집

`cbt_pdf_crawler.py`는 COMCBT 게시글에서 PDF 첨부를 찾아 `docs/`에 저장하는 보조 스크립트입니다. `해설집`을 우선 받고, 없으면 `교사용` PDF를 받습니다.

```powershell
python cbt_pdf_crawler.py --start-url https://www.comcbt.com/xe/r2 --max-articles 10 --dry-run
python cbt_pdf_crawler.py --start-url https://www.comcbt.com/xe/r2 --max-articles 10
```

주요 옵션:

| 옵션               | 기본값                         | 설명                                      |
| ------------------ | ------------------------------ | ----------------------------------------- |
| `--start-url`      | `https://www.comcbt.com/xe/r2` | 게시판 목록 URL 또는 상세 글 URL          |
| `--output-dir`     | `docs`                         | PDF 저장 폴더                             |
| `--max-list-pages` | `1`                            | 게시판 목록을 몇 페이지까지 순회할지 설정 |
| `--max-articles`   | 없음                           | 상세 글 최대 처리 개수                    |
| `--delay`          | `1.0`                          | 요청 사이 대기 시간(초)                   |
| `--overwrite`      | 꺼짐                           | 이미 저장된 PDF를 다시 다운로드           |
| `--dry-run`        | 꺼짐                           | 실제 저장 없이 대상 PDF 링크만 출력       |

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

| 필드          | 설명                                       |
| ------------- | ------------------------------------------ |
| `id`          | 문서명, 페이지, 청크 번호를 조합한 고유 ID |
| `source`      | 원본 PDF 파일명                            |
| `source_path` | `docs/` 기준 원본 PDF 경로                 |
| `category`    | `docs/` 하위 폴더명                        |
| `page`        | PDF 페이지 번호                            |
| `chunk_index` | 페이지 안의 청크 번호                      |
| `text`        | RAG 검색에 사용할 본문                     |
| `metadata`    | 시험명, 시험일, 문서 유형 등 보조 정보     |

## 3. 벡터 DB 저장

`rag_chroma_store.py`는 `rag_chunks.jsonl`과 `docs/syllabus/*.md`를 읽어 Chroma 로컬 벡터 DB로 저장합니다. 기본 출력 폴더는 `chroma_db/`입니다.

`docs/syllabus`의 마크다운 파일은 자격증별 과목/범위 참고자료입니다. 파일명은 자격증명으로 사용되고, `## 과목명` 단위로 나뉘어 `doc_type=syllabus` 청크로 저장됩니다. PDF 기출 청크는 `doc_type=exam_pdf`로 저장됩니다.

```powershell
python rag_chroma_store.py --input rag_chunks.jsonl --syllabus-dir docs/syllabus --db-dir chroma_db --reset
```

## 4. 문제 생성 API

`main.py`는 FastAPI 서버입니다. 자격증 이름을 받으면 `chat_with_rag.py`가 Chroma에서 같은 자격증의 syllabus와 기출 청크만 검색하고, 검색 컨텍스트를 바탕으로 문제 JSON을 생성합니다. RAG에 해당 자격증이 없으면 다른 자격증 자료를 참조하지 않고 일반 범위 기반으로 문제를 만듭니다.

문제 생성은 `초안 생성 -> 검수 AI 피드백 -> 피드백 반영 최종 수정` 순서로 진행됩니다. syllabus는 과목 범위와 주제 균형에, PDF 기출은 난이도와 출제 스타일 참고에 사용합니다. 요청은 한 번에 문제 1개만 생성하며, `problemType`으로 객관식(`MULTIPLE_CHOICE`) 또는 주관식(`SHORT_ANSWER`)을 지정합니다. `CODING`은 현재 이 엔드포인트에서 지원하지 않습니다. `MULTIPLE_CHOICE` 요청은 평평한 객관식 JSON으로, `SHORT_ANSWER` 요청은 `question`과 `answer`만 가진 평평한 JSON으로 반환됩니다.

```powershell
uvicorn main:app --reload
```

요청 예시:

```http
POST /questions/generate
```

```json
{
	"certification": "정보처리기사",
	"referenceText": null,
	"problemType": "MULTIPLE_CHOICE"
}
```

`referenceText`는 선택 입력입니다. `null`이면 무시하고, 문자열이 있으면 해당 내용을 우선 참고해 문제를 생성합니다.

```json
{
	"certification": "정보처리기사",
	"referenceText": "운영체제의 프로세스 스케줄링과 교착 상태 개념을 참고해 문제를 생성해 주세요.",
	"problemType": "SHORT_ANSWER"
}
```

`problemType`은 `MULTIPLE_CHOICE`, `SHORT_ANSWER`, `CODING` 중 하나를 받을 수 있지만, 현재 `/questions/generate`에서는 `MULTIPLE_CHOICE`와 `SHORT_ANSWER`만 지원합니다.

```json
{
  "question": "다음 중 정보처리기사 시험에서 소프트웨어 개발 방법론에 대한 설명으로 옳지 않은 것은 무엇인가?",
  "choice1Content": "애자일 방법론은 유연성과 반복성을 강조한다.",
  "choice2Content": "폭포수 모델은 단계별 순차적 개발 프로세스를 따른다.",
  "choice3Content": "프로토타입 모델은 사용자 요구사항을 빠르게 반영하는 데 적합하다.",
  "choice4Content": "기능 중심 설계는 전체 시스템을 먼저 설계한 후 개발하는 방식이다.",
  "answerNumber": 4
}
```

주관식 응답 예시:

```json
{
  "question": "정보처리기사 시험에서 분할과 정복 기법이 포함된 소프트웨어 개발 방법론은 무엇인가?",
  "answer": "구조적 방법론"
}
```

## 5. 권장 실행 순서

```powershell
python ./crawl/cbt_pdf_crawler.py --start-url https://www.comcbt.com/xe/r2 --max-articles 10 --dry-run
python ./crawl/cbt_pdf_crawler.py --start-url https://www.comcbt.com/xe/r2 --max-articles 10
python rag_pdf_parser.py --docs-dir docs --output rag_chunks.jsonl
python rag_chroma_store.py --input rag_chunks.jsonl --syllabus-dir docs/syllabus --db-dir chroma_db --reset
uvicorn main:app --reload
```

먼저 `--dry-run`으로 대상 PDF를 확인한 뒤 실제 다운로드하는 흐름을 권장합니다.

---

### [PDF 및 파싱 텍스트 다운로드](https://drive.google.com/drive/folders/1BZeehYWFoxrqVpYpuIM7dGBwnBV7onOp?usp=drive_link)
