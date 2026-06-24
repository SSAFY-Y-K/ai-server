# AI Server

자격증 기출/출제범위 기반 문제 생성과 알고리즘 코딩 문제 생성을 담당하는 FastAPI 서버입니다.

이 README는 제출용 소개보다 팀원이 나중에 코드를 다시 볼 때 빠르게 구조를 파악하고, 데이터 갱신이나 서버 실행을 이어갈 수 있도록 정리한 운영 문서입니다.

## 전체 흐름

```text
COMCBT PDF 수집
  -> PDF 텍스트 파싱
  -> rag_chunks.jsonl 생성
  -> docs/syllabus/*.md와 함께 Chroma DB 저장
  -> FastAPI에서 RAG 검색
  -> LLM으로 문제 생성 및 검수
```

알고리즘 문제 생성은 별도 RAG 자료를 사용하지 않고, `main.py` 내부의 생성/검증 파이프라인으로 문제 명세, 레퍼런스 풀이, 테스트케이스를 생성합니다.

## 주요 파일

| 경로 | 역할 |
| --- | --- |
| `main.py` | FastAPI 앱 진입점. 자격증 문제 생성 API와 알고리즘 문제 생성 API를 제공 |
| `chat_with_rag.py` | 자격증 문제 생성 흐름 조립. RAG 검색, 초안 생성, 검수, 최종 수정 수행 |
| `question_prompts.py` | 자격증 문제 생성/검수/수정 프롬프트 |
| `rag_retriever.py` | Chroma DB에서 자격증별 syllabus와 기출 청크 검색 |
| `rag_models.py` | 자격증 문제 생성 응답 모델 |
| `rag_config.py` | OpenAI 호환 API, 모델명, Chroma 경로 등 환경설정 |
| `rag_embeddings.py` | Chroma DB와 호환되는 임베딩 함수 선택 |
| `rag_pdf_parser.py` | `docs/` 아래 PDF를 읽어 `rag_chunks.jsonl` 생성 |
| `rag_chroma_store.py` | JSONL 청크와 syllabus 문서를 Chroma DB로 저장 |
| `crawl/cbt_pdf_crawler.py` | COMCBT 게시글에서 PDF 첨부파일 다운로드 |
| `test_validation.py` | 알고리즘 테스트케이스/검증 관련 테스트 |
| `Dockerfile` | 배포 컨테이너 이미지 정의 |

## 데이터 디렉터리

| 경로 | 설명 |
| --- | --- |
| `docs/` | 원본 PDF와 syllabus 문서가 들어가는 기준 디렉터리 |
| `docs/syllabus/` | 자격증별 출제범위 마크다운. 파일명이 자격증명으로 사용됨 |
| `rag_chunks.jsonl` | PDF 파싱 결과. Chroma DB를 재생성할 때 입력으로 사용 |
| `chroma_db/` | 로컬 Chroma persistent DB |

`docs/syllabus/*.md`는 `## 과목명` 단위로 쪼개져 `doc_type=syllabus` 청크로 저장됩니다. PDF에서 만든 청크는 `doc_type=exam_pdf`로 저장됩니다.

## 환경설정

`.env` 또는 실행 환경에 아래 값을 설정합니다.

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `OPENAI_API_KEY` | 없음 | 필수. OpenAI 호환 API 키 |
| `OPENAI_BASE_URL` | `https://gms.ssafy.io/gmsapi/api.openai.com/v1` | OpenAI 호환 API 엔드포인트 |
| `OPENAI_CHAT_MODEL` | `gpt-4.1` | 문제 생성에 사용하는 채팅 모델 |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI 임베딩 사용 시 모델명 |
| `CHROMA_DIR` | `./chroma_db` | Chroma DB 경로 |
| `CHROMA_COLLECTION` | `rag_chunks` | Chroma 컬렉션명 |
| `RAG_TOP_K` | `12` | 자격증 문제 생성 시 검색할 기출 청크 수 |
| `RAG_QUESTION_COUNT` | `20` | 현재 설정값만 존재하며 API에서는 단일 문제 생성 흐름을 사용 |

주의할 점:

- API 키는 로컬 `.env`에만 두고 README나 코드에 직접 적지 않습니다.
- 기존 `chroma_db`가 로컬 해시 임베딩으로 생성된 경우 `rag_embeddings.py`가 이를 감지해 같은 방식으로 검색합니다.
- OpenAI 임베딩으로 만들어진 DB를 사용할 경우, DB 생성 시점과 서버 실행 시점의 임베딩 설정이 일치해야 합니다.

## 로컬 실행

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload
```

서버는 기본적으로 `http://127.0.0.1:8000`에서 실행됩니다.

Docker로 실행할 때는 이미지 빌드 후 8000 포트를 연결합니다.

```powershell
docker build -t ai-server .
docker run --env-file .env -p 8000:8000 ai-server
```

## RAG 데이터 갱신 절차

자격증 PDF나 syllabus를 갱신한 경우 아래 순서로 다시 생성합니다.

### 1. PDF 수집

```powershell
python .\crawl\cbt_pdf_crawler.py --start-url https://www.comcbt.com/xe/r2 --max-articles 10 --dry-run
python .\crawl\cbt_pdf_crawler.py --start-url https://www.comcbt.com/xe/r2 --max-articles 10
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--start-url` | `https://www.comcbt.com/xe/r2` | 게시판 목록 URL 또는 상세 글 URL |
| `--output-dir` | `docs` | PDF 저장 폴더 |
| `--max-list-pages` | `1` | 게시판 목록 순회 페이지 수 |
| `--max-articles` | 없음 | 상세 글 최대 처리 개수 |
| `--delay` | `1.0` | 요청 사이 대기 시간(초) |
| `--overwrite` | 꺼짐 | 이미 저장된 PDF 다시 다운로드 |
| `--dry-run` | 꺼짐 | 실제 저장 없이 대상 PDF 링크만 출력 |

### 2. PDF 청크 생성

```powershell
python .\rag_pdf_parser.py --docs-dir docs --output rag_chunks.jsonl
```

청크 크기를 조정해야 하면 아래 옵션을 사용합니다.

```powershell
python .\rag_pdf_parser.py --docs-dir docs --output rag_chunks.jsonl --chunk-size 1200 --chunk-overlap 150
```

`rag_chunks.jsonl`의 주요 필드:

| 필드 | 설명 |
| --- | --- |
| `id` | 문서명, 페이지, 청크 번호를 조합한 고유 ID |
| `source` | 원본 PDF 파일명 |
| `source_path` | `docs/` 기준 원본 PDF 경로 |
| `category` | `docs/` 하위 폴더명. RAG 필터의 자격증명으로 사용 |
| `page` | PDF 페이지 번호 |
| `chunk_index` | 페이지 안의 청크 번호 |
| `text` | RAG 검색에 사용할 본문 |
| `metadata` | 시험명, 시험일, 문서 유형 등 보조 정보 |

### 3. Chroma DB 재생성

```powershell
python .\rag_chroma_store.py --input rag_chunks.jsonl --syllabus-dir docs/syllabus --db-dir chroma_db --reset
```

`--reset`은 기존 컬렉션을 삭제하고 다시 넣습니다. 일부 자료만 추가하는 목적이 아니라면 데이터 불일치를 줄이기 위해 `--reset`을 사용하는 편이 관리하기 쉽습니다.

## API

### 상태 확인

```http
GET /
```

서버가 떠 있으면 간단한 상태 메시지를 반환합니다.

### 자격증 문제 생성

```http
POST /questions/generate
```

요청:

```json
{
  "certification": "정보처리기사",
  "referenceText": null,
  "problemType": "MULTIPLE_CHOICE"
}
```

필드 설명:

| 필드 | 설명 |
| --- | --- |
| `certification` | 검색할 자격증명. `docs/` 하위 폴더명과 `docs/syllabus/*.md` 파일명과 맞추는 것이 좋음 |
| `referenceText` | 선택값. 특정 개념이나 방향을 우선 참고시키고 싶을 때 사용 |
| `problemType` | `MULTIPLE_CHOICE`, `SHORT_ANSWER`, `CODING` 중 하나. 현재 이 API는 앞의 두 타입만 처리 |

객관식 응답:

```json
{
  "certId": null,
  "title": "문제 제목",
  "question": "문제 본문",
  "choice1Content": "보기 1",
  "choice2Content": "보기 2",
  "choice3Content": "보기 3",
  "choice4Content": "보기 4",
  "answerNumber": 2
}
```

주관식 응답:

```json
{
  "certId": null,
  "title": "문제 제목",
  "question": "문제 본문",
  "answer": "정답"
}
```

내부 흐름:

1. `rag_retriever.py`가 같은 자격증명의 `syllabus`와 `exam_pdf` 청크만 검색합니다.
2. `question_prompts.py`의 생성 프롬프트로 초안을 만듭니다.
3. 검수 프롬프트로 오류와 품질을 점검합니다.
4. 수정 프롬프트로 최종 JSON을 다시 생성합니다.
5. `rag_models.py`의 Pydantic 모델로 응답 형태를 검증합니다.

### 알고리즘 문제 생성

```http
POST /algorithm/generate
```

요청:

```json
{
  "difficulty": "MEDIUM",
  "category": "dp",
  "language": "java"
}
```

지원 난이도:

| 값 | 설명 |
| --- | --- |
| `EASY` | 단순 아이디어, 가벼운 구현. opt_tc는 LINEAR/LOG_N/N_LOG_N/RC/V_PLUS_E만 허용 |
| `MEDIUM` | 표준 알고리즘을 정확히 파악하고 구현해야 통과 |
| `HARD` | 핵심 아이디어 발견이 어렵거나 엣지케이스 처리가 복잡 |

지원 카테고리:

| 값 | 허용 input_model | 허용 opt_tc |
| --- | --- | --- |
| `구현` | ARRAY, GRID, STRING, INTERVAL | LINEAR, N_LOG_N, N2, RC, RC_LOG_RC |
| `dp` | ARRAY, GRID, STRING, INTERVAL | LINEAR, N_LOG_N, N2, RC |
| `graph` | GRAPH | V_PLUS_E, E_LOG_V |
| `정렬` | ARRAY | N_LOG_N, LINEAR |
| `이분탐색` | ARRAY, ANSWER_SEARCH | LOG_N, N_LOG_N, Q_LOG_N |
| `greedy` | ARRAY, INTERVAL, STRING | LINEAR, N_LOG_N |
| `bfs` | GRAPH, GRID | V_PLUS_E, RC, E_LOG_V, RC_LOG_RC |
| `string` | STRING | LINEAR, N_LOG_N, N2 |

지원 언어 (`language` 파라미터, 기본값 `java`):

| 값 | 1초당 유효 연산량 | N2 허용 max_n |
| --- | --- | --- |
| `cpp` | 5,000만 | 10,000 |
| `java` | 2,500만 | 5,000 |
| `python` | 1,000만 | 2,000 |

응답의 주요 필드:

| 필드 | 설명 |
| --- | --- |
| `title` | 문제 제목 |
| `description` | 문제 설명 |
| `input_description` | 입력 형식 |
| `output_description` | 출력 형식 |
| `constraint_text` | 제약 조건 |
| `time_limit` | 시간 제한(ms) |
| `memory_limit` | 메모리 제한(MB) |
| `category` | 최종 문제 카테고리 |
| `test_cases` | 예제/채점용 테스트케이스 목록 |

`test_cases` 필드:

| 필드 | 설명 |
| --- | --- |
| `input_data` | 실제 입력 문자열 |
| `expected_output` | 정답 출력 |
| `is_sample` | 예제 케이스 여부 (앞 2개가 예제) |
| `case_order` | 케이스 순서 |

#### 내부 생성 파이프라인

```
1. Plan 생성 (최대 3회 재시도)
   LLM이 알고리즘·복잡도·제약 수치를 JSON으로 설계
   → 시간복잡도 budget, 브루트포스 격차, 카테고리 규칙 검증

2. 문제 Spec 생성
   확정된 Plan을 잠금 후 LLM이 제목·설명·입출력 형식 작성

3. 레퍼런스 솔루션 생성 (최대 3회 재시도)
   LLM이 Python 정답 코드 생성
   → TLE/RuntimeError 발생 시 피드백 포함해서 재생성

4. 테스트케이스 생성 + 실행 검증
   LLM이 4개 케이스 생성 (sample 2 + hidden 2)
   → 서버에서 Python 직접 실행해 expected_output 자동 채점
   → anti-naive 케이스 포함 여부, 중복 입력, 크기 기준 검증
```

#### Plan 검증 기준

시간복잡도 budget 공식:

```
budget = max(20,000,000, time_limit_ms × 50,000) × language_multiplier
  cpp    multiplier = 1.0  →  2000ms = 100,000,000 ops
  java   multiplier = 0.5  →  2000ms =  50,000,000 ops
  python multiplier = 0.2  →  2000ms =  20,000,000 ops
```

통과 조건:
- `opt_ops ≤ budget` : 최적해가 시간 안에 통과
- `bf_ops > budget × 1.1` : 브루트포스는 통과 불가
- `bf_ops ≥ opt_ops × 8` : 두 코드 간 격차가 충분히 존재

## 유지보수 메모

- 자격증명을 추가할 때는 `docs/<자격증명>/` PDF 경로와 `docs/syllabus/<자격증명>.md` 이름을 맞추면 검색 필터가 가장 단순해집니다.
- `rag_chroma_store.py`는 기본적으로 로컬 해시 임베딩을 사용해 DB를 만듭니다. 이 방식은 외부 임베딩 호출 없이 재현 가능하지만, 검색 품질을 올리려면 OpenAI 임베딩 기반 DB 생성 흐름과 함께 `rag_embeddings.py` 설정도 같이 점검해야 합니다.
- `/questions/generate`에서 `CODING`은 요청 모델에는 포함되어 있지만 현재 처리하지 않습니다. 코딩 문제는 `/algorithm/generate`를 사용합니다.
- `main.py`에는 알고리즘 문제 생성 로직이 많이 모여 있습니다. 카테고리나 검증 규칙을 확장할 때는 `VALID_CATEGORIES`, `CATEGORY_PROFILES`, `EASY_ALLOWED_OPT_TC`, `MAX_N_FOR_N2`, `LANGUAGE_MULTIPLIERS`, 테스트케이스 검증 함수들을 함께 확인해야 합니다.
- 언어별 연산 속도 기준은 `OPERATIONS_PER_MS`(C++ baseline 50,000)와 `LANGUAGE_MULTIPLIERS`로 관리합니다. 채점 서버 환경에 따라 이 값을 조정하면 budget 전체가 일괄 변경됩니다.
- N2 복잡도의 언어별 max_n 한계는 `MAX_N_FOR_N2`에서 관리합니다 (cpp: 10000, java: 5000, python: 2000).
- LLM 응답은 JSON 형태를 기대하므로, 프롬프트를 수정한 뒤에는 최소한 객관식/주관식/알고리즘 문제 생성 요청을 각각 한 번씩 실행해 응답 모델 검증을 확인하는 것이 좋습니다.

## 빠른 점검 명령

```powershell
python .\rag_pdf_parser.py --docs-dir docs --output rag_chunks.jsonl
python .\rag_chroma_store.py --input rag_chunks.jsonl --syllabus-dir docs/syllabus --db-dir chroma_db --reset
uvicorn main:app --reload
```

```powershell
python .\test_validation.py
```
