"""FastAPI 엔드포인트를 정의해 자격증별 RAG 문제 생성을 제공한다."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from chat_with_rag import generate_questions_for_certification
from rag_config import DEFAULT_TOP_K
from rag_models import ProblemSetPayload


app = FastAPI(title="Certification RAG Question Generator")


class GenerateQuestionsRequest(BaseModel):
    """문제 생성 API가 받는 요청 본문 구조."""

    certification: str = Field(..., min_length=1, examples=["정보처리기사"])
    questionCount: int = Field(default=20, ge=1, le=50)
    topK: int = Field(default=DEFAULT_TOP_K, ge=1, le=50)


@app.get("/")
def read_root() -> dict[str, str]:
    """서버 상태 확인용 기본 응답을 반환한다."""

    return {"message": "Certification RAG Question Generator"}


@app.post("/questions/generate", response_model=ProblemSetPayload)
async def generate_questions(request: GenerateQuestionsRequest) -> ProblemSetPayload:
    """자격증 이름을 받아 RAG 기반 문제 세트를 생성한다."""

    try:
        result = await generate_questions_for_certification(
            request.certification,
            question_count=request.questionCount,
            top_k=request.topK,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    return result.problem_set
