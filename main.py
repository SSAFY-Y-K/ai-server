"""FastAPI 엔드포인트를 정의해 자격증별 RAG 문제 생성을 제공한다."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from chat_with_rag import generate_questions_for_certification


app = FastAPI(title="Certification RAG Question Generator")


class GenerateQuestionsRequest(BaseModel):
    """문제 생성 API가 받는 요청 본문 구조."""

    certification_name: str = Field(..., min_length=1, examples=["정보처리기사"])
    question_count: int = Field(default=20, ge=1, le=50)
    top_k: int = Field(default=12, ge=1, le=50)


class SourceResponse(BaseModel):
    """생성에 참고한 원본 PDF 출처 정보."""

    source: str
    page: int | str
    category: str


class GenerateQuestionsResponse(BaseModel):
    """문제 생성 API가 반환하는 응답 본문 구조."""

    certification_name: str
    question_count: int
    content: str
    review_feedback: str
    sources: list[SourceResponse]


@app.get("/")
def read_root() -> dict[str, str]:
    """서버 상태 확인용 기본 응답을 반환한다."""

    return {"message": "Certification RAG Question Generator"}


@app.post("/questions/generate", response_model=GenerateQuestionsResponse)
async def generate_questions(request: GenerateQuestionsRequest) -> GenerateQuestionsResponse:
    """자격증 이름을 받아 RAG 기반 객관식 문제 세트를 생성한다."""

    try:
        result = await generate_questions_for_certification(
            request.certification_name,
            question_count=request.question_count,
            top_k=request.top_k,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    return GenerateQuestionsResponse(
        certification_name=result.certification_name,
        question_count=result.question_count,
        content=result.content,
        review_feedback=result.review_feedback,
        sources=[
            SourceResponse(
                source=source.source,
                page=source.page,
                category=source.category,
            )
            for source in result.sources
        ],
    )
