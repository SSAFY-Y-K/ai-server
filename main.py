from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from chat_with_rag import generate_questions_for_certification


app = FastAPI(title="Certification RAG Question Generator")


class GenerateQuestionsRequest(BaseModel):
    certification_name: str = Field(..., min_length=1, examples=["정보처리기사"])
    question_count: int = Field(default=20, ge=1, le=50)
    top_k: int = Field(default=12, ge=1, le=50)


class SourceResponse(BaseModel):
    source: str
    page: int | str
    category: str


class GenerateQuestionsResponse(BaseModel):
    certification_name: str
    question_count: int
    content: str
    sources: list[SourceResponse]


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "Certification RAG Question Generator"}


@app.post("/questions/generate", response_model=GenerateQuestionsResponse)
async def generate_questions(request: GenerateQuestionsRequest) -> GenerateQuestionsResponse:
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
        sources=[
            SourceResponse(
                source=source.source,
                page=source.page,
                category=source.category,
            )
            for source in result.sources
        ],
    )
