from fastapi import FastAPI, HTTPException
from app.risk_summary_api import router as risk_summary_router
from pydantic import BaseModel, Field

from app.services.nlp import orchestrator


# Instantiate the FastAPI application with metadata for documentation.

app = FastAPI(
    title="Just Beat It API",
    description="FastAPI backend for the Just Beat It project.",
    version="0.1.0",
)
app.include_router(risk_summary_router)


@app.get("/")
async def read_root() -> dict[str, str]:
    """Return a friendly message to confirm the service is reachable."""
    return {"message": "Hello from Just Beat It API"}


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Simple health endpoint for uptime checks."""
    return {"status": "ok"}


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="使用者輸入的自然語句")


class EntityResult(BaseModel):
    text: str = Field(..., description="抽出的主要實體")
    label: str = Field(..., description="實體分類（店家 / 公司）")
    confidence: float = Field(..., ge=0.0, le=1.0)
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)


class AnalyzeResponse(BaseModel):
    action: str = Field(..., description="系統推論出的動作（查詢餐廳 / 查詢公司）")
    intent: str = Field(..., description="更細的意圖標籤")
    intent_confidence: float = Field(..., ge=0.0, le=1.0)
    entities: list[EntityResult] = Field(default_factory=list)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_intent(request: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze user text and return the action intent plus extracted entities."""
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="請提供有效的句子。")

    action, intent_pred, entities = orchestrator.analyze(text)

    return AnalyzeResponse(
        action=action,
        intent=intent_pred.label,
        intent_confidence=intent_pred.confidence,
        entities=[
            EntityResult(
                text=entity.text,
                label=entity.label,
                confidence=entity.confidence,
                start=entity.start,
                end=entity.end,
            )
            for entity in entities
        ],
    )
