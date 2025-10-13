from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import httpx

router = APIRouter()

class RiskSummaryRequest(BaseModel):
    comments: List[str] = Field(..., description="店家評論內容（多條）")
    scores: Dict[str, float] = Field(..., description="各項風險分數，分數越高越危險")
    user_weights: Dict[str, float] = Field(..., description="使用者設定的各項權重")

class RiskSummaryResponse(BaseModel):
    summary: str = Field(..., description="風險評估摘要")
    final_risk_score: float = Field(..., description="加權後的總風險分數")

async def call_llm_rag(prompt: str) -> str:
    url = "http://45.32.22.69:11434/api/generate"
    payload = {"model": "gemma3:12b", "prompt": prompt}
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response") or data.get("text") or str(data)

def build_rag_prompt(comments: List[str], scores: Dict[str, float], user_weights: Dict[str, float]) -> str:
    score_str = "，".join([f"{k}:{v}" for k, v in scores.items()])
    weight_str = "，".join([f"{k}:{v}" for k, v in user_weights.items()])
    comments_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(comments)])
    prompt = (
        f"請根據以下多條評論內容與風險分數，並考慮使用者權重設定，生成一段風險評估摘要：\n"
        f"評論內容：\n{comments_str}\n"
        f"風險分數：{score_str}\n"
        f"使用者權重：{weight_str}\n"
        f"請以條列式摘要店家風險，並說明各面向的風險重點。"
    )
    return prompt

@router.post("/risk/summary", response_model=RiskSummaryResponse)
async def risk_summary_api(req: RiskSummaryRequest) -> RiskSummaryResponse:
    # 計算加權總分
    total = 0.0
    weight_sum = 0.0
    for k, v in req.scores.items():
        w = req.user_weights.get(k, 1.0)
        total += v * w
        weight_sum += w
    final_score = total / weight_sum if weight_sum else 0.0
    # 準備 prompt 並呼叫 LLM RAG
    prompt = build_rag_prompt(req.comments, req.scores, req.user_weights)
    try:
        summary = await call_llm_rag(prompt)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM API 錯誤: {e}")
    return RiskSummaryResponse(summary=summary, final_risk_score=final_score)
