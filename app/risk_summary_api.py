from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict

import httpx
import psycopg2
from sentence_transformers import SentenceTransformer, util
import torch
import logging
from datetime import datetime
import os

router = APIRouter()

# 設定 log 檔案（/var/log/just_beat_it_py/risk_summary_api.log）
LOG_DIR = "/var/log/just_beat_it_py"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "risk_summary_api.log")
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    encoding="utf-8"
)

# 初始化 embedding 模型（可放 global）
embedding_model = SentenceTransformer("BAAI/bge-m3")

# === DB 連線設定 ===
DB_CONFIG = {
    "dbname": "justbeatit",
    "user": "myuser",
    "password": "justbeatit2486123",
    "host": "202.182.96.90",
    "port": "5432"
}

# === 資料模型 ===
class RiskSummaryRequest(BaseModel):
    entity_id: int = Field(..., description="店家 ID(由 .NET 傳入)")
    scores: Dict[str, float] = Field(..., description="各項風險分數,分數越高越危險")
    user_weights: Dict[str, float] = Field(..., description="使用者設定的各項權重")

class RiskSummaryResponse(BaseModel):
    summary: str = Field(..., description="風險評估摘要")
    final_risk_score: float = Field(..., description="加權後的總風險分數")

# === 工具函式 ===
def get_comments_from_db(entity_id: int, limit: int = 100) -> List[str]:
    """根據店家 ID 從 review_table 撈取評論"""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        SELECT content
        FROM review_table
        WHERE entity_id = %s
        ORDER BY created_at DESC
        LIMIT %s;
    """, (entity_id, limit))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [r[0] for r in rows if r[0]]

def select_top_reviews(comments: List[str], top_k: int = 5) -> List[str]:
    """利用 embedding 選出最具代表性的評論"""
    if not comments:
        return []
    embeddings = embedding_model.encode(comments, convert_to_tensor=True)
    centroid = embeddings.mean(dim=0, keepdim=True)
    scores = util.cos_sim(centroid, embeddings)[0]
    top_k_idx = torch.topk(scores, k=min(top_k, len(comments))).indices
    return [comments[i] for i in top_k_idx]

async def call_llm_rag(prompt: str) -> str:
    """呼叫 LLM(Ollama / TGI)"""
    url = "http://45.32.22.69:11434/api/generate"
    payload = {"model": "gemma3:12b", "prompt": prompt}
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response") or data.get("text") or str(data)

def build_rag_prompt(comments: List[str], scores: Dict[str, float], user_weights: Dict[str, float]) -> str:
    """建立 RAG prompt"""
    score_str = "，".join([f"{k}:{v:.2f}" for k, v in scores.items()])
    weight_str = "，".join([f"{k}:{v:.2f}" for k, v in user_weights.items()])
    comments_str = "\n".join([f"{i+1}. {c}" for i, c in enumerate(comments)])
    return (
        f"請根據以下評論與風險分數,生成店家風險評估摘要:\n"
        f"=== 評論摘要(Top {len(comments)} 條) ===\n{comments_str}\n\n"
        f"=== 風險分數 === {score_str}\n"
        f"=== 使用者權重 === {weight_str}\n"
        f"請以條列式摘要店家風險,說明主要問題面向(例如服務、品質、環境、價格),"
        f"最後給一句整體評價。"
    )

# === 主 API ===
@router.post("/risk/summary", response_model=RiskSummaryResponse)
async def risk_summary_api(req: RiskSummaryRequest) -> RiskSummaryResponse:
    # 1️⃣ 計算加權總分
    total = 0.0
    weight_sum = 0.0
    for k, v in req.scores.items():
        w = req.user_weights.get(k, 1.0)
        total += v * w
        weight_sum += w
    final_score = total / weight_sum if weight_sum else 0.0

    # 2️⃣ 撈取評論
    try:
        comments = get_comments_from_db(req.entity_id)
    except Exception as e:
        logging.error(f"DB ERROR entity_id={req.entity_id} error={e}")
        raise HTTPException(status_code=500, detail=f"資料庫查詢錯誤: {e}")
    
    if not comments:
        logging.info(f"NO COMMENTS entity_id={req.entity_id}")
        raise HTTPException(status_code=404, detail=f"找不到店家 {req.entity_id} 的評論資料")

    # 3️⃣ 取出最具代表性的評論 (Top-K)
    top_reviews = select_top_reviews(comments, top_k=5)

    # 4️⃣ 建立 Prompt
    prompt = build_rag_prompt(top_reviews, req.scores, req.user_weights)

    # 5️⃣ 呼叫 LLM
    try:
        summary = await call_llm_rag(prompt)
    except Exception as e:
        logging.error(f"LLM ERROR entity_id={req.entity_id} error={e}")
        raise HTTPException(status_code=502, detail=f"LLM API 錯誤: {e}")

    # 6️⃣ 記錄 log
    logging.info(
        f"entity_id={req.entity_id} final_score={final_score:.2f} "
        f"scores={req.scores} user_weights={req.user_weights} "
        f"top_reviews={top_reviews} prompt={prompt[:200]}... summary={summary[:200]}..."
    )

    # 7️⃣ 回傳結果
    return RiskSummaryResponse(summary=summary, final_risk_score=final_score)