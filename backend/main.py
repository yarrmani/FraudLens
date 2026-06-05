import os
import sys
import json
import joblib
import numpy as np
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
from dotenv import load_dotenv

# Fix import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import FeatureExtractor

# Load env
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, '.env'))

# ── Global state ──────────────────────────────────────────
model = None
extractor = None
active_websockets: list[WebSocket] = []


# ── Startup / Shutdown ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, extractor
    print("[FraudLens] Loading model and connecting to DB...")
    model = joblib.load(
        os.path.join(BASE_DIR, 'model', 'model_artifacts', 'fraud_model.pkl')
    )
    extractor = FeatureExtractor()
    print("[FraudLens] Ready.")
    yield
    if extractor:
        extractor.close()
    print("[FraudLens] Shutdown complete.")


# ── App ───────────────────────────────────────────────────
app = FastAPI(
    title="FraudLens API",
    description="Real-time transaction fraud detection",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── DB helper ─────────────────────────────────────────────
def get_db():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", 5433)),
        database=os.getenv("POSTGRES_DB", "frauddb"),
        user=os.getenv("POSTGRES_USER", "frauduser"),
        password=os.getenv("POSTGRES_PASSWORD", "fraudpass")
    )


# ── Pydantic Models (request/response shapes) ─────────────
class TransactionRequest(BaseModel):
    user_id: int
    amount: float
    timestamp: Optional[str] = None
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0


class ReviewRequest(BaseModel):
    transaction_id: str
    officer_decision: str  # 'confirmed_fraud' or 'false_positive'
    notes: Optional[str] = ""


# ── WebSocket broadcast helper ────────────────────────────
async def broadcast(data: dict):
    """Push a message to all connected dashboard clients."""
    disconnected = []
    for ws in active_websockets:
        try:
            await ws.send_json(data)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        try:
            active_websockets.remove(ws)
        except ValueError:
            pass


# ── Routes ────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "FraudLens API is running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "db_connected": extractor is not None
    }


@app.post("/transaction")
async def score_transaction(payload: TransactionRequest):
    """
    Receives a transaction, scores it with the ML model,
    saves result to PostgreSQL, broadcasts to dashboard.
    """
    # Set timestamp to now if not provided
    timestamp = payload.timestamp or datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Build transaction dict for feature extractor
    transaction = payload.dict()
    transaction['timestamp'] = timestamp

    # Extract features
    feature_array = extractor.extract(transaction)

    # Score with model
    fraud_score = float(model.predict_proba(feature_array)[0][1])
    is_flagged = fraud_score >= 0.5

    # Get top 3 features driving the prediction
    feature_importances = model.feature_importances_
    feature_names = extractor.feature_columns
    top_indices = np.argsort(feature_importances)[-3:][::-1]
    top_features = {
        feature_names[i]: round(float(feature_importances[i]), 4)
        for i in top_indices
    }

    # Save to PostgreSQL
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO transactions
            (user_id, amount, hour, amount_scaled, fraud_score, is_flagged, top_features)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING transaction_id, created_at
    """, (
        payload.user_id,
        payload.amount,
        datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').hour,
        float(feature_array[0][-1]),
        round(fraud_score, 4),
        is_flagged,
        json.dumps(top_features)
    ))

    row = cursor.fetchone()
    transaction_id = str(row[0])
    created_at = row[1].isoformat()
    conn.commit()
    cursor.close()
    conn.close()

    # Build response
    result = {
        "transaction_id": transaction_id,
        "user_id": payload.user_id,
        "amount": payload.amount,
        "fraud_score": round(fraud_score, 4),
        "is_flagged": is_flagged,
        "confidence_pct": round(fraud_score * 100, 1),
        "top_features": top_features,
        "timestamp": created_at
    }

    # Push to all connected dashboard clients
    await broadcast(result)

    return result


@app.get("/transactions")
def get_transactions(limit: int = 50, flagged_only: bool = False):
    """Returns recent transactions for the dashboard table."""
    conn = get_db()
    cursor = conn.cursor()

    if flagged_only:
        cursor.execute("""
            SELECT transaction_id, user_id, amount, fraud_score,
                   is_flagged, top_features, created_at
            FROM transactions
            WHERE is_flagged = TRUE
            ORDER BY created_at DESC
            LIMIT %s
        """, (limit,))
    else:
        cursor.execute("""
            SELECT transaction_id, user_id, amount, fraud_score,
                   is_flagged, top_features, created_at
            FROM transactions
            ORDER BY created_at DESC
            LIMIT %s
        """, (limit,))

    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    return [
        {
            "transaction_id": str(r[0]),
            "user_id": r[1],
            "amount": float(r[2]),
            "fraud_score": float(r[3]) if r[3] else 0.0,
            "is_flagged": r[4],
            "top_features": r[5],
            "timestamp": r[6].isoformat()
        }
        for r in rows
    ]


@app.post("/review")
def submit_review(payload: ReviewRequest):
    """
    Compliance officer marks a transaction as confirmed fraud
    or false positive. Stored for Airflow retraining.
    """
    if payload.officer_decision not in ['confirmed_fraud', 'false_positive']:
        raise HTTPException(
            status_code=400,
            detail="officer_decision must be 'confirmed_fraud' or 'false_positive'"
        )

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO fraud_reviews (transaction_id, officer_decision, notes)
        VALUES (%s, %s, %s)
        RETURNING review_id
    """, (payload.transaction_id, payload.officer_decision, payload.notes))

    review_id = cursor.fetchone()[0]
    conn.commit()
    cursor.close()
    conn.close()

    return {
        "review_id": review_id,
        "transaction_id": payload.transaction_id,
        "decision": payload.officer_decision,
        "status": "saved"
    }


@app.get("/stats")
def get_stats():
    """Summary stats for the dashboard header."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM transactions")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM transactions WHERE is_flagged = TRUE")
    flagged = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM fraud_reviews")
    reviewed = cursor.fetchone()[0]

    cursor.close()
    conn.close()

    return {
        "total_transactions": total,
        "flagged": flagged,
        "reviewed": reviewed,
        "flag_rate_pct": round((flagged / total * 100), 2) if total > 0 else 0.0
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Dashboard connects here to receive live fraud alerts."""
    await websocket.accept()
    active_websockets.append(websocket)
    print(f"[WS] Client connected. Total: {len(active_websockets)}")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_websockets.remove(websocket)
        print(f"[WS] Client disconnected. Total: {len(active_websockets)}")
