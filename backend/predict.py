"""
/api/predict/{ticker}

Loads the trained XGBoost model, rebuilds the indicators for the latest
price window and returns movement prediction + confidence.
"""
from __future__ import annotations

import joblib
import yfinance as yf
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from utils.indicators import add_features, FEATURE_COLUMNS

# ------------------------------------------------------------------
# Load model + metadata once at startup
# ------------------------------------------------------------------
_MODEL_PATH = "model/xgboost_stock_predictor.pkl"
_META_PATH  = "model/model_meta.json"

try:
    model = joblib.load(_MODEL_PATH)
except FileNotFoundError as e:
    raise RuntimeError(
        f"Model file not found at '{_MODEL_PATH}'. "
        "Did you export the model and copy it into backend/model/?"
    ) from e


# ------------------------------------------------------------------
# FastAPI router
# ------------------------------------------------------------------
router = APIRouter(prefix="/api")


class PredictionResponse(BaseModel):
    ticker: str
    prediction: str            # "UP" / "DOWN"
    probability: float         # model confidence for predicted class
    next_price: float | None   # optional absolute target
    model_type: str            # e.g. "XGBoost"
    features_used: list[str]


@router.get("/predict/{ticker}", response_model=PredictionResponse)
def predict_ticker(
    ticker: str,
    horizon_days: int = Query(
        1, ge=1, le=5, description="Days ahead to predict movement for"
    ),
) -> PredictionResponse:
    """
    Predicts whether the closing price will be UP or DOWN after
    `horizon_days` days (default 1).
    """
    yf_symbol = f"{ticker.strip().upper()}.NS"

    # Fetch a small recent window (30 trading days) → fast
    hist = yf.download(yf_symbol, period="60d", interval="1d", progress=False)
    if hist.empty or "Close" not in hist.columns:
        raise HTTPException(status_code=404, detail="Ticker not found or no data.")

    # Compute indicators
    hist = add_features(hist)

    if hist.empty:
        raise HTTPException(status_code=400, detail="Not enough data for indicators.")

    latest_row = hist.iloc[-1]             # features known up to *today*
    X_live     = latest_row[FEATURE_COLUMNS].values.reshape(1, -1)

    # Probability of class 1 ("UP") from XGBoost
    proba_up = model.predict_proba(X_live)[0][1]
    prediction_label = "UP" if proba_up >= 0.5 else "DOWN"
    confidence = proba_up if prediction_label == "UP" else 1 - proba_up

    # Optional absolute price target (naïve: last close ± pct * close)
    last_close = latest_row["Close"]
    pct_move   = 0.02                        # 2 % stub – replace with regression model
    target_price = (
        last_close * (1 + pct_move) if prediction_label == "UP"
        else last_close * (1 - pct_move)
    )

    return PredictionResponse(
        ticker=ticker.upper(),
        prediction=prediction_label,
        probability=round(float(confidence), 4),
        next_price=round(float(target_price), 2),
        model_type="XGBoost",
        features_used=FEATURE_COLUMNS,
    )
