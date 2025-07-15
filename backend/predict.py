"""
/api/predict/{ticker}

Enhanced prediction API with multi-class classification and confidence scores.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from pathlib import Path

# Import the enhanced indicators
from utils.indicators import add_features, add_index_features

# ------------------------------------------------------------------
# Load models and metadata
# ------------------------------------------------------------------
MODEL_DIR = Path("model")

try:
    # Load classifier (ensemble or single model)
    classifier = joblib.load(MODEL_DIR / "xgboost_stock_predictor.pkl")
    
    # Load regressor
    regressor = joblib.load(MODEL_DIR / "xgb_regressor.pkl")
    
    # Load scaler
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    
    # Load feature columns
    feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")
    
except FileNotFoundError as e:
    raise RuntimeError(
        f"Model files not found in '{MODEL_DIR}'. "
        "Did you run the training script and copy models to backend/model/?"
    ) from e

# Load index data for features (simplified - in production, cache these)
INDEX_TICKERS = {
    "NIFTY50": "^NSEI",
    "NIFTYBANK": "^NSEBANK",
    # Add more as needed
}

# ------------------------------------------------------------------
# FastAPI router
# ------------------------------------------------------------------
router = APIRouter(prefix="/api")

class PredictionResponse(BaseModel):
    ticker: str
    prediction: str              # "BEARISH" / "NEUTRAL" / "BULLISH"
    confidence: float            # Probability of predicted class
    probabilities: dict          # All class probabilities
    expected_return: float       # From regression model
    horizon_days: int
    recommendation: str          # Investment recommendation
    risk_level: str             # "LOW" / "MEDIUM" / "HIGH"
    technical_signals: dict      # Key technical indicator values


class MarketOverview(BaseModel):
    market_trend: str
    volatility_regime: str
    sector_performance: dict


@router.get("/predict/{ticker}", response_model=PredictionResponse)
async def predict_ticker(
    ticker: str,
    include_technicals: bool = Query(True, description="Include technical indicator values"),
) -> PredictionResponse:
    """
    Predicts price movement for the given ticker over the next 30 days.
    
    Returns:
    - Multi-class prediction (Bearish/Neutral/Bullish)
    - Confidence score
    - Expected return percentage
    - Investment recommendation
    """
    ticker = ticker.strip().upper()
    yf_symbol = f"{ticker}.NS"
    
    try:
        # Fetch recent data (need enough for 200-day SMA)
        stock = yf.Ticker(yf_symbol)
        hist = stock.history(period="1y", interval="1d")
        
        if hist.empty or len(hist) < 200:
            raise HTTPException(
                status_code=404,
                detail=f"Insufficient data for {ticker}. Need at least 200 days of history."
            )
        
        # Convert to expected format
        hist = hist.reset_index()
        hist.columns = [col.replace(' ', '_') for col in hist.columns]
        
        # Add features
        df = add_features(hist)
        
        # Add dummy index features (in production, fetch real index data)
        for idx_name in ["NIFTY50", "NIFTYBANK", "NIFTYMIDCAP150", "NIFTYSMALLCAP250", "NIFTYIT", "NIFTYFMCG"]:
            df[f"{idx_name}_1d_return"] = 0.001  # Dummy value
            df[f"{idx_name}_5d_return"] = 0.005  # Dummy value
        
        if "bank_it_divergence" not in df.columns:
            df["bank_it_divergence"] = 0
        
        # Get latest row
        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="Not enough data to calculate indicators"
            )
        
        latest_row = df.iloc[-1]
        
        # Ensure all features are present
        X_live = []
        for col in feature_columns:
            if col in latest_row:
                X_live.append(latest_row[col])
            else:
                X_live.append(0)  # Default value for missing features
        
        X_live = np.array(X_live).reshape(1, -1)
        
        # Scale features
        X_scaled = scaler.transform(X_live)
        
        # Get predictions
        class_pred = classifier.predict(X_scaled)[0]
        class_proba = classifier.predict_proba(X_scaled)[0]
        return_pred = regressor.predict(X_scaled)[0]
        
        # Map predictions
        class_names = ["BEARISH", "NEUTRAL", "BULLISH"]
        prediction_label = class_names[class_pred]
        confidence = float(class_proba[class_pred])
        
        # Create probability dictionary
        probabilities = {
            class_names[i]: float(class_proba[i])
            for i in range(len(class_names))
        }
        
        # Generate recommendation
        if prediction_label == "BULLISH" and confidence > 0.6:
            recommendation = "STRONG BUY"
            risk_level = "MEDIUM"
        elif prediction_label == "BULLISH":
            recommendation = "BUY"
            risk_level = "MEDIUM"
        elif prediction_label == "BEARISH" and confidence > 0.6:
            recommendation = "STRONG SELL"
            risk_level = "HIGH"
        elif prediction_label == "BEARISH":
            recommendation = "SELL"
            risk_level = "HIGH"
        else:
            recommendation = "HOLD"
            risk_level = "LOW"
        
        # Adjust risk based on volatility
        current_volatility = latest_row.get('volatility_20', 0)
        if current_volatility > 0.4:  # High volatility (40% annualized)
            risk_level = "HIGH"
        elif current_volatility > 0.25:
            risk_level = "MEDIUM"
        
        # Technical signals
        technical_signals = {}
        if include_technicals:
            technical_signals = {
                "rsi": round(float(latest_row.get('rsi', 50)), 2),
                "price_to_sma200": round(float(latest_row.get('price_to_sma200_ratio', 1)), 3),
                "golden_cross": bool(latest_row.get('golden_cross', 0)),
                "macd_bullish": bool(latest_row.get('macd_bullish', 0)),
                "volatility": round(float(current_volatility * 100), 2),  # As percentage
                "trend_strength": round(float(latest_row.get('trend_strength_20d', 0) * 100), 2),
            }
        
        return PredictionResponse(
            ticker=ticker,
            prediction=prediction_label,
            confidence=round(confidence, 3),
            probabilities=probabilities,
            expected_return=round(float(return_pred * 100), 2),  # As percentage
            horizon_days=30,
            recommendation=recommendation,
            risk_level=risk_level,
            technical_signals=technical_signals,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing {ticker}: {str(e)}"
        )


@router.get("/market/overview", response_model=MarketOverview)
async def get_market_overview() -> MarketOverview:
    """
    Get overall market conditions and sector performance.
    """
    try:
        # Fetch major indices
        nifty50 = yf.Ticker("^NSEI")
        nifty_hist = nifty50.history(period="1mo")
        
        if nifty_hist.empty:
            raise HTTPException(status_code=503, detail="Market data unavailable")
        
        # Calculate market metrics
        nifty_return = (nifty_hist['Close'][-1] / nifty_hist['Close'][0] - 1) * 100
        nifty_volatility = nifty_hist['Close'].pct_change().std() * np.sqrt(252) * 100
        
        # Determine market trend
        if nifty_return > 3:
            market_trend = "BULLISH"
        elif nifty_return < -3:
            market_trend = "BEARISH"
        else:
            market_trend = "NEUTRAL"
        
        # Volatility regime
        if nifty_volatility > 25:
            volatility_regime = "HIGH"
        elif nifty_volatility > 15:
            volatility_regime = "MEDIUM"
        else:
            volatility_regime = "LOW"
        
        # Sector performance (simplified)
        sectors = {
            "Banking": "^NSEBANK",
            "IT": "^CNXIT",
            "Pharma": "^CNXPHARMA",
            "Auto": "^CNXAUTO",
        }
        
        sector_performance = {}
        for sector, symbol in sectors.items():
            try:
                sector_data = yf.Ticker(symbol).history(period="1mo")
                if not sector_data.empty:
                    sector_return = (sector_data['Close'][-1] / sector_data['Close'][0] - 1) * 100
                    sector_performance[sector] = round(sector_return, 2)
            except:
                sector_performance[sector] = 0.0
        
        return MarketOverview(
            market_trend=market_trend,
            volatility_regime=volatility_regime,
            sector_performance=sector_performance
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching market overview: {str(e)}"
        )


@router.get("/predict/batch", response_model=list[PredictionResponse])
async def predict_batch(
    tickers: str = Query(..., description="Comma-separated list of tickers"),
) -> list[PredictionResponse]:
    """
    Get predictions for multiple tickers at once.
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    
    if len(ticker_list) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 tickers allowed per request"
        )
    
    predictions = []
    for ticker in ticker_list:
        try:
            pred = await predict_ticker(ticker, include_technicals=False)
            predictions.append(pred)
        except HTTPException:
            # Skip failed tickers
            continue
    
    return predictions