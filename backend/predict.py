"""
/api/predict/{ticker}

Enhanced prediction API with multi-class classification and confidence scores.
Now with real index data fetching and improved features.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
import warnings

# Import the enhanced indicators
from utils.indicators import add_features, add_index_features

warnings.filterwarnings('ignore')

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
    
    print(f"✅ Models loaded successfully. Using {len(feature_columns)} features.")
    
except FileNotFoundError as e:
    raise RuntimeError(
        f"Model files not found in '{MODEL_DIR}'. "
        "Did you run the training script and copy models to backend/model/?"
    ) from e

# Enhanced index data mapping with real Yahoo Finance symbols
INDEX_TICKERS = {
    "NIFTY50": "^NSEI",
    "NIFTYBANK": "^NSEBANK", 
    "NIFTYMIDCAP150": "^NSEMDCP50",  # Closest proxy
    "NIFTYSMALLCAP250": "^NSESMLCAP",  # Closest proxy
    "NIFTYIT": "^CNXIT",
    "NIFTYFMCG": "^CNXFMCG",
}

# Cache for index data to avoid repeated API calls
INDEX_DATA_CACHE = {}
CACHE_DURATION = 3600  # 1 hour in seconds

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
    market_context: dict         # New: Market context information


class MarketOverview(BaseModel):
    market_trend: str
    volatility_regime: str
    sector_performance: dict
    market_breadth: dict         # New: Market breadth indicators
    fear_greed_index: float      # New: Fear & greed indicator


class BatchPredictionItem(BaseModel):
    ticker: str
    prediction: str
    confidence: float
    expected_return: float
    recommendation: str
    risk_level: str


def fetch_index_data(period: str = "1y") -> dict:
    """Fetch real index data from Yahoo Finance with caching"""
    import time
    
    current_time = time.time()
    
    # Check cache
    if INDEX_DATA_CACHE.get('timestamp', 0) + CACHE_DURATION > current_time:
        return INDEX_DATA_CACHE.get('data', {})
    
    index_data = {}
    
    for name, symbol in INDEX_TICKERS.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if not hist.empty:
                # Calculate returns
                hist['1d_return'] = hist['Close'].pct_change(1)
                hist['5d_return'] = hist['Close'].pct_change(5)
                
                # Store data with proper column names
                index_data[name] = hist[['Close', '1d_return', '5d_return']].copy()
                index_data[name].columns = ['Close', f'{name}_1d_return', f'{name}_5d_return']
                
                print(f"✅ Fetched {name} data: {len(hist)} records")
            else:
                print(f"⚠️ No data for {name} ({symbol})")
                
        except Exception as e:
            print(f"❌ Error fetching {name} ({symbol}): {str(e)}")
            continue
    
    # Update cache
    INDEX_DATA_CACHE['data'] = index_data
    INDEX_DATA_CACHE['timestamp'] = current_time
    
    return index_data

def add_real_index_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add real index features from Yahoo Finance data"""
    df = df.copy()
    
    # Fetch index data
    index_data = fetch_index_data()
    
    if not index_data:
        print("⚠️ No index data available, using dummy values")
        # Fallback to dummy values
        for idx_name in INDEX_TICKERS.keys():
            df[f"{idx_name}_1d_return"] = 0.001
            df[f"{idx_name}_5d_return"] = 0.005
        return df
    
    # Merge index data with main dataframe
    for name, idx_df in index_data.items():
        try:
            # Align dates
            if 'Date' in df.columns:
                df_dates = pd.to_datetime(df['Date'])
                idx_dates = idx_df.index
                
                # Find matching dates
                merged = pd.merge_asof(
                    df.sort_values('Date'),
                    idx_df.reset_index().rename(columns={'Date': 'idx_date'}),
                    left_on='Date',
                    right_on='idx_date',
                    direction='backward'
                )
                
                # Update main dataframe
                df[f"{name}_1d_return"] = merged[f"{name}_1d_return"].fillna(0.001)
                df[f"{name}_5d_return"] = merged[f"{name}_5d_return"].fillna(0.005)
                
            else:
                # Use latest values if no date column
                latest_idx = idx_df.iloc[-1]
                df[f"{name}_1d_return"] = latest_idx[f"{name}_1d_return"] if not pd.isna(latest_idx[f"{name}_1d_return"]) else 0.001
                df[f"{name}_5d_return"] = latest_idx[f"{name}_5d_return"] if not pd.isna(latest_idx[f"{name}_5d_return"]) else 0.005
                
        except Exception as e:
            print(f"⚠️ Error adding {name} features: {str(e)}")
            df[f"{name}_1d_return"] = 0.001
            df[f"{name}_5d_return"] = 0.005
    
    # Add interaction features
    if all(col in df.columns for col in ['NIFTYBANK_5d_return', 'NIFTYIT_5d_return']):
        df['bank_it_divergence'] = df['NIFTYBANK_5d_return'] - df['NIFTYIT_5d_return']
    else:
        df['bank_it_divergence'] = 0
    
    return df

def get_market_context(latest_row: pd.Series, index_data: dict) -> dict:
    """Extract market context information"""
    context = {
        'market_breadth': 'neutral',
        'sector_leadership': 'mixed',
        'volatility_regime': 'normal',
        'momentum_trend': 'neutral'
    }
    
    try:
        # Market breadth
        if 'market_breadth' in latest_row:
            breadth_val = latest_row['market_breadth']
            if breadth_val > 0.01:
                context['market_breadth'] = 'positive'
            elif breadth_val < -0.01:
                context['market_breadth'] = 'negative'
        
        # Sector leadership
        if 'sector_momentum' in latest_row:
            sector_val = latest_row['sector_momentum']
            if sector_val > 0.005:
                context['sector_leadership'] = 'IT leading'
            elif sector_val < -0.005:
                context['sector_leadership'] = 'Banking leading'
        
        # Volatility regime
        if 'volatility_regime' in latest_row:
            vol_regime = latest_row['volatility_regime']
            if vol_regime == 2:
                context['volatility_regime'] = 'high'
            elif vol_regime == 0:
                context['volatility_regime'] = 'low'
        
        # Momentum trend
        if 'momentum_consistency' in latest_row:
            momentum_val = latest_row['momentum_consistency']
            if momentum_val > 0.6:
                context['momentum_trend'] = 'bullish'
            elif momentum_val < 0.4:
                context['momentum_trend'] = 'bearish'
                
    except Exception as e:
        print(f"⚠️ Error extracting market context: {str(e)}")
    
    return context

@router.get("/predict/{ticker}", response_model=PredictionResponse)
async def predict_ticker(
    ticker: str,
    include_technicals: bool = Query(True, description="Include technical indicator values"),
    use_real_index_data: bool = Query(True, description="Use real index data instead of dummy values"),
) -> PredictionResponse:
    """
    Predicts price movement for the given ticker over the next 30 days.
    
    Enhanced with real index data and improved features.
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
        
        # Add basic features
        df = add_features(hist)
        
        # Add index features
        if use_real_index_data:
            df = add_real_index_features(df)
        else:
            # Use dummy values for backward compatibility
            for idx_name in INDEX_TICKERS.keys():
                df[f"{idx_name}_1d_return"] = 0.001
                df[f"{idx_name}_5d_return"] = 0.005
            
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
        missing_features = []
        
        for col in feature_columns:
            if col in latest_row and not pd.isna(latest_row[col]):
                X_live.append(float(latest_row[col]))
            else:
                X_live.append(0.0)  # Default value for missing features
                missing_features.append(col)
        
        if missing_features:
            print(f"⚠️ Missing features for {ticker}: {missing_features[:5]}...")  # Show first 5
        
        X_live = np.array(X_live).reshape(1, -1)
        
        # Handle any remaining inf or nan values
        X_live = np.nan_to_num(X_live, nan=0.0, posinf=1.0, neginf=-1.0)
        
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
        
        # Enhanced recommendation logic
        volatility_factor = latest_row.get('volatility_20', 0.2)
        market_context = get_market_context(latest_row, INDEX_DATA_CACHE.get('data', {}))
        
        # Base recommendation
        if prediction_label == "BULLISH" and confidence > 0.7:
            recommendation = "STRONG BUY"
            risk_level = "MEDIUM"
        elif prediction_label == "BULLISH" and confidence > 0.5:
            recommendation = "BUY"
            risk_level = "MEDIUM"
        elif prediction_label == "BEARISH" and confidence > 0.7:
            recommendation = "STRONG SELL"
            risk_level = "HIGH"
        elif prediction_label == "BEARISH" and confidence > 0.5:
            recommendation = "SELL"
            risk_level = "HIGH"
        else:
            recommendation = "HOLD"
            risk_level = "LOW"
        
        # Adjust based on market context
        if market_context['volatility_regime'] == 'high':
            risk_level = "HIGH"
            if recommendation in ["BUY", "STRONG BUY"]:
                recommendation = "HOLD"  # Be more conservative in high volatility
        
        # Adjust risk based on volatility
        if volatility_factor > 0.4:  # High volatility (40% annualized)
            risk_level = "HIGH"
        elif volatility_factor > 0.25:
            if risk_level == "LOW":
                risk_level = "MEDIUM"
        
        # Technical signals
        technical_signals = {}
        if include_technicals:
            technical_signals = {
                "rsi": round(float(latest_row.get('rsi', 50)), 2),
                "price_to_sma200": round(float(latest_row.get('price_to_sma200_ratio', 1)), 3),
                "golden_cross": bool(latest_row.get('golden_cross', 0)),
                "macd_bullish": bool(latest_row.get('macd_bullish', 0)),
                "volatility": round(float(volatility_factor * 100), 2),  # As percentage
                "trend_strength": round(float(latest_row.get('trend_strength_20d', 0) * 100), 2),
                "adx": round(float(latest_row.get('adx', 25)), 2),
                "volume_spike": bool(latest_row.get('volume_spike', 0)),
                "price_position": round(float(latest_row.get('price_position', 0.5)), 3),
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
            market_context=market_context,
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
    Get enhanced market conditions and sector performance.
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
        
        # Enhanced sector performance
        sectors = {
            "Banking": "^NSEBANK",
            "IT": "^CNXIT",
            "Pharma": "^CNXPHARMA",
            "Auto": "^CNXAUTO",
            "FMCG": "^CNXFMCG",
            "Metal": "^CNXMETAL",
            "Energy": "^CNXENERGY",
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
        
        # Market breadth indicators
        index_data = fetch_index_data(period="1mo")
        market_breadth = {
            "advancing_sectors": 0,
            "declining_sectors": 0,
            "breadth_ratio": 0.5
        }
        
        if index_data:
            advancing = sum(1 for name, data in index_data.items() 
                          if not data.empty and data.iloc[-1][f'{name}_1d_return'] > 0)
            declining = len(index_data) - advancing
            
            market_breadth = {
                "advancing_sectors": advancing,
                "declining_sectors": declining,
                "breadth_ratio": advancing / len(index_data) if index_data else 0.5
            }
        
        # Fear & Greed Index (simplified)
        fear_greed_components = {
            "volatility": min(100, max(0, 100 - (nifty_volatility / 30 * 100))),
            "momentum": min(100, max(0, 50 + nifty_return * 2)),
            "market_breadth": market_breadth["breadth_ratio"] * 100,
        }
        
        fear_greed_index = np.mean(list(fear_greed_components.values()))
        
        return MarketOverview(
            market_trend=market_trend,
            volatility_regime=volatility_regime,
            sector_performance=sector_performance,
            market_breadth=market_breadth,
            fear_greed_index=round(fear_greed_index, 2),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching market overview: {str(e)}"
        )


@router.get("/predict/batch", response_model=list[BatchPredictionItem])
async def predict_batch(
    tickers: str = Query(..., description="Comma-separated list of tickers"),
    use_real_index_data: bool = Query(True, description="Use real index data"),
) -> list[BatchPredictionItem]:
    """
    Get predictions for multiple tickers at once.
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    
    if len(ticker_list) > 20:
        raise HTTPException(
            status_code=400,
            detail="Maximum 20 tickers allowed per request"
        )
    
    predictions = []
    
    # Pre-fetch index data once for all tickers
    if use_real_index_data:
        fetch_index_data()
    
    for ticker in ticker_list:
        try:
            pred = await predict_ticker(
                ticker, 
                include_technicals=False, 
                use_real_index_data=use_real_index_data
            )
            predictions.append(BatchPredictionItem(
                ticker=pred.ticker,
                prediction=pred.prediction,
                confidence=pred.confidence,
                expected_return=pred.expected_return,
                recommendation=pred.recommendation,
                risk_level=pred.risk_level,
            ))
        except HTTPException:
            # Skip failed tickers
            continue
        except Exception as e:
            print(f"⚠️ Error predicting {ticker}: {str(e)}")
            continue
    
    return predictions


@router.get("/market/indices", response_model=dict)
async def get_market_indices() -> dict:
    """
    Get current performance of major market indices.
    """
    try:
        indices_performance = {}
        
        for name, symbol in INDEX_TICKERS.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                
                if not hist.empty and len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2]
                    daily_change = ((current_price - prev_price) / prev_price) * 100
                    
                    # Week performance
                    week_start = hist['Close'].iloc[0]
                    week_change = ((current_price - week_start) / week_start) * 100
                    
                    indices_performance[name] = {
                        "current_price": round(current_price, 2),
                        "daily_change": round(daily_change, 2),
                        "weekly_change": round(week_change, 2),
                        "symbol": symbol
                    }
                    
            except Exception as e:
                print(f"⚠️ Error fetching {name}: {str(e)}")
                continue
        
        return {
            "indices": indices_performance,
            "timestamp": pd.Timestamp.now().isoformat(),
            "market_status": "open" if 9 <= pd.Timestamp.now().hour <= 15 else "closed"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching indices data: {str(e)}"
        )


@router.get("/model/info", response_model=dict)
async def get_model_info() -> dict:
    """
    Get information about the current model.
    """
    try:
        # Load metadata
        metadata_path = MODEL_DIR / "model_meta.json"
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {"error": "Metadata not found"}
        
        # Add runtime information
        runtime_info = {
            "features_count": len(feature_columns),
            "model_type": str(type(classifier).__name__),
            "regressor_type": str(type(regressor).__name__),
            "scaler_type": str(type(scaler).__name__),
            "cache_status": {
                "index_data_cached": bool(INDEX_DATA_CACHE.get('data')),
                "cache_timestamp": INDEX_DATA_CACHE.get('timestamp'),
            }
        }
        
        return {
            "metadata": metadata,
            "runtime_info": runtime_info,
            "feature_sample": feature_columns[:10],  # First 10 features
            "total_features": len(feature_columns)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model info: {str(e)}"
        )


# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": pd.Timestamp.now().isoformat(),
        "model_loaded": classifier is not None,
        "features_loaded": len(feature_columns) > 0
    }