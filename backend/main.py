# main.py ─────────────────────────────────────────────────────────────
import os
import time
from datetime import datetime
from typing import Dict, List

import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import pandas as pd
from fastapi import Query 

# ────────────────────────────────────────────────────────────────────
# Config / env
# ────────────────────────────────────────────────────────────────────
load_dotenv()

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://portfolio-peace.vercel.app",  # ← your Vercel front-end
    "https://oneglance.vercel.app",
    "https://portfolio-peace.onrender.com"
]

CACHE_TTL = int(os.getenv("CACHE_TTL", 300))  # seconds (default 5 min)
# Cache for historical data:  {(ticker,start,end): {"series": [...], "ts": float}}
hist_cache: Dict[tuple, Dict] = {}
HIST_CACHE_TTL = 60 * 60  # 1-hour cache


POPULAR = {
    "RELIANCE": "Reliance Industries",
    "TCS": "Tata Consultancy Services",
    "INFY": "Infosys",
    "HDFCBANK": "HDFC Bank",
    "ICICIBANK": "ICICI Bank",
    "WIPRO": "Wipro",
    "ITC": "ITC Limited",
    "SBIN": "State Bank of India",
    "BHARTIARTL": "Bharti Airtel",
    "ASIANPAINT": "Asian Paints",
}

# ────────────────────────────────────────────────────────────────────
# Enhanced cache with daily change data
# ────────────────────────────────────────────────────────────────────
cache: Dict[str, Dict] = {}

def get_cached(ticker: str):
    entry = cache.get(ticker)
    if entry and (time.time() - entry["ts"] < CACHE_TTL):
        return entry
    return None

def set_cache(ticker: str, data: dict):
    cache[ticker] = {**data, "ts": time.time()}

# ────────────────────────────────────────────────────────────────────
# yfinance fetch with better daily change calculation
# ────────────────────────────────────────────────────────────────────
def fetch_price_yf(ticker: str) -> dict | None:
    ticker = ticker.strip().upper()
    
    # Check cache first
    cached = get_cached(ticker)
    if cached:
        return {
            "price": cached.get("price"),
            "previous_close": cached.get("previous_close"),
            "daily_change_pct": cached.get("daily_change_pct")
        }

    yf_symbol = f"{ticker}.NS"
    stock = yf.Ticker(yf_symbol)

    try:
        # Get info for current price and previous close
        info = stock.info
        current_price = None
        previous_close = None
        
        # Try multiple fields for current price
        for field in ['currentPrice', 'regularMarketPrice', 'lastPrice', 'price']:
            if field in info and info[field]:
                current_price = float(info[field])
                break
        
        # Try to get previous close
        if 'previousClose' in info and info['previousClose']:
            previous_close = float(info['previousClose'])
        elif 'regularMarketPreviousClose' in info and info['regularMarketPreviousClose']:
            previous_close = float(info['regularMarketPreviousClose'])
        
        # If we still don't have prices, try history
        if not current_price or not previous_close:
            hist = stock.history(period="5d")
            if not hist.empty and len(hist) >= 2:
                current_price = float(hist["Close"].iloc[-1])
                previous_close = float(hist["Close"].iloc[-2])
        
        if current_price and previous_close:
            daily_change_pct = ((current_price - previous_close) / previous_close) * 100
            
            # Cache the data
            set_cache(ticker, {
                "price": round(current_price, 2),
                "previous_close": round(previous_close, 2),
                "daily_change_pct": round(daily_change_pct, 2)
            })
            
            return {
                "price": round(current_price, 2),
                "previous_close": round(previous_close, 2),
                "daily_change_pct": round(daily_change_pct, 2)
            }
        
        # If we only have current price
        if current_price:
            set_cache(ticker, {
                "price": round(current_price, 2),
                "previous_close": None,
                "daily_change_pct": None
            })
            
            return {
                "price": round(current_price, 2),
                "previous_close": None,
                "daily_change_pct": None
            }
            
    except Exception as e:
        print(f"[YF] Error fetching data for {ticker}: {e}")
    
    print(f"[YF] No price data found for {ticker}")
    return None



# ────────────────────────────────────────────────────────────────────
# Historical price fetch
# ────────────────────────────────────────────────────────────────────
def fetch_history_yf(
    ticker: str,
    start: str,
    end: str | None = None,
    ) -> list[dict]:
    """
    Returns list of {"date": "YYYY-MM-DD", "close": float}
    """

    print(f"[DEBUG] fetch_history_yf called with ticker={ticker}, start={start}, end={end}")
    # Validate date
    try:
        datetime.strptime(start, "%Y-%m-%d")
    except ValueError:
        print(f"[ERROR] Invalid start date format: {start}")
        return []

    ticker = ticker.strip().upper()
    yf_symbol = f"{ticker}.NS"
    end = end or datetime.utcnow().date().isoformat()
    cache_key = (ticker, start, end)

    # Return cached if fresh
    cached = hist_cache.get(cache_key)
    if cached and time.time() - cached["ts"] < HIST_CACHE_TTL:
        return cached["series"]

    try:
        # Use Ticker.history() instead of download()
        stock = yf.Ticker(yf_symbol)
        hist = stock.history(
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,   # or False, per your preference
        )

        if hist.empty or "Close" not in hist.columns:
            print(f"[YF] Empty or no 'Close' for {ticker}.")
            return []

        output = []
        for dt, row in hist.iterrows():
            close = row.get("Close")
            if close is None:
                continue
            output.append({
                "date": dt.date().isoformat(),
                "close": round(float(close), 2),
            })

        if not output:
            print(f"[YF] No valid data for {ticker} from {start} to {end}")
            return []

        # Cache and return
        hist_cache[cache_key] = {"series": output, "ts": time.time()}
        return output

    except Exception as e:
        print(f"[YF] history() failed for {ticker}: {e}")
        return []


# ────────────────────────────────────────────────────────────────────
# FastAPI setup
# ────────────────────────────────────────────────────────────────────
app = FastAPI(title="ClearTrack API – Indian Stocks (yfinance)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────────────────────────────────────────
# Pydantic model
# ────────────────────────────────────────────────────────────────────
class StockPrice(BaseModel):
    ticker: str
    price: float
    previous_close: float | None = None
    daily_change_pct: float | None = None
    currency: str = "INR"
    exchange: str = "NSE"


class PricePoint(BaseModel):
    date: str   # YYYY-MM-DD
    close: float

class HistoryResponse(BaseModel):
    ticker: str
    series: List[PricePoint]


# ────────────────────────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "ClearTrack API running with yfinance.",
        "cache_entries": len(cache),
        "cache_ttl_seconds": CACHE_TTL,
    }

@app.get("/api/history/{ticker}", response_model=HistoryResponse)
def get_history(
    ticker: str,
    start: str = Query(..., description="YYYY-MM-DD"),
    end: str | None = Query(None, description="YYYY-MM-DD (optional)"),
):
    """
    Return daily closing prices from `start` to `end` (today if end is None)
    """
    series = fetch_history_yf(ticker, start, end)
    if not series:
        raise HTTPException(
            status_code=404,
            detail=f"No historical data for '{ticker.upper()}' in given range.",
        )
    return {"ticker": ticker.upper(), "series": series}

@app.get("/api/price/{ticker}", response_model=StockPrice)
def get_price(ticker: str):
    data = fetch_price_yf(ticker)
    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Could not retrieve price for '{ticker.upper()}'. "
                   f"Is the symbol correct and listed on NSE?",
        )

    return StockPrice(
        ticker=ticker.upper(),
        price=data["price"],
        previous_close=data.get("previous_close"),
        daily_change_pct=data.get("daily_change_pct"),
    )

@app.post("/api/prices")
def get_prices(tickers: List[str]):
    out: Dict[str, Dict] = {}
    for t in tickers:
        data = fetch_price_yf(t)
        if data is None:
            out[t.upper()] = {"success": False, "error": "price_not_found"}
        else:
            out[t.upper()] = {
                "success": True, 
                "price": data["price"], 
                "previous_close": data.get("previous_close"),
                "daily_change_pct": data.get("daily_change_pct"),
                "currency": "INR"
            }
    return out

@app.get("/api/popular-stocks")
def popular():
    result = []
    for sym, name in POPULAR.items():
        data = fetch_price_yf(sym)
        if data is not None:
            result.append({
                "ticker": sym, 
                "name": name, 
                "price": data["price"], 
                "daily_change_pct": data.get("daily_change_pct"),
                "currency": "INR"
            })
    return result

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cache_entries": len(cache),
    }

# ────────────────────────────────────────────────────────────────────
# Local dev entry  ➜  python main.py
# (Render will run uvicorn with these args automatically)
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"\n🚀  Starting ClearTrack API at http://localhost:{port} …\n")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)