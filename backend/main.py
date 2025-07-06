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
# Simple in-memory cache  {TICKER: {"price": float, "ts": float}}
# ────────────────────────────────────────────────────────────────────
cache: Dict[str, Dict[str, float]] = {}

def get_cached(ticker: str):
    entry = cache.get(ticker)
    if entry and (time.time() - entry["ts"] < CACHE_TTL):
        return entry["price"]
    return None

def set_cache(ticker: str, price: float):
    cache[ticker] = {"price": price, "ts": time.time()}

# ────────────────────────────────────────────────────────────────────
# yfinance fetch (history ➜ fast_info)
# ────────────────────────────────────────────────────────────────────
def fetch_price_yf(ticker: str) -> float | None:
    ticker = ticker.strip().upper()
    if cached := get_cached(ticker):
        return cached

    yf_symbol = f"{ticker}.NS"
    stock = yf.Ticker(yf_symbol)

    # 1) history (most reliable)
    try:
        hist = stock.history(period="2d")
        if not hist.empty:
            price = float(hist["Close"].iloc[-1])
            set_cache(ticker, price)
            return price
    except Exception as e:
        print(f"[YF] history() failed for {ticker}: {e}")

    # 2) fast_info fallback
    try:
        info = stock.fast_info
        price = info.get("last_price") or info.get("previous_close")
        if price:
            set_cache(ticker, float(price))
            return float(price)
    except Exception as e:
        print(f"[YF] fast_info failed for {ticker}: {e}")

    print(f"[YF] No price for {ticker}")
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
    price = fetch_price_yf(ticker)
    if price is None:
        raise HTTPException(
            status_code=404,
            detail=f"Could not retrieve price for '{ticker.upper()}'. "
                   f"Is the symbol correct and listed on NSE?",
        )
    return StockPrice(ticker=ticker.upper(), price=round(price, 2))

@app.post("/api/prices")
def get_prices(tickers: List[str]):
    out: Dict[str, Dict] = {}
    for t in tickers:
        price = fetch_price_yf(t)
        if price is None:
            out[t.upper()] = {"success": False, "error": "price_not_found"}
        else:
            out[t.upper()] = {"success": True, "price": round(price, 2), "currency": "INR"}
    return out

@app.get("/api/popular-stocks")
def popular():
    result = []
    for sym, name in POPULAR.items():
        price = fetch_price_yf(sym)
        if price is not None:
            result.append(
                {"ticker": sym, "name": name, "price": round(price, 2), "currency": "INR"}
            )
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
