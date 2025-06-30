# main.py
# -------------------------------------------------------------------
# ClearTrack – FastAPI backend for Indian stocks (NSE live prices)
# -------------------------------------------------------------------
import os
import time
from datetime import datetime
from typing import Dict, List

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ────────────────────────────────────────────────────────────────────
#  Environment / config
# ────────────────────────────────────────────────────────────────────
load_dotenv()

ALLOWED_ORIGINS = [
    "http://localhost:3000",                # local dev
    "https://cleartrack.vercel.app",        # prod front-end on Vercel
]

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL", 300))  # default: 5 min

# Popular tickers you may want to show on the front-end “quick list”
POPULAR_TICKERS = {
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
#  Requests session pre-configured for NSE endpoints
# ────────────────────────────────────────────────────────────────────
session = requests.Session()
session.headers.update(
    {
        # NSE blocks generic user-agents; mimic a normal browser
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    }
)

# ────────────────────────────────────────────────────────────────────
#  Simple cache  { "TICKER": { "price": float, "timestamp": float } }
# ────────────────────────────────────────────────────────────────────
price_cache: Dict[str, Dict[str, float]] = {}

def _get_cached_price(ticker: str):
    entry = price_cache.get(ticker)
    if entry and (time.time() - entry["timestamp"] < CACHE_TTL_SECONDS):
        return entry["price"]
    return None

def _set_cache(ticker: str, price: float):
    price_cache[ticker] = {"price": price, "timestamp": time.time()}

# ────────────────────────────────────────────────────────────────────
#  Low-level NSE fetch
# ────────────────────────────────────────────────────────────────────
def fetch_price_nse(ticker: str) -> float | None:
    """
    Query https://www.nseindia.com/api/quote-equity?symbol=...
    Return last traded price as float, or None on failure.
    """
    ticker = ticker.strip().upper()

    # 0. serve from cache if fresh
    cached = _get_cached_price(ticker)
    if cached is not None:
        return cached

    # 1. We must first hit the root page once to obtain cookies
    try:
        session.get("https://www.nseindia.com", timeout=5)
    except Exception as e:
        print(f"[NSE] cookie pre-fetch failed: {e}")

    url = f"https://www.nseindia.com/api/quote-equity?symbol={ticker}"
    try:
        r = session.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        price_raw = data.get("priceInfo", {}).get("lastPrice")
        if price_raw:
            # Convert '2,845.65' ➜ 2845.65
            price = float(str(price_raw).replace(",", ""))
            _set_cache(ticker, price)
            return price
        else:
            print(f"[NSE] lastPrice missing for {ticker}")
            return None
    except Exception as e:
        print(f"[NSE] fetch error for {ticker}: {e}")
        return None

# ────────────────────────────────────────────────────────────────────
#  FastAPI setup
# ────────────────────────────────────────────────────────────────────
app = FastAPI(title="ClearTrack API – Indian Stocks (NSE)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────────────────────────────────────────
#  Pydantic models
# ────────────────────────────────────────────────────────────────────
class StockPrice(BaseModel):
    ticker: str
    price: float
    currency: str = "INR"
    exchange: str = "NSE"

# ────────────────────────────────────────────────────────────────────
#  Routes
# ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "ClearTrack API for Indian stocks is running.",
        "powered_by": "Official NSE JSON endpoint",
        "cache_entries": len(price_cache),
        "cache_ttl_seconds": CACHE_TTL_SECONDS,
    }

@app.get("/api/price/{ticker}", response_model=StockPrice)
def get_single_price(ticker: str):
    price = fetch_price_nse(ticker)
    if price is None:
        raise HTTPException(
            status_code=404,
            detail=f"Could not retrieve price for '{ticker.upper()}'. "
                   f"Is the symbol correct and listed on NSE?",
        )
    return StockPrice(ticker=ticker.upper(), price=round(price, 2))

@app.post("/api/prices")
def get_multiple_prices(tickers: List[str]):
    results: Dict[str, Dict] = {}
    for t in tickers:
        price = fetch_price_nse(t)
        if price is None:
            results[t.upper()] = {"success": False, "error": "price_not_found"}
        else:
            results[t.upper()] = {
                "success": True,
                "price": round(price, 2),
                "currency": "INR",
            }
    return results

@app.get("/api/popular-stocks")
def get_popular_stocks():
    out = []
    for symbol, name in POPULAR_TICKERS.items():
        price = fetch_price_nse(symbol)
        if price is not None:
            out.append(
                {
                    "ticker": symbol,
                    "name": name,
                    "price": round(price, 2),
                    "currency": "INR",
                }
            )
    return out

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cache_entries": len(price_cache),
    }

# ────────────────────────────────────────────────────────────────────
#  Local dev entry point  ➜  `python main.py`
#  (Render will run the same uvicorn command automatically)
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    print(f"\n🚀  Starting ClearTrack API on http://localhost:{port} (NSE)…\n")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
