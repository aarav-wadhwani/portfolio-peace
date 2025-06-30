from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime
import yfinance as yf
import time
import os
from dotenv import load_dotenv

# ────────────────────────────────────────────────────────────────────────────────
# Environment & App setup
# ────────────────────────────────────────────────────────────────────────────────
load_dotenv()

app = FastAPI(title="ClearTrack API – Indian Stocks")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",            # local dev
        "https://cleartrack.vercel.app"     # production front-end
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────────────────────────────────────────────────────
# Models & simple in-memory cache
# ────────────────────────────────────────────────────────────────────────────────
class StockPrice(BaseModel):
    ticker: str
    price: float
    currency: str = "INR"
    exchange: str = "NSE"

# { "RELIANCE": {"price": ..., "timestamp": ...}, ... }
price_cache: Dict[str, Dict[str, float]] = {}
CACHE_TTL = 300  # seconds

# Optionally keep a list of “popular” tickers just for convenience routes
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

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────
def _get_cached(ticker: str):
    entry = price_cache.get(ticker)
    if entry and (time.time() - entry["timestamp"] < CACHE_TTL):
        return entry["price"]
    return None


def fetch_price(ticker: str) -> float | None:
    """
    Return the latest close (or last traded) price for an NSE symbol.

    Raises nothing—just returns None if data cannot be obtained.
    """
    ticker = ticker.strip().upper()
    cached = _get_cached(ticker)
    if cached is not None:
        return cached

    yf_symbol = f"{ticker}.NS"
    try:
        # 1) history() is the most reliable for equities
        hist = yf.Ticker(yf_symbol).history(period="2d", threads=False, progress=False)
        if not hist.empty:
            price = float(hist["Close"].iloc[-1])
        else:
            # 2) fallback inside yfinance (fast_info)
            info = yf.Ticker(yf_symbol).fast_info
            price = info.get("last_price") or info.get("previous_close")
            if price is None:
                return None
    except Exception as exc:
        print(f"[yfinance] error fetching {yf_symbol}: {exc}")
        return None

    # cache & return
    price_cache[ticker] = {"price": price, "timestamp": time.time()}
    return price


# ────────────────────────────────────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "ClearTrack API for Indian stocks is running.",
        "powered_by": "yfinance",
        "cache_size": len(price_cache),
    }


@app.get("/api/price/{ticker}", response_model=StockPrice)
def get_price(ticker: str):
    price = fetch_price(ticker)
    if price is None:
        raise HTTPException(
            status_code=404,
            detail=f"Could not retrieve price for '{ticker.upper()}'. "
                   f"Is the symbol correct and listed on NSE?",
        )
    return StockPrice(ticker=ticker.upper(), price=round(price, 2))


@app.post("/api/prices")
def get_prices(tickers: List[str]):
    results: Dict[str, Dict] = {}
    for t in tickers:
        price = fetch_price(t)
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
def popular():
    out = []
    for tkr, name in POPULAR_TICKERS.items():
        price = fetch_price(tkr)
        if price is not None:
            out.append(
                {
                    "ticker": tkr,
                    "name": name,
                    "price": round(price, 2),
                    "currency": "INR",
                }
            )
    return out


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cache_entries": len(price_cache),
    }


# ────────────────────────────────────────────────────────────────────────────────
# Run with `python main.py` for local dev
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    print("\n🚀  Starting ClearTrack API (yfinance-only)…")
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
