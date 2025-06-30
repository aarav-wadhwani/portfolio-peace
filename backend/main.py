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
        "https://portfolio-peace.vercel.app/" ,
        "https://portfolio-peace.onrender.com"    # production front-end
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
    import sys
    ticker = ticker.strip().upper()
    yf_symbol = f"{ticker}.NS"
    print(f"Fetching: {yf_symbol}", file=sys.stderr)

    try:
        stock = yf.Ticker(yf_symbol)
        hist = stock.history(period="2d", threads=False, progress=False)

        if not hist.empty:
            price = float(hist["Close"].iloc[-1])
            print(f"✔️ Got price for {yf_symbol}: {price}", file=sys.stderr)
            price_cache[ticker] = {"price": price, "timestamp": time.time()}
            return price
        else:
            print(f"⚠️ Empty history for {yf_symbol}: {hist}", file=sys.stderr)

        # Try fallback to fast_info
        try:
            info = stock.fast_info
            price = info.get("last_price") or info.get("previous_close")
            if price:
                print(f"✔️ Fallback fast_info price for {yf_symbol}: {price}", file=sys.stderr)
                price_cache[ticker] = {"price": price, "timestamp": time.time()}
                return price
        except Exception as fallback_e:
            print(f"❌ fast_info failed: {fallback_e}", file=sys.stderr)

    except Exception as e:
        print(f"❌ yfinance fetch failed for {yf_symbol}: {e}", file=sys.stderr)

    return None



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
