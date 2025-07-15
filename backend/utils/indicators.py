from __future__ import annotations
import numpy as np
import pandas as pd

# ───── Optional TA‑Lib import ─────
try:
    import talib
    _USE_TALIB = True
except ModuleNotFoundError:
    _USE_TALIB = False

# ───── Short‑Term Core ─────
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    if _USE_TALIB:
        return pd.Series(talib.RSI(close.values, timeperiod=period), index=close.index)
    delta = close.diff()
    up   = np.where(delta > 0,  delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up   = pd.Series(up,   index=close.index).rolling(period).mean()
    roll_down = pd.Series(down, index=close.index).rolling(period).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

def ema(close: pd.Series, period: int = 10) -> pd.Series:
    if _USE_TALIB:
        return pd.Series(talib.EMA(close.values, timeperiod=period), index=close.index)
    return close.ewm(span=period, adjust=False).mean()

def ema_long(close: pd.Series, period: int = 50) -> pd.Series:
    return ema(close, period)   # uses same helper

def sma_long(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(window=period).mean()

def lag_return(close: pd.Series, lag: int = 1) -> pd.Series:
    return close.pct_change(lag).shift(lag)

# ───── Momentum / Trend ─────
def rolling_momentum(close: pd.Series, window: int = 10) -> pd.Series:
    return close / close.shift(window) - 1

def trend_strength(close: pd.Series, window: int = 30) -> pd.Series:
    return (close - close.shift(window)) / close.shift(window)

# ───── MACD, BB%, ATR, CCI, ROC, etc. (unchanged) ─────
def macd_line(close: pd.Series) -> pd.Series:
    if _USE_TALIB:
        macd, _, _ = talib.MACD(close.values)
        return pd.Series(macd, index=close.index)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    return ema12 - ema26

def bollinger_percent_b(close: pd.Series, period: int = 20) -> pd.Series:
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (close - lower) / (upper - lower)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    if _USE_TALIB:
        return pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=period), index=close.index)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    return (tp - sma) / (0.015 * mad)

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    if _USE_TALIB:
        return pd.Series(talib.ADX(high.values, low.values, close.values, timeperiod=period), index=close.index)
    # Simple pandas fallback (true range method)
    plus_dm  = (high.diff()  > low.diff()).where(high.diff()  > 0, 0.0)
    minus_dm = (low.diff().abs() > high.diff()).where(low.diff() < 0, 0.0)
    tr = atr(high, low, close, 1).abs()
    plus_di  = 100 * (plus_dm.rolling(period).sum() / tr.rolling(period).sum())
    minus_di = 100 * (minus_dm.rolling(period).sum()  / tr.rolling(period).sum())
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    return dx.rolling(period).mean()

def rate_of_change(close: pd.Series, period: int = 5) -> pd.Series:
    return close.pct_change(period)

def force_index(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (close.diff() * volume).fillna(0)

def open_close_diff(open_: pd.Series, close: pd.Series) -> pd.Series:
    return close - open_

def high_low_range(high: pd.Series, low: pd.Series) -> pd.Series:
    return high - low

def rolling_volatility(close: pd.Series, period: int = 20) -> pd.Series:
    return close.pct_change().rolling(window=period).std()

# ───── Feature Column Builder ─────
def build_feature_columns(index_aliases: list[str] = []) -> list[str]:
    base = [
    "EMA", "SMA_10", "SMA_20", "BB%", "CCI", "ROC", "Volatility", "Momentum10",
    "EMA50", "SMA_50", "SMA_200", "Trend30", "Momentum30", "Volatility60", "ADX14", "LagReturn_5", "LagReturn_10"
]

    idx_feats = [f"{name}_{p}d_return" for name in index_aliases for p in (1, 5)]
    return base + idx_feats

# ───── Main Feature Generator ─────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # Short‑term set
    #df["RSI"]       = rsi(df["Close"])
    df["EMA"]       = ema(df["Close"])
    #df["LagReturn"] = lag_return(df["Close"])
    #df["MACD"]      = macd_line(df["Close"])
    df["SMA_10"]    = df["Close"].rolling(10).mean()
    df["SMA_20"]    = df["Close"].rolling(20).mean()
    df["BB%"]       = bollinger_percent_b(df["Close"])
    #df["ATR"]       = atr(df["High"], df["Low"], df["Close"])
    df["CCI"]       = cci(df["High"], df["Low"], df["Close"])
    df["ROC"]       = rate_of_change(df["Close"])
    # df["ForceIdx"]  = force_index(df["Close"], df["Volume"])
    # df["OC_Diff"]   = open_close_diff(df["Open"], df["Close"])
    # df["HL_Range"]  = high_low_range(df["High"], df["Low"])
    df["Volatility"]= rolling_volatility(df["Close"])
    df["Momentum10"]= rolling_momentum(df["Close"], 10)

    # Long‑term layer
    df["EMA50"]     = ema_long(df["Close"], 50)
    df["SMA_50"]    = sma_long(df["Close"], 50)
    df["SMA_200"]   = sma_long(df["Close"], 200)
    df["Trend30"]   = trend_strength(df["Close"], 30)
    df["Momentum30"]= rolling_momentum(df["Close"], 30)
    df["Volatility60"] = rolling_volatility(df["Close"], 60)
    df["ADX14"]     = adx(df["High"], df["Low"], df["Close"])
    df["LagReturn_5"] = df["Close"].pct_change(5).shift(1)
    df["LagReturn_10"] = df["Close"].pct_change(10).shift(1)


    return df.dropna()

# ───── Add Sectoral/Broad Indices (unchanged) ─────
def add_index_features(df: pd.DataFrame, index_paths: dict[str, str]) -> pd.DataFrame:
    for name, path in index_paths.items():
        idx = pd.read_csv(path)
        for col in ("Index Name", "Index", "Name"):
            if col in idx.columns:
                idx.drop(columns=col, inplace=True)
        idx["Date"] = pd.to_datetime(idx["Date"], format="%d %b %Y")
        idx["Close"] = (
            idx["Close"].astype(str).str.replace(",", "", regex=False).str.strip().astype(float)
        )
        idx.sort_values("Date", inplace=True)
        idx[f"{name}_1d_return"] = idx["Close"].pct_change(1)
        idx[f"{name}_5d_return"] = idx["Close"].pct_change(5)
        df = df.merge(
            idx[["Date", f"{name}_1d_return", f"{name}_5d_return"]],
            on="Date", how="left"
        )
    return df
