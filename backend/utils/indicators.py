from __future__ import annotations
import numpy as np
import pandas as pd

# ───── Optional TA‑Lib import ─────
try:
    import talib
    _USE_TALIB = True
except ModuleNotFoundError:
    _USE_TALIB = False

_USE_TALIB = False

# ───── Core Technical Indicators ─────
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index"""
    if _USE_TALIB:
        return pd.Series(talib.RSI(close.values, timeperiod=period), index=close.index)
    delta = close.diff()
    up   = np.where(delta > 0,  delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up   = pd.Series(up,   index=close.index).rolling(period).mean()
    roll_down = pd.Series(down, index=close.index).rolling(period).mean()
    rs = roll_up / (roll_down + 1e-10)  # Avoid division by zero
    return 100 - (100 / (1 + rs))

def ema(close: pd.Series, period: int = 10) -> pd.Series:
    """Exponential Moving Average"""
    if _USE_TALIB:
        return pd.Series(talib.EMA(close.values, timeperiod=period), index=close.index)
    return close.ewm(span=period, adjust=False).mean()

def sma(close: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average"""
    return close.rolling(window=period).mean()

# ───── MACD Components ─────
def macd_components(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns MACD line, signal line, and histogram"""
    if _USE_TALIB:
        macd, signal, hist = talib.MACD(close.values)
        return (pd.Series(macd, index=close.index), 
                pd.Series(signal, index=close.index),
                pd.Series(hist, index=close.index))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# ───── Bollinger Bands ─────
def bollinger_bands(close: pd.Series, period: int = 20, std_dev: int = 2) -> dict:
    """Returns upper band, middle band (SMA), lower band, and %B"""
    middle = sma(close, period)
    std = close.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    percent_b = (close - lower) / (upper - lower + 1e-10)
    
    return {
        'upper': upper,
        'middle': middle,
        'lower': lower,
        'percent_b': percent_b
    }

# ───── ATR (Average True Range) ─────
def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range - volatility indicator"""
    if _USE_TALIB:
        return pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=period), index=close.index)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ───── CCI (Commodity Channel Index) ─────
def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Commodity Channel Index"""
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    return (tp - sma_tp) / (0.015 * mad + 1e-10)

# ───── Fixed ADX (Average Directional Index) ─────
def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index - trend strength"""
    if _USE_TALIB:
        return pd.Series(talib.ADX(high.values, low.values, close.values, timeperiod=period), index=close.index)
    
    # Calculate True Range first
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate directional movement
    high_diff = high.diff()
    low_diff = low.shift().diff()  # Fixed: should be low.shift().diff()
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # Convert to Series for proper indexing
    plus_dm_series = pd.Series(plus_dm, index=close.index)
    minus_dm_series = pd.Series(minus_dm, index=close.index)
    
    # Smoothed directional indicators
    tr_sum = tr.rolling(period).sum()
    plus_di = 100 * plus_dm_series.rolling(period).sum() / (tr_sum + 1e-10)
    minus_di = 100 * minus_dm_series.rolling(period).sum() / (tr_sum + 1e-10)
    
    # ADX calculation
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx_result = dx.rolling(period).mean()
    
    return adx_result.fillna(0)

# ───── Rate of Change ─────
def rate_of_change(close: pd.Series, period: int = 10) -> pd.Series:
    """Rate of Change - momentum indicator"""
    return ((close - close.shift(period)) / (close.shift(period) + 1e-10)) * 100

# ───── Force Index ─────
def force_index(close: pd.Series, volume: pd.Series, period: int = 13) -> pd.Series:
    """Force Index - combines price and volume"""
    fi = close.diff() * volume
    return fi.ewm(span=period, adjust=False).mean()

# ───── Volatility Measures ─────
def rolling_volatility(close: pd.Series, period: int = 20) -> pd.Series:
    """Rolling standard deviation of returns"""
    returns = close.pct_change()
    return returns.rolling(window=period).std() * np.sqrt(252)  # Annualized

# ───── Enhanced Advanced Features ─────
def price_to_ma_ratios(close: pd.Series, sma_50: pd.Series, sma_200: pd.Series, ema_50: pd.Series) -> dict:
    """Price relative to moving averages"""
    return {
        'price_to_sma50_ratio': close / (sma_50 + 1e-10),
        'price_to_sma200_ratio': close / (sma_200 + 1e-10),
        'price_to_ema50_ratio': close / (ema_50 + 1e-10),
    }

def moving_average_signals(sma_50: pd.Series, sma_200: pd.Series, ema_12: pd.Series, ema_26: pd.Series) -> dict:
    """Trading signals from moving averages"""
    return {
        'golden_cross': (sma_50 > sma_200).astype(int),
        'death_cross': (sma_50 < sma_200).astype(int),
        'ema_bullish': (ema_12 > ema_26).astype(int),
    }

def momentum_features(close: pd.Series, rsi_val: pd.Series, macd_hist: pd.Series) -> dict:
    """Advanced momentum features"""
    price_change_20 = close.pct_change(20)
    
    return {
        'momentum_10d': (close / (close.shift(10) + 1e-10) - 1) * 100,
        'momentum_30d': (close / (close.shift(30) + 1e-10) - 1) * 100,
        'rsi_oversold': (rsi_val < 30).astype(int),
        'rsi_overbought': (rsi_val > 70).astype(int),
        'price_rsi_divergence': ((price_change_20 > 0) & (rsi_val < 50)).astype(int),
        'macd_bullish': (macd_hist > 0).astype(int),
    }

def volatility_regime(volatility_20: pd.Series, volatility_60: pd.Series) -> pd.Series:
    """Classify volatility regime"""
    vol_ratio = volatility_20 / (volatility_60 + 1e-10)
    conditions = [
        vol_ratio < 0.8,
        (vol_ratio >= 0.8) & (vol_ratio <= 1.2),
        vol_ratio > 1.2
    ]
    choices = [0, 1, 2]  # Low, Normal, High volatility regime
    return pd.Series(np.select(conditions, choices), index=volatility_20.index)

def trend_strength_features(close: pd.Series, adx_val: pd.Series) -> dict:
    """Trend strength indicators"""
    returns_5d = close.pct_change(5)
    returns_20d = close.pct_change(20)
    
    return {
        'trend_strength_5d': returns_5d.abs(),
        'trend_strength_20d': returns_20d.abs(),
        'strong_trend': (adx_val > 25).astype(int),
        'trend_consistency': (returns_5d * returns_20d > 0).astype(int),
    }

# ───── Market Context Features ─────
def market_breadth_indicators(df: pd.DataFrame, index_columns: list) -> dict:
    """Market breadth indicators using multiple indices"""
    features = {}
    
    # Basic market breadth
    idx_1d = [col for col in index_columns if col.endswith('_1d_return')]
    idx_5d = [col for col in index_columns if col.endswith('_5d_return')]
    
    if len(idx_1d) >= 3:
        features['market_breadth'] = df[idx_1d[:3]].mean(axis=1)
    if len(idx_5d) >= 3:
        features['market_breadth_5d'] = df[idx_5d[:3]].mean(axis=1)
    
    # Advanced breadth indicators
    if len(idx_1d) >= 2:
        advancing_count = sum((df[col] > 0).astype(int) for col in idx_1d)
        declining_count = sum((df[col] < 0).astype(int) for col in idx_1d)
        total_count = len(idx_1d)
        
        features['advance_decline_ratio'] = advancing_count / (total_count + 1e-10)
    
    return features

def sector_rotation_signals(df: pd.DataFrame) -> dict:
    """Sector rotation and leadership signals"""
    features = {}
    
    # IT vs Banking rotation
    if 'NIFTYIT_1d_return' in df.columns and 'NIFTYBANK_1d_return' in df.columns:
        features['sector_momentum'] = df['NIFTYIT_1d_return'] - df['NIFTYBANK_1d_return']
        features['it_outperformance'] = (df['NIFTYIT_1d_return'] > df['NIFTYBANK_1d_return']).astype(int)
    
    if 'NIFTYIT_5d_return' in df.columns and 'NIFTYBANK_5d_return' in df.columns:
        features['sector_rotation'] = df['NIFTYIT_5d_return'] - df['NIFTYBANK_5d_return']
        features['it_leadership_5d'] = (df['NIFTYIT_5d_return'] > df['NIFTYBANK_5d_return']).astype(int)
    
    # Large vs Small cap rotation
    if 'NIFTY50_1d_return' in df.columns and 'NIFTYSMALLCAP250_1d_return' in df.columns:
        features['large_small_rotation'] = df['NIFTY50_1d_return'] - df['NIFTYSMALLCAP250_1d_return']
        features['large_cap_leadership'] = (df['NIFTY50_1d_return'] > df['NIFTYSMALLCAP250_1d_return']).astype(int)
    
    return features

def fear_greed_indicators(df: pd.DataFrame) -> dict:
    """Fear and greed indicators based on volatility and momentum"""
    features = {}
    
    if 'volatility_20' in df.columns and 'volatility_60' in df.columns:
        # Volatility-based fear index
        features['fear_greed_index'] = (df['volatility_20'] - df['volatility_60']) / (df['volatility_60'] + 1e-10)
        features['volatility_spike'] = (df['volatility_20'] > df['volatility_60'] * 1.5).astype(int)
        features['low_volatility_regime'] = (df['volatility_20'] < df['volatility_60'] * 0.8).astype(int)
    
    # VIX-like calculations
    if 'Close' in df.columns:
        returns = df['Close'].pct_change()
        
        # Fear indicator based on consecutive negative days
        negative_days = (returns < 0).astype(int)
        features['fear_streak'] = negative_days.rolling(5).sum()
        
        # Greed indicator based on consecutive positive days
        positive_days = (returns > 0).astype(int)
        features['greed_streak'] = positive_days.rolling(5).sum()
    
    return features

def momentum_consistency_features(df: pd.DataFrame) -> dict:
    """Multi-timeframe momentum consistency"""
    features = {}
    
    if 'momentum_10d' in df.columns and 'momentum_30d' in df.columns:
        # Momentum alignment
        features['momentum_consistency'] = (
            (df['momentum_10d'] > 0).astype(int) * 0.3 +
            (df['momentum_30d'] > 0).astype(int) * 0.7
        )
        
        # Momentum acceleration/deceleration
        features['momentum_acceleration'] = df['momentum_10d'] - df['momentum_30d']
        features['momentum_divergence'] = ((df['momentum_10d'] > 0) & (df['momentum_30d'] < 0)).astype(int)
    
    return features

def price_position_features(df: pd.DataFrame) -> dict:
    """Price position relative to recent ranges"""
    features = {}
    
    if all(col in df.columns for col in ['Close', 'High', 'Low']):
        # 20-day range position
        high_20 = df['High'].rolling(20).max()
        low_20 = df['Low'].rolling(20).min()
        features['price_position'] = (df['Close'] - low_20) / (high_20 - low_20 + 1e-10)
        
        # 60-day range position
        high_60 = df['High'].rolling(60).max()
        low_60 = df['Low'].rolling(60).min()
        features['price_position_60'] = (df['Close'] - low_60) / (high_60 - low_60 + 1e-10)
        
        # Support/Resistance breaks
        features['above_20d_high'] = (df['Close'] > high_20.shift(1)).astype(int)
        features['below_20d_low'] = (df['Close'] < low_20.shift(1)).astype(int)
        
        # Near highs/lows
        features['near_20d_high'] = (features['price_position'] > 0.9).astype(int)
        features['near_20d_low'] = (features['price_position'] < 0.1).astype(int)
    
    return features

def volume_analysis_features(df: pd.DataFrame) -> dict:
    """Volume-based analysis features"""
    features = {}
    
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        volume = df['Volume']
        
        # Volume moving averages
        vol_sma_20 = volume.rolling(20).mean()
        vol_sma_50 = volume.rolling(50).mean()
        
        features['volume_ratio_20'] = volume / (vol_sma_20 + 1e-10)
        features['volume_ratio_50'] = volume / (vol_sma_50 + 1e-10)
        
        # Volume spikes
        features['volume_spike'] = (volume > vol_sma_20 * 2).astype(int)
        features['low_volume'] = (volume < vol_sma_20 * 0.5).astype(int)
        
        # Price-volume relationship
        if 'Close' in df.columns:
            price_change = df['Close'].pct_change()
            features['price_volume_trend'] = price_change * np.log(volume + 1)
    else:
        # Default values when volume is not available
        features['volume_ratio_20'] = 1.0
        features['volume_ratio_50'] = 1.0
        features['volume_spike'] = 0
        features['low_volume'] = 0
        features['price_volume_trend'] = 0.0
    
    return features

def regime_detection_features(df: pd.DataFrame) -> dict:
    """Market regime detection features"""
    features = {}
    
    if 'Close' in df.columns:
        # Trend regime
        sma_20 = df['Close'].rolling(20).mean()
        sma_50 = df['Close'].rolling(50).mean()
        
        features['uptrend_regime'] = (sma_20 > sma_50).astype(int)
        features['trend_strength'] = abs(sma_20 - sma_50) / (sma_50 + 1e-10)
        
        # Volatility regime
        if 'volatility_20' in df.columns:
            vol_percentile = df['volatility_20'].rolling(252).rank(pct=True)
            features['high_vol_regime'] = (vol_percentile > 0.8).astype(int)
            features['low_vol_regime'] = (vol_percentile < 0.2).astype(int)
    
    return features

# ───── Enhanced Feature Column Names ─────
def build_feature_columns(index_aliases: list[str] = None) -> list[str]:
    """Build list of all feature column names including new market context features"""
    if index_aliases is None:
        index_aliases = []
        
    base_features = [
        # Moving averages
        'sma_50', 'sma_200', 'ema_50',
        
        # Price ratios
        'price_to_sma50_ratio', 'price_to_sma200_ratio', 'price_to_ema50_ratio',
        
        # Technical indicators
        'rsi', 'macd_line', 'macd_histogram', 'bb_percent_b',
        'cci', 'roc', 'atr', 'adx', 'force_index',
        
        # Volatility
        'volatility_20', 'volatility_60', 'volatility_regime',
        
        # Signals
        'golden_cross', 'death_cross', 'ema_bullish', 'macd_bullish',
        
        # Momentum
        'momentum_10d', 'momentum_30d',
        'rsi_oversold', 'rsi_overbought', 'price_rsi_divergence',
        
        # Trend
        'trend_strength_5d', 'trend_strength_20d',
        'strong_trend', 'trend_consistency',
        
        # Calendar
        'day_of_week', 'month',
        
        # Volume analysis
        'volume_ratio_20', 'volume_ratio_50', 'volume_spike', 'low_volume',
        'price_volume_trend',
        
        # Regime detection
        'uptrend_regime', 'trend_strength', 'high_vol_regime', 'low_vol_regime',
    ]
    
    # Add index features
    index_features = []
    for idx in index_aliases:
        index_features.extend([
            f'{idx}_1d_return',
            f'{idx}_5d_return',
        ])
    
    # Add interaction features
    if 'NIFTYBANK' in index_aliases and 'NIFTYIT' in index_aliases:
        index_features.append('bank_it_divergence')
    
    # Add market context features (only if we have index data)
    market_context_features = []
    if len(index_aliases) > 0:
        market_context_features = [
            # Market breadth (only add if we have enough indices)
            'market_breadth', 'market_breadth_5d', 'advance_decline_ratio',
            
            # Sector rotation
            'sector_momentum', 'it_outperformance', 'sector_rotation', 'it_leadership_5d',
            'large_small_rotation', 'large_cap_leadership',
            
            # Fear & Greed
            'fear_greed_index', 'volatility_spike', 'low_volatility_regime',
            'fear_streak', 'greed_streak',
            
            # Momentum consistency
            'momentum_consistency', 'momentum_acceleration', 'momentum_divergence',
            
            # Price position
            'price_position', 'price_position_60', 'above_20d_high', 'below_20d_low',
            'near_20d_high', 'near_20d_low',
        ]
    
    return base_features + index_features + market_context_features

# ───── Enhanced Main Feature Engineering Function ─────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators and features to dataframe"""
    df = df.copy()
    
    # Basic moving averages
    df['sma_50'] = sma(df['Close'], 50)
    df['sma_200'] = sma(df['Close'], 200)
    df['ema_12'] = ema(df['Close'], 12)
    df['ema_26'] = ema(df['Close'], 26)
    df['ema_50'] = ema(df['Close'], 50)
    
    # Price ratios
    ratios = price_to_ma_ratios(df['Close'], df['sma_50'], df['sma_200'], df['ema_50'])
    for key, value in ratios.items():
        df[key] = value
    
    # RSI
    df['rsi'] = rsi(df['Close'])
    
    # MACD
    macd_line, macd_signal, macd_hist = macd_components(df['Close'])
    df['macd_line'] = macd_line
    df['macd_signal'] = macd_signal
    df['macd_histogram'] = macd_hist
    
    # Bollinger Bands
    bb = bollinger_bands(df['Close'])
    df['bb_upper'] = bb['upper']
    df['bb_lower'] = bb['lower']
    df['bb_percent_b'] = bb['percent_b']
    
    # Other indicators
    df['cci'] = cci(df['High'], df['Low'], df['Close'])
    df['roc'] = rate_of_change(df['Close'])
    df['atr'] = atr(df['High'], df['Low'], df['Close'])
    df['adx'] = adx(df['High'], df['Low'], df['Close'])
    
    # Volume-based
    if 'Volume' in df.columns:
        df['force_index'] = force_index(df['Close'], df['Volume'])
    else:
        df['force_index'] = 0
    
    # Volatility
    df['volatility_20'] = rolling_volatility(df['Close'], 20)
    df['volatility_60'] = rolling_volatility(df['Close'], 60)
    df['volatility_regime'] = volatility_regime(df['volatility_20'], df['volatility_60'])
    
    # Moving average signals
    ma_signals = moving_average_signals(df['sma_50'], df['sma_200'], df['ema_12'], df['ema_26'])
    for key, value in ma_signals.items():
        df[key] = value
    
    # Momentum features
    momentum_feat = momentum_features(df['Close'], df['rsi'], df['macd_histogram'])
    for key, value in momentum_feat.items():
        df[key] = value
    
    # Trend features
    trend_feat = trend_strength_features(df['Close'], df['adx'])
    for key, value in trend_feat.items():
        df[key] = value
    
    # Calendar features
    if 'Date' in df.columns:
        df['day_of_week'] = pd.to_datetime(df['Date']).dt.dayofweek
        df['month'] = pd.to_datetime(df['Date']).dt.month
    
    # Add volume analysis features
    volume_feat = volume_analysis_features(df)
    for key, value in volume_feat.items():
        df[key] = value
    
    # Add regime detection features
    regime_feat = regime_detection_features(df)
    for key, value in regime_feat.items():
        df[key] = value
    
    # Drop intermediate columns not needed for modeling
    cols_to_drop = ['ema_12', 'ema_26', 'macd_signal', 'bb_upper', 'bb_lower']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Replace infinities with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Only drop rows where all values are NaN, so we keep any row with at least one valid feature
    df = df.dropna(how='all')

    return df


# ───── Fixed Add Index Features Function ─────
def add_index_features(df: pd.DataFrame, index_paths: dict[str, str]) -> pd.DataFrame:
    """Add sectoral index features for market breadth analysis"""
    df = df.copy()
    
    for name, path in index_paths.items():
        try:
            # Read index data
            idx = pd.read_csv(path)
            
            # Clean column names
            for col in ("Index Name", "Index", "Name"):
                if col in idx.columns:
                    idx.drop(columns=col, inplace=True)
            
            # Process date
            idx["Date"] = pd.to_datetime(idx["Date"], format="%d %b %Y")
            
            # Clean close price
            idx["Close"] = (
                idx["Close"]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
                .astype(float)
            )
            
            # Sort by date
            idx.sort_values("Date", inplace=True)
            
            # Calculate returns
            idx[f"{name}_1d_return"] = idx["Close"].pct_change(1)
            idx[f"{name}_5d_return"] = idx["Close"].pct_change(5)
            
            # Merge with main dataframe
            df = df.merge(
                idx[["Date", f"{name}_1d_return", f"{name}_5d_return"]],
                on="Date",
                how="left"
            )
            
        except Exception as e:
            print(f"Warning: Could not process index {name}: {e}")
            df[f"{name}_1d_return"] = np.nan
            df[f"{name}_5d_return"] = np.nan
    
    # Add market context features after index data is available
    index_columns = [col for col in df.columns if any(idx in col for idx in index_paths.keys())]
    
    # Only add market context features if we have index data
    if index_columns:
        # Market breadth indicators
        breadth_feat = market_breadth_indicators(df, index_columns)
        for key, value in breadth_feat.items():
            df[key] = value
        
        # Sector rotation signals
        rotation_feat = sector_rotation_signals(df)
        for key, value in rotation_feat.items():
            df[key] = value
        
        # Fear & greed indicators
        fear_greed_feat = fear_greed_indicators(df)
        for key, value in fear_greed_feat.items():
            df[key] = value
        
        # Momentum consistency
        momentum_consist_feat = momentum_consistency_features(df)
        for key, value in momentum_consist_feat.items():
            df[key] = value
        
        # Price position features
        price_pos_feat = price_position_features(df)
        for key, value in price_pos_feat.items():
            df[key] = value
    
    # Add original interaction features
    if 'NIFTYBANK_5d_return' in df.columns and 'NIFTYIT_5d_return' in df.columns:
        df['bank_it_divergence'] = df['NIFTYBANK_5d_return'] - df['NIFTYIT_5d_return']
    
    return df

# ───── Fixed Export ─────
# Export key feature list for backward compatibility
try:
    FEATURE_COLUMNS = build_feature_columns([])
except Exception as e:
    print(f"Warning: Could not build feature columns: {e}")
    FEATURE_COLUMNS = []