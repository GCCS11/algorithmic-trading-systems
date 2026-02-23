import pandas as pd


def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_atr(df, period=14):
    hl  = df['High'] - df['Low']
    hc  = (df['High'] - df['Close'].shift(1)).abs()
    lc  = (df['Low']  - df['Close'].shift(1)).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast    = compute_ema(series, fast)
    ema_slow    = compute_ema(series, slow)
    macd_line   = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(series, period=20, std_dev=2):
    mid   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


def add_indicators(df):
    df = df.copy()

    df['ema20'] = compute_ema(df['Close'], 20)
    df['ema50'] = compute_ema(df['Close'], 50)
    df['rsi']   = compute_rsi(df['Close'])
    df['atr']   = compute_atr(df)

    df['macd'], df['macd_signal'], df['macd_hist'] = compute_macd(df['Close'])
    df['bb_upper'], df['bb_mid'], df['bb_lower']   = compute_bollinger(df['Close'])

    indicator_cols = ['ema20', 'ema50', 'rsi', 'atr', 'macd', 'macd_signal', 'macd_hist', 'bb_upper', 'bb_mid', 'bb_lower']
    df = df.dropna(subset=indicator_cols).reset_index(drop=True)

    return df