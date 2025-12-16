import pandas as pd

def ATR(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)

    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low - close).abs()
    ], axis=1).max(axis=1)

    return tr.rolling(period).mean()
