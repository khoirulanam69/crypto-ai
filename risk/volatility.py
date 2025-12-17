import numpy as np

def atr(candles, period=14):
    """
    candles: list of [ts, open, high, low, close, volume]
    """
    if len(candles) < period + 1:
        return 0.0

    highs = np.array([c[2] for c in candles])
    lows  = np.array([c[3] for c in candles])
    closes = np.array([c[4] for c in candles])

    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            abs(highs[1:] - closes[:-1]),
            abs(lows[1:] - closes[:-1])
        )
    )

    return np.mean(tr[-period:])
