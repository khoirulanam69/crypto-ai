# import pandas as pd
# import numpy as np


# # =========================================================
# # ATR — Average True Range
# # =========================================================
# def ATR(df: pd.DataFrame, period: int = 14) -> pd.Series:
#     """
#     Average True Range
#     Digunakan untuk stop loss, trailing stop, dan position sizing

#     df wajib punya kolom:
#     ['open', 'high', 'low', 'close']
#     """

#     high = df["high"]
#     low = df["low"]
#     close = df["close"]

#     prev_close = close.shift(1)

#     tr1 = high - low
#     tr2 = (high - prev_close).abs()
#     tr3 = (low - prev_close).abs()

#     true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
#     atr = true_range.rolling(window=period).mean()

#     return atr


# # =========================================================
# # EMA — Exponential Moving Average
# # =========================================================
# def EMA(series: pd.Series, period: int = 20) -> pd.Series:
#     return series.ewm(span=period, adjust=False).mean()


# # =========================================================
# # SMA — Simple Moving Average
# # =========================================================
# def SMA(series: pd.Series, period: int = 20) -> pd.Series:
#     return series.rolling(window=period).mean()


# # =========================================================
# # RSI — Relative Strength Index
# # =========================================================
# def RSI(series: pd.Series, period: int = 14) -> pd.Series:
#     delta = series.diff()

#     gain = delta.where(delta > 0, 0.0)
#     loss = -delta.where(delta < 0, 0.0)

#     avg_gain = gain.rolling(period).mean()
#     avg_loss = loss.rolling(period).mean()

#     rs = avg_gain / (avg_loss + 1e-9)
#     rsi = 100 - (100 / (1 + rs))

#     return rsi


# # =========================================================
# # VOLATILITY — Standard Deviation of Returns
# # =========================================================
# def volatility(series: pd.Series, period: int = 20) -> pd.Series:
#     returns = series.pct_change()
#     return returns.rolling(period).std()


# # =========================================================
# # TREND STRENGTH (EMA SLOPE)
# # =========================================================
# def trend_strength(series: pd.Series, period: int = 20) -> pd.Series:
#     ema = EMA(series, period)
#     slope = ema.diff()
#     return slope


# # =========================================================
# # SUPPORT / RESISTANCE (NAIVE)
# # =========================================================
# def support_resistance(df: pd.DataFrame, lookback: int = 50):
#     """
#     Mengembalikan level support & resistance sederhana
#     """
#     support = df["low"].rolling(lookback).min()
#     resistance = df["high"].rolling(lookback).max()
#     return support, resistance
