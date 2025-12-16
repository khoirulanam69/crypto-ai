import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import os

MODEL = os.getenv("MODEL", "models/ppo_live.zip")
WINDOW = int(os.getenv("AI_WINDOW", "50"))

_model = None


def load_model():
    global _model
    if _model is None:
        _model = PPO.load(MODEL)
    return _model


def candles_to_features(candles: list):
    """
    candles: list of [ts, open, high, low, close, volume]
    return numpy obs sesuai ExchangeEnv
    """
    df = pd.DataFrame(
        candles,
        columns=["ts", "open", "high", "low", "close", "volume"]
    )

    closes = df["close"].values[-WINDOW:]

    mean = closes.mean() if closes.mean() != 0 else 1.0
    closes_norm = closes / mean

    # feature tambahan
    returns = np.diff(closes) / closes[:-1]
    volatility = np.std(returns) if len(returns) > 1 else 0.0

    cash_ratio = 1.0    # realtime cash ditangani env
    pos_ratio = 0.0     # realtime position ditangani env

    obs = np.concatenate(
        [closes_norm, [cash_ratio, pos_ratio]]
    ).astype(np.float32)

    return obs


class AIDecisionEngine:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.model = load_model()

    def decide(self, candles: list):
        """
        candles = list of OHLCV
        return action int
        """
        obs = candles_to_features(candles)
        action, _ = self.model.predict(obs, deterministic=False)
        return int(action)
