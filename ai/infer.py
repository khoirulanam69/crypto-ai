# ai/infer.py
import numpy as np
import pandas as pd
import os
from stable_baselines3 import PPO
from executor.state_manager import StateManager
from executor.order_manager import OrderManager

MODEL = os.getenv("MODEL", "models/ppo_live.zip")
WINDOW = int(os.getenv("AI_WINDOW", "50"))

_model = None


def load_model():
    global _model
    if _model is None:
        _model = PPO.load(MODEL)
    return _model


def candles_to_features(candles, state):
    """
    candles: list [ts, open, high, low, close, volume]
    state: dict dari StateManager
    """

    df = pd.DataFrame(
        candles,
        columns=["ts", "open", "high", "low", "close", "volume"]
    )

    closes = df["close"].values[-WINDOW:]

    mean = closes.mean() if closes.mean() != 0 else 1.0
    closes_norm = closes / mean

    # ====== STATE REAL ======
    cash = state["cash"]
    position = state["position"]
    equity = state["equity"]

    cash_ratio = cash / (equity + 1e-9)
    pos_ratio = position

    obs = np.concatenate(
        [closes_norm, [cash_ratio, pos_ratio]]
    ).astype(np.float32)

    return obs


class AIDecisionEngine:
    def __init__(self, order_manager: OrderManager):
        self.om = order_manager
        self.model = load_model()

    def decide(self, candles: list):
        # ====== SYNC REAL STATE ======
        state = self.om.state.sync()

        obs = candles_to_features(candles, state)
        action, _ = self.model.predict(obs, deterministic=False)

        action = int(action)

        # ====== GUARD FINAL ======
        if action == 1 and not self.om.state.can_buy():
            return 0

        if action == 2 and not self.om.state.can_sell():
            return 0

        return action
