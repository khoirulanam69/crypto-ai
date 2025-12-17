# ai/ensemble/ppo_trend.py
import os
import numpy as np
from stable_baselines3 import PPO

class PPOTrend:
    def __init__(self):
        model_path = os.getenv("PPO_TREND_MODEL")
        if not model_path:
            raise ValueError("PPO_TREND_MODEL env not set")

        self.model = PPO.load(model_path)

    def predict(self, state):
        """
        state: np.ndarray shape (window+2,)
        return: (action, confidence)
        """
        state = np.array(state, dtype=np.float32).reshape(1, -1)

        action, _ = self.model.predict(state, deterministic=True)

        # Confidence sederhana (nanti bisa pakai entropy / logits)
        confidence = 0.6

        return int(action), confidence
