# ai/ensemble/ppo_mean.py
import os
import numpy as np
from stable_baselines3 import PPO

class PPOMean:
    def __init__(self):
        model_path = os.getenv("PPO_MEAN_MODEL")
        if not model_path:
            raise ValueError("PPO_MEAN_MODEL env not set")

        self.model = PPO.load(model_path)

    def predict(self, state):
        state = np.array(state, dtype=np.float32).reshape(1, -1)
        action, _ = self.model.predict(state, deterministic=True)
        confidence = 0.55
        return int(action), confidence
