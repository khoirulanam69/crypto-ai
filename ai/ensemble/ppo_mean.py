from stable_baselines3 import PPO
from .base import BaseModel
import os
import numpy as np

class PPOMean(BaseModel):
    name = "ppo_mean"
    weight = 1.0

    def __init__(self):
        self.model = PPO.load(os.getenv("PPO_MEAN_MODEL"))

    def predict(self, state):
        action, _ = self.model.predict(state, deterministic=True)

        # confidence lebih tinggi saat deviasi besar
        vol = np.std(state)
        confidence = min(1.0, vol * 10)

        return int(action), confidence
