from stable_baselines3 import PPO
from .base import BaseModel
import os

class PPOTrend(BaseModel):
    name = "ppo_trend"
    weight = 1.2

    def __init__(self):
        self.model = PPO.load(os.getenv("PPO_TREND_MODEL"))

    def predict(self, state):
        action, _ = self.model.predict(state, deterministic=False)
        confidence = 0.6
        return int(action), confidence
