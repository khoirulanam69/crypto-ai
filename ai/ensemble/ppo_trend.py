import os
from stable_baselines3 import PPO

class PPOTrend:
    def __init__(self):
        model_path = os.getenv(
            "PPO_TREND_MODEL",
            "models/ppo_trend.zip"   # DEFAULT WAJIB
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"PPOTrend model not found: {model_path}"
            )

        self.model = PPO.load(model_path)

    def decide(self, state):
        action, _ = self.model.predict(state, deterministic=True)
        return int(action)
