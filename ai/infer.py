import os
import json
import numpy as np
from stable_baselines3 import PPO
from ai.experience_logger import ExperienceLogger

class AIDecisionEngine:
    def __init__(self, model_path=None, experience_path="data/experience.csv"):
        self.model_path = model_path or os.getenv(
            "MODEL", "models/ppo_live.zip"
        )
        self.logger = ExperienceLogger(experience_path)
        self.model = PPO.load(self.model_path)

    def predict(self, obs, deterministic=False):
        action, _ = self.model.predict(
            obs, deterministic=deterministic
        )
        return int(action)

    def step_and_learn(self, env, obs):
        """
        env : ExchangeEnv (LIVE)
        obs : current observation
        """

        action = self.predict(obs, deterministic=False)

        next_obs, reward, done, _, info = env.step(action)

        # log experience
        self.logger.log(
            obs=obs,
            action=action,
            reward=float(reward),
            next_obs=next_obs,
            done=done
        )

        return action, next_obs, reward, done, info
