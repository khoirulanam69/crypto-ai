import os
import numpy as np
from stable_baselines3 import PPO
from ai.experience_logger import ExperienceLogger


class AIDecisionEngine:
    def __init__(
        self,
        symbol: str,
        model_path: str | None = None,
        experience_path: str | None = None,
    ):
        """
        AI Decision Engine (PPO-based)

        symbol           : trading pair (ex: BTC/USDT)
        model_path       : path model PPO
        experience_path  : path experience log
        """

        self.symbol = symbol

        self.model_path = model_path or os.getenv(
            "MODEL", "models/ppo_live.zip"
        )

        self.experience_path = experience_path or os.getenv(
            "EXPERIENCE_PATH", f"history/experience_{symbol.replace('/', '_')}.csv"
        )

        self.logger = ExperienceLogger(self.experience_path)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model PPO tidak ditemukan: {self.model_path}"
            )

        self.model = PPO.load(self.model_path)

        print(f"[AI] PPO model loaded for {self.symbol}")
        print(f"[AI] Experience log: {self.experience_path}")

    # ======================================================
    # PREDICTION
    # ======================================================

    def predict(self, obs, deterministic=False) -> int:
        action, _ = self.model.predict(
            obs, deterministic=deterministic
        )
        return int(action)

    # ======================================================
    # LIVE STEP + EXPERIENCE LOGGING
    # ======================================================

    def step_and_learn(self, env, obs):
        """
        env : ExchangeEnv (LIVE)
        obs : current observation
        """

        action = self.predict(obs, deterministic=False)

        next_obs, reward, done, _, info = env.step(action)

        # Log experience (untuk training selanjutnya)
        self.logger.log(
            symbol=self.symbol,
            obs=obs,
            action=action,
            reward=float(reward),
            next_obs=next_obs,
            done=done,
        )

        return action, next_obs, reward, done, info
