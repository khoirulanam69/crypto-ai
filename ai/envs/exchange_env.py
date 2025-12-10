import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class ExchangeEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, df, window_size=50, initial_cash=10000.0):

        super().__init__()

        # Pastikan cukup data
        if len(df) <= window_size + 1:
            raise ValueError(
                f"Data terlalu sedikit! Butuh minimal {window_size+2} baris, "
                f"tapi hanya ada {len(df)}."
            )

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_cash = initial_cash

        # State
        self.reset()

        # Action space: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)

        # Observation space: window_size harga closing + cash_ratio + position_ratio
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size + 2,), dtype=np.float32
        )

    # ===================================================
    # RESET
    # ===================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.t = self.window_size  # mulai dari window_size
        self.cash = self.initial_cash
        self.position = 0.0  # jumlah BTC/asset yg dipegang

        return self._get_obs(), {}

    # ===================================================
    # AMBIL HARGA
    # ===================================================
    def _price(self):
        return float(self.df.loc[self.t, "close"])

    # ===================================================
    # OBSERVASI
    # ===================================================
    def _get_obs(self):
        window = self.df["close"].iloc[self.t - self.window_size:self.t].values

        # Normalisasi sederhana
        mean = window.mean() if window.mean() != 0 else 1.0
        window_norm = window / mean

        cash_ratio = self.cash / (self.cash + (self.position * self._price()) + 1e-9)
        pos_ratio = self.position

        obs = np.concatenate([window_norm, [cash_ratio, pos_ratio]]).astype(np.float32)
        return obs

    # ===================================================
    # STEP
    # ===================================================
    def step(self, action):

        price = self._price()

        # BUY
        if action == 1:
            amount = self.cash / price
            self.position += amount
            self.cash = 0

        # SELL
        elif action == 2 and self.position > 0:
            self.cash += self.position * price
            self.position = 0

        # NEXT STEP
        self.t += 1
        terminated = (self.t >= len(self.df) - 1)

        obs = self._get_obs()
        reward = self.cash + self.position * price
        info = {"portfolio_value": reward}

        return obs, reward, terminated, False, info
