import numpy as np
import gymnasium as gym
from gymnasium import spaces

TRADING_FEE = 0.001  # 0.1%

class ExchangeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, window_size=50, initial_cash=10000.0):
        super().__init__()

        if len(df) <= window_size + 1:
            raise ValueError("Not enough data")

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_cash = initial_cash

        # ACTIONS
        # 0 hold
        # 1 buy 25%
        # 2 buy 50%
        # 3 sell 25%
        # 4 sell 50%
        self.action_space = spaces.Discrete(5)

        # OBS: price window + cash ratio + position ratio
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size + 2,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.t = self.window_size
        self.cash = self.initial_cash
        self.position = 0.0
        self.prev_networth = self.initial_cash

        return self._get_obs(), {}

    def _price(self):
        return float(self.df.loc[self.t, "close"])

    def _networth(self, price):
        return self.cash + self.position * price

    def _get_obs(self):
        window = self.df["close"].iloc[self.t - self.window_size:self.t].values
        mean = window.mean() if window.mean() != 0 else 1.0
        window = window / mean

        networth = self._networth(self._price())
        cash_ratio = self.cash / (networth + 1e-9)
        pos_ratio = self.position

        return np.concatenate(
            [window, [cash_ratio, pos_ratio]]
        ).astype(np.float32)

    def step(self, action):
        price = self._price()
        trade_value = 0.0

        # BUY
        if action in (1, 2):
            pct = 0.25 if action == 1 else 0.5
            trade_value = self.cash * pct
            if trade_value > 0:
                qty = (trade_value * (1 - TRADING_FEE)) / price
                self.cash -= trade_value
                self.position += qty

        # SELL
        elif action in (3, 4):
            pct = 0.25 if action == 3 else 0.5
            qty = self.position * pct
            if qty > 0:
                proceeds = qty * price * (1 - TRADING_FEE)
                self.position -= qty
                self.cash += proceeds
                trade_value = proceeds

        self.t += 1
        terminated = self.t >= len(self.df) - 1

        networth = self._networth(price)

        # ===== REWARD ENGINE =====
        reward = networth - self.prev_networth

        # risk penalty (overtrading)
        if trade_value > 0:
            reward -= trade_value * 0.0005

        self.prev_networth = networth

        info = {
            "networth": networth,
            "cash": self.cash,
            "position": self.position
        }

        return self._get_obs(), reward, terminated, False, info
