import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SimpleTradingEnv(gym.Env):
    """Very small example env for learning and testing."""
    metadata = {'render.modes': ['human']}
    def __init__(self, prices):
        super().__init__()
        self.prices = prices
        self.current = 0
        self.position = 0  # 0=no, 1=long
        self.cash = 10000.0
        self.asset = 0.0
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def _get_obs(self):
        price = self.prices[self.current]
        sma_short = np.mean(self.prices[max(0, self.current-5):self.current+1])
        sma_long = np.mean(self.prices[max(0, self.current-20):self.current+1])
        return np.array([price, sma_short, sma_long, self.position], dtype=np.float32)

    def step(self, action):
        done = False
        price = self.prices[self.current]
        # execute action
        if action == 1 and self.position == 0:
            self.asset = self.cash / price
            self.cash = 0.0
            self.position = 1
        elif action == 2 and self.position == 1:
            self.cash = self.asset * price
            self.asset = 0.0
            self.position = 0
        self.current += 1
        if self.current >= len(self.prices)-1:
            done = True
        port_value = self.cash + self.asset * self.prices[self.current]
        reward = port_value
        obs = self._get_obs()
        info = {'portfolio_value': port_value}
        return obs, reward, done, info

    def reset(self):
        self.current = 0
        self.position = 0
        self.cash = 10000.0
        self.asset = 0.0
        return self._get_obs()

    def render(self, mode='human'):
        print(f"Step {self.current} | Position: {self.position} | Cash: {self.cash} | Asset: {self.asset}")
