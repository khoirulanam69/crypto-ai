import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from ai.envs.exchange_env import ExchangeEnv

DATA = "data/history_BTCUSDT_1h.csv"
MODEL_OUT = "models/ppo_trend.zip"

df = pd.read_csv(DATA)

def make_env():
    return ExchangeEnv(df, window_size=50, initial_cash=10000)

env = DummyVecEnv([make_env])

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=256,
    batch_size=64,
    learning_rate=3e-4,
)

model.learn(total_timesteps=10_000)  # CEPAT (~1 menit)
model.save(MODEL_OUT)

print("Saved:", MODEL_OUT)
