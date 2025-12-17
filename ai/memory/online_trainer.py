# ai/memory/online_trainer.py
import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from ai.envs.exchange_env import ExchangeEnv

MODEL = os.getenv("MODEL", "models/ppo_live.zip")
BUFFER = os.getenv("REPLAY_BUFFER", "data/replay_buffer.csv")
FINE_TUNE_STEPS = int(os.getenv("FINE_TUNE_STEPS", "2048"))

def build_env_from_memory(hist_df, buffer_df):
    """
    Gabungkan historical OHLCV + replay buffer
    """
    rows = []

    last = hist_df.iloc[-1].copy()

    for _, r in buffer_df.iterrows():
        new = last.copy()
        new["timestamp"] = int(r["ts"])
        new["open"] = r["price"]
        new["high"] = r["price"]
        new["low"] = r["price"]
        new["close"] = r["price"]
        new["volume"] = last["volume"]
        rows.append(new)

    df = pd.concat([hist_df, pd.DataFrame(rows)], ignore_index=True)
    return DummyVecEnv([lambda: ExchangeEnv(df, window_size=50)])

def fine_tune(hist_csv="data/history.csv"):
    if not os.path.exists(BUFFER):
        return

    buffer_df = pd.read_csv(BUFFER)
    if len(buffer_df) < 100:
        return

    hist_df = pd.read_csv(hist_csv)

    env = build_env_from_memory(hist_df, buffer_df)
    model = PPO.load(MODEL)
    model.set_env(env)

    model.learn(total_timesteps=FINE_TUNE_STEPS)
    model.save(MODEL)

    print("[AI] Online fine-tune completed")
