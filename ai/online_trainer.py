import os
import pandas as pd
import json
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from ai.envs.exchange_env import ExchangeEnv

MODEL = "models/ppo_live.zip"
EXPERIENCE = "data/experience.csv"
MIN_SAMPLES = 500
STEPS = 2048

def build_env_from_experience(df):
    prices = [json.loads(x)[-3] for x in df["obs"]]
    fake_df = pd.DataFrame({"close": prices})
    return DummyVecEnv([lambda: ExchangeEnv(fake_df, window_size=50)])

def main():
    if not os.path.exists(EXPERIENCE):
        print("No experience yet")
        return

    df = pd.read_csv(EXPERIENCE)
    if len(df) < MIN_SAMPLES:
        print("Not enough samples:", len(df))
        return

    env = build_env_from_experience(df.tail(2000))
    model = PPO.load(MODEL)
    model.set_env(env)

    model.learn(total_timesteps=STEPS)
    model.save(MODEL)

    print("Online fine-tune complete")

if __name__ == "__main__":
    main()
