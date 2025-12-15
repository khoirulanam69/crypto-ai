# ai/train.py
import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from ai.envs.exchange_env import ExchangeEnv

DATA_CSV = os.getenv("HIST_CSV", "data/history_BTCUSDT_1h.csv")
MODEL_OUT = os.getenv("MODEL_OUT", "models/ppo_live.zip")
TIMESTEPS = int(os.getenv("TRAIN_TIMESTEPS", "200000"))

def load_df(path):
    df = pd.read_csv(path)
    # Expect columns: timestamp, open, high, low, close, volume
    return df

def make_env(df):
    def _init():
        return ExchangeEnv(df, window_size=50, initial_cash=10000.0)
    return _init

def main():
    os.makedirs(os.path.dirname(MODEL_OUT) or ".", exist_ok=True)
    df = load_df(DATA_CSV)
    env_fns = [make_env(df) for _ in range(4)]
    vec = DummyVecEnv(env_fns)
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False)

    model = PPO("MlpPolicy", vec, verbose=1, tensorboard_log="./tb/ppo")
    model.learn(total_timesteps=TIMESTEPS)
    model.save(MODEL_OUT)
    vec.save(MODEL_OUT + ".vec.pkl")
    print("Saved model:", MODEL_OUT)

if __name__ == "__main__":
    main()
