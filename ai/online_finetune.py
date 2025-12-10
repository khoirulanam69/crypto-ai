# ai/online_finetune.py
import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from ai.envs.exchange_env import ExchangeEnv

MODEL = os.getenv("MODEL", "models/ppo_baseline.zip")
LIVE_BUFFER = os.getenv("LIVE_BUFFER", "data/live_buffer.csv")
FINE_TUNE_STEPS = int(os.getenv("FINE_TUNE_STEPS", "2048"))

def build_env_from_buffer(buffer_csv, hist_csv="data/history_BTCUSDT_1h.csv"):
    """
    This naive approach appends buffer rows to historical data tail for a mini env.
    For production: better replay buffer + prioritized sampling.
    """
    if not os.path.exists(buffer_csv):
        raise FileNotFoundError(buffer_csv)
    df_hist = pd.read_csv(hist_csv)
    df_live = pd.read_csv(buffer_csv)
    # create synthetic ohlcv rows from buffer 'price' by duplicating last known open/high/low/volume pattern
    last_row = df_hist.iloc[-1].copy()
    rows = []
    for _, r in df_live.iterrows():
        new = last_row.copy()
        new['timestamp'] = int(r['ts'])
        new['open'] = r['price']
        new['high'] = r['price']
        new['low'] = r['price']
        new['close'] = r['price']
        new['volume'] = last_row['volume']
        rows.append(new)
    df_aug = pd.concat([df_hist, pd.DataFrame(rows)], ignore_index=True)
    env = DummyVecEnv([lambda: ExchangeEnv(df_aug, window_size=50)])
    return env

def main():
    model = PPO.load(MODEL)
    env = build_env_from_buffer(LIVE_BUFFER)
    model.set_env(env)
    model.learn(total_timesteps=FINE_TUNE_STEPS)
    model.save(MODEL)
    print("Fine-tune done and model saved.")

if __name__ == "__main__":
    main()
