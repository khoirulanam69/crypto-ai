# ai/evaluate.py
import os
import pandas as pd
from stable_baselines3 import PPO
from ai.envs.exchange_env import ExchangeEnv

MODEL = os.getenv("MODEL", "models/ppo_baseline.zip")
TEST_CSV = os.getenv("TEST_CSV", "data/history_BTCUSDT_1h_test.csv")

def run_episode(env, model):
    obs = env.reset()
    done = False
    total_reward = 0
    trades = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(int(action))
        total_reward += reward
        if 'trade' in info:
            trades.append(info['trade'])
    return total_reward, trades, env.networth_history

def main():
    df = pd.read_csv(TEST_CSV)
    env = ExchangeEnv(df, window_size=50)
    model = PPO.load(MODEL)
    reward, trades, history = run_episode(env, model)
    print("Total reward:", reward)
    print("Trades:", len(trades))
    print("Final networth:", history[-1] if history else None)

if __name__ == "__main__":
    main()
