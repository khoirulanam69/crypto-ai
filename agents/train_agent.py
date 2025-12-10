# Minimal example using stable-baselines3 (PPO) with the SimpleTradingEnv
import numpy as np
from stable_baselines3 import PPO
from envs.gym_env import SimpleTradingEnv

def load_prices_from_csv(path='data/btc_usdt_1h.csv'):
    import pandas as pd
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df['close'].values

def train():
    prices = load_prices_from_csv()
    env = SimpleTradingEnv(prices)
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save('models/ppo_simple')

if __name__ == '__main__':
    train()
