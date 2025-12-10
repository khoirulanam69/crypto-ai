# Simple backtest runner that replays prices through the env and a random policy
import numpy as np
from envs.gym_env import SimpleTradingEnv

def run_backtest(prices):
    env = SimpleTradingEnv(prices)
    obs = env.reset()
    done = False
    while not done:
        action = np.random.choice([0,1,2])
        obs, reward, done, info = env.step(action)
    print('Backtest finished. Portfolio value:', info.get('portfolio_value'))

if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('data/btc_usdt_1h.csv', index_col=0, parse_dates=True)
    run_backtest(df['close'].values)
