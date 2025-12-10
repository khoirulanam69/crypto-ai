# Simple paper-trade runner that uses trained model if exists, otherwise random actions.
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from agents.train_agent import load_prices_from_csv
from envs.gym_env import SimpleTradingEnv

def run():
    prices = load_prices_from_csv()
    env = SimpleTradingEnv(prices)
    obs = env.reset()
    done = False
    while not done:
        action = np.random.choice([0,1,2])
        obs, reward, done, info = env.step(action)
    print('Paper-trade finished. Portfolio value:', info.get('portfolio_value'))

if __name__ == '__main__':
    run()
