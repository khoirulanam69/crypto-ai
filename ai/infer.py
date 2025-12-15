# ai/infer.py
import os
import pandas as pd
from stable_baselines3 import PPO
from ai.envs.exchange_env import ExchangeEnv
from executor.order_manager import OrderManager
from ai.experience_logger import ExperienceLogger

logger = ExperienceLogger()
MODEL = os.getenv("MODEL", "models/ppo_baseline.zip")
SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
WINDOW = int(os.getenv("AI_WINDOW", "50"))

_model_cache = None

def load_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = PPO.load(MODEL)
    return _model_cache

def predict_action_from_ohlcv(df_window):
    """
    df_window: pandas DataFrame with at least 'close' column, length == WINDOW
    returns: action int (0,1,2)
    """
    model = load_model()
    env = ExchangeEnv(df_window, window_size=len(df_window))
    obs = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    return int(action)

def decide_and_execute(order_manager, env, obs):
    """
    env: ExchangeEnv live instance
    obs: current observation
    """

    model = load_model()
    action, _ = model.predict(obs, deterministic=False)

    next_obs, reward, done, _, info = env.step(int(action))

    # EKSEKUSI REAL
    price = info["networth"]

    if action == 1:
        order_manager.create_market_buy()
    elif action == 2:
        order_manager.create_market_buy()
    elif action == 3:
        order_manager.create_market_sell()
    elif action == 4:
        order_manager.create_market_sell()

    # LOG EXPERIENCE
    logger.log(
        obs=obs,
        action=int(action),
        reward=float(reward),
        next_obs=next_obs,
        done=done
    )

    return next_obs, done, info
