# ai/infer.py
import os
import pandas as pd
from stable_baselines3 import PPO
from ai.envs.exchange_env import ExchangeEnv
from executor.order_manager import OrderManager

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

def decide_and_execute(order_manager: OrderManager, recent_ohlcv_df: pd.DataFrame, live_buffer_path: str = "data/live_buffer.csv"):
    """
    recent_ohlcv_df: dataframe with last WINDOW rows (timestamp, open, high, low, close, volume)
    Executes order via order_manager based on action prediction.
    Also append experience to live_buffer (for online fine-tune).
    """
    action = predict_action_from_ohlcv(recent_ohlcv_df)
    price = float(recent_ohlcv_df.iloc[-1]['close'])
    note = {}
    result = None

    if action == 1:
        qty_quote = float(os.getenv("TRADE_AMOUNT_QUOTE", "10"))
        result = order_manager.create_market_buy(SYMBOL, qty_quote)
        note = {"action": "buy", "quote": qty_quote}
    elif action == 2:
        qty_base = float(os.getenv("TRADE_AMOUNT_BASE", "0.0005"))
        result = order_manager.create_market_sell(SYMBOL, qty_base)
        note = {"action": "sell", "base": qty_base}
    else:
        note = {"action": "hold"}

    # Append to live buffer: timestamp, action, price, note (simple CSV)
    try:
        import csv, time
        ts = int(time.time() * 1000)
        row = [ts, action, price, str(note)]
        os.makedirs(os.path.dirname(live_buffer_path) or ".", exist_ok=True)
        write_header = not os.path.exists(live_buffer_path)
        with open(live_buffer_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["ts", "action", "price", "meta"])
            writer.writerow(row)
    except Exception as e:
        print("Append live buffer failed:", e)

    return {"action": action, "result": result, "note": note}
