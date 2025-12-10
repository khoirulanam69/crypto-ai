# scripts/run_realtime_ai.py
import os
import time
from dotenv import load_dotenv
from executor.order_manager import OrderManager
from ai.infer import decide_and_execute
import pandas as pd

load_dotenv()
SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
WINDOW = int(os.getenv("AI_WINDOW", "50"))
SLEEP = float(os.getenv("LOOP_INTERVAL", "10"))
LIVE_BUFFER = os.getenv("LIVE_BUFFER", "data/live_buffer.csv")

om = OrderManager()  # uses paper mode by default from env

def fetch_recent_ohlcv_df(symbol=SYMBOL, window=WINDOW):
    # Use OrderManager.fetch_ohlcv to get recent candles
    ohlcv = om.fetch_ohlcv(symbol, timeframe='1m', limit=window)
    # ohlcv is list of [ts, open, high, low, close, volume]
    import pandas as pd
    cols = ['timestamp','open','high','low','close','volume']
    df = pd.DataFrame(ohlcv, columns=cols)
    return df

print("Starting realtime AI loop (paper=%s)..." % os.getenv("MODE", "paper"))
while True:
    try:
        df_window = fetch_recent_ohlcv_df()
        if df_window is None or df_window.shape[0] < WINDOW:
            print("Not enough data, waiting...")
            time.sleep(SLEEP)
            continue

        res = decide_and_execute(om, df_window, live_buffer_path=LIVE_BUFFER)
        print("AI decided:", res.get("action"), "note:", res.get("note"))
    except KeyboardInterrupt:
        print("Stopping by user")
        break
    except Exception as e:
        print("Loop error:", e)
    time.sleep(SLEEP)
