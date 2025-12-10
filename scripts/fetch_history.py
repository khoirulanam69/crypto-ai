import ccxt
import pandas as pd
import time
import os

PAIR = "BTC/USDT"
TIMEFRAME = "1h"
LIMIT = 1000

exchange = ccxt.binance({
    'enableRateLimit': True
})

os.makedirs("data", exist_ok=True)

def fetch_ohlcv():
    all_data = []
    since = None

    while True:
        try:
            print("Fetching...")
            candles = exchange.fetch_ohlcv(PAIR, TIMEFRAME, since=since, limit=LIMIT)

            if not candles:
                break

            all_data.extend(candles)
            since = candles[-1][0] + 1 * 60 * 60 * 1000  # next 1h
            time.sleep(exchange.rateLimit / 1000)

        except Exception as e:
            print("Error:", e)
            break

    df = pd.DataFrame(all_data, columns=[
        "timestamp","open","high","low","close","volume"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    file_path = f"data/history_BTCUSDT_1h.csv"
    df.to_csv(file_path, index=False)

    print("Saved:", file_path)


if __name__ == "__main__":
    fetch_ohlcv()
