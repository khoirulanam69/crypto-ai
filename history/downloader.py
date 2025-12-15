import ccxt
import pandas as pd
import os
from datetime import datetime

def fetch_ohlcv(symbol='BTC/USDT', timeframe='1h', limit=1000, since=None):
    exchange = ccxt.binance({'enableRateLimit': True})
    if since:
        since = exchange.parse8601(since)
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

if __name__ == '__main__':
    df = fetch_ohlcv('BTC/USDT','1h',limit=500)
    out = 'data/btc_usdt_1h.csv'
    os.makedirs('data', exist_ok=True)
    df.to_csv(out)
    print(f"Saved {len(df)} rows to {out}")
