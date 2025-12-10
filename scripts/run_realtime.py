# scripts/run_realtime.py
import os
import time
import signal
import sqlite3
import traceback
from datetime import datetime
from dotenv import load_dotenv
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from executor.order_manager import OrderManager


# load env
load_dotenv()

# imports from your project
# adjust import path if different (e.g., executor.order_manager)
from executor.order_manager import OrderManager  # your existing patched file
from utils.proxy_manager import ProxyManager

# settings
SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
SLEEP_SECONDS = float(os.getenv("LOOP_INTERVAL", "10"))   # how often to check price
PAPER = os.getenv("MODE", "paper").lower() == "paper"
DB_PATH = os.getenv("TRADES_DB", "trades.db")
MAX_ERRORS_BEFORE_RESTART = int(os.getenv("MAX_ERRORS_BEFORE_RESTART", "10"))

stop_requested = False

def signal_handler(signum, frame):
    global stop_requested
    print(f"[run_realtime] Signal {signum} received — stopping gracefully...")
    stop_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# lightweight DB helper
def init_db(path=DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT,
        side TEXT,
        symbol TEXT,
        amount REAL,
        price REAL,
        mode TEXT,
        note TEXT
    )
    """)
    conn.commit()
    return conn

def log_trade(conn, side, symbol, amount, price, mode, note=""):
    ts = datetime.utcnow().isoformat()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO trades (ts, side, symbol, amount, price, mode, note) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (ts, side, symbol, amount, price, mode, note)
    )
    conn.commit()

def safe_print(*a, **k):
    print(f"[{datetime.now().isoformat()}]", *a, **k)

def main_loop():
    safe_print("Starting realtime bot (paper=%s)..." % PAPER)
    # create proxy manager (it auto-loads PROXIES from env if set)
    proxy_mgr = ProxyManager()
    # create order manager (this should use resolver/proxy inside)
    om = OrderManager()
    # ensure safe_request helper exists on om
    if not hasattr(om, "_safe_request"):
        safe_print("Warning: OrderManager._safe_request not found — some requests may not retry properly.")

    conn = init_db()

    error_count = 0

    while not stop_requested:
        try:
            # pick a working proxy and apply if needed (OrderManager may already do this)
            proxy = proxy_mgr.get_working_proxy()
            if proxy:
                safe_print("[proxy] using", proxy)
            else:
                safe_print("[proxy] none (direct)")

            # fetch latest price via a safe call
            # prefer using order manager or exchange wrapper that returns ticker
            if hasattr(om, "get_ticker"):
                ticker = om._safe_request(om.get_ticker, SYMBOL) if hasattr(om, "_safe_request") else om.get_ticker(SYMBOL)
                price = float(ticker.get("last") if isinstance(ticker, dict) and "last" in ticker else ticker)
            else:
                # try fetch_ohlcv fallback: use latest close
                ohlcv = om._safe_request(om.fetch_ohlcv, SYMBOL, '1m', 2) if hasattr(om, "_safe_request") else om.fetch_ohlcv(SYMBOL, '1m', 2)
                # ohlcv: list of lists
                price = float(ohlcv[-1][4])

            safe_print("Price:", price)

            # DECISION LOGIC (simple example - replace with your AI/strategy)
            # You can import your existing strategy and call it instead
            # example: from strategy.my_strategy import decide; signal = decide(recent_prices)
            # For demo: trivial example - alternate buy/sell every N loops (not financial advice)
            # Replace this with your real strategy / AI call
            signal = None
            # simple threshold demo (placeholder)
            # load last N closes
            if hasattr(om, "fetch_ohlcv"):
                candles = om._safe_request(om.fetch_ohlcv, SYMBOL, '1m', 20) if hasattr(om, "_safe_request") else om.fetch_ohlcv(SYMBOL, '1m', 20)
                closes = [c[4] for c in candles]
                # naive moving average signal
                if len(closes) >= 10:
                    sma_short = sum(closes[-5:]) / 5
                    sma_long = sum(closes[-10:]) / 10
                    if sma_short > sma_long:
                        signal = "BUY"
                    elif sma_short < sma_long:
                        signal = "SELL"
                    else:
                        signal = "HOLD"
                else:
                    signal = "HOLD"
            else:
                signal = "HOLD"

            safe_print("Signal:", signal)

            # Execute signal
            if signal == "BUY":
                amount = float(os.getenv("TRADE_AMOUNT_QUOTE", "10"))  # buy for $10
                # order_manager should accept amount in quote or base depending on implementation
                if hasattr(om, "market_buy"):
                    res = om._safe_request(om.market_buy, SYMBOL, amount) if hasattr(om, "_safe_request") else om.market_buy(SYMBOL, amount)
                    safe_print("Buy order result:", res)
                    # try to extract executed price & amount
                    try:
                        exec_price = float(res.get("price") or res.get("avgPrice") or price)
                        exec_amount = float(res.get("filledQty") or res.get("amount") or (amount / exec_price))
                    except Exception:
                        exec_price = price
                        exec_amount = 0.0
                    log_trade(conn, "BUY", SYMBOL, exec_amount, exec_price, "paper" if PAPER else "live", note=str(res))
                else:
                    safe_print("OrderManager has no market_buy method; skipping execution.")

            elif signal == "SELL":
                # attempt to sell a fixed base amount or compute from holdings
                amount_base = float(os.getenv("TRADE_AMOUNT_BASE", "0.0005"))
                if hasattr(om, "market_sell"):
                    res = om._safe_request(om.market_sell, SYMBOL, amount_base) if hasattr(om, "_safe_request") else om.market_sell(SYMBOL, amount_base)
                    safe_print("Sell order result:", res)
                    try:
                        exec_price = float(res.get("price") or res.get("avgPrice") or price)
                        exec_amount = float(res.get("filledQty") or res.get("amount") or amount_base)
                    except Exception:
                        exec_price = price
                        exec_amount = amount_base
                    log_trade(conn, "SELL", SYMBOL, exec_amount, exec_price, "paper" if PAPER else "live", note=str(res))
                else:
                    safe_print("OrderManager has no market_sell method; skipping execution.")
            else:
                safe_print("No action.")

            error_count = 0
            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            error_count += 1
            safe_print("Loop error:", str(e))
            traceback.print_exc()
            if error_count >= MAX_ERRORS_BEFORE_RESTART:
                safe_print("Too many errors — exiting to allow supervisor to restart.")
                break
            # small backoff
            time.sleep(min(60, 5 * error_count))

    safe_print("Realtime bot stopping.")
    try:
        conn.close()
    except Exception:
        pass

if __name__ == "__main__":
    main_loop()
