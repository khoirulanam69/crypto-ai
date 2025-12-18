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

# load env
load_dotenv()

# imports from your project
# adjust import path if different (e.g., executor.order_manager)
from executor.order_manager import OrderManager  # your existing patched file
from utils.proxy_manager import ProxyManager
from ai.ensemble.aggregator import EnsembleAggregator
from ai.ensemble.ppo_trend import PPOTrend
from ai.ensemble.ppo_mean import PPOMean
from ai.ensemble.lstm import LSTMPrice
from ai.ensemble.rule import RuleBased
from executor.position_tracker import PositionTracker
from ai.memory.replay_buffer import ReplayBuffer
from ai.memory.online_trainer import fine_tune
from risk.risk_engine import RiskEngine

tracker = PositionTracker()
replay = ReplayBuffer()
risk_engine = RiskEngine(max_dd=0.20)

# settings
SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
SLEEP_SECONDS = float(os.getenv("LOOP_INTERVAL", "15"))   # how often to check price
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
    last_equity = None
    step_counter = 0
    safe_print("Starting realtime bot (paper=%s)..." % PAPER)
    # create proxy manager (it auto-loads PROXIES from env if set)
    proxy_mgr = ProxyManager()
    # create order manager (this should use resolver/proxy inside)
    om = OrderManager()
    # =========================
    # INIT AI ENSEMBLE (ONE TIME)
    # =========================
    ensemble = EnsembleAggregator([
        PPOTrend(),
        PPOMean(),
        LSTMPrice(),
        RuleBased()
    ])

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

            # =========================
            # AI ENSEMBLE DECISION
            # =========================

            # ambil OHLCV window untuk AI
            WINDOW = int(os.getenv("AI_WINDOW", "50"))

            ohlcv = om._safe_request(
                om.fetch_ohlcv, SYMBOL, '1m', WINDOW
            ) if hasattr(om, "_safe_request") else om.fetch_ohlcv(SYMBOL, '1m', WINDOW)

            # build state (samakan dengan ExchangeEnv)
            closes = [c[4] for c in ohlcv]

            import numpy as np
            window = np.array(closes, dtype=float)
            mean = window.mean() if window.mean() != 0 else 1.0
            window_norm = window / mean

            loop_count = 0
            if loop_count % 5 == 0 or 'balance' not in locals():
                balance = om.exchange.fetch_balance()
                loop_count += 1

            usdt = balance['free'].get('USDT', 0)
            btc = balance['free'].get('BTC', 0)

            portfolio_value = usdt + btc * price + 1e-9
            cash_ratio = usdt / portfolio_value
            pos_ratio = btc

            state = np.concatenate([window_norm, [cash_ratio, pos_ratio]]).astype(np.float32)

            # AI decision
            raw_action = ensemble.decide(state)

            # =========================
            # RISK MANAGEMENT GATE
            # =========================

            equity = om.get_equity(SYMBOL)

            risk_decision = risk_engine.evaluate(
                signal=raw_action,
                candles=ohlcv,
                equity=equity,
                price=price,
                has_position=tracker.has_position()
            )

            # =========================
            # MEMORY REWARD
            # =========================
            reward = 0.0
            if last_equity is not None:
                reward = equity - last_equity

            action = risk_decision["action"]

            safe_print(f"[RISK] Decision → {action}")

            if action == "BUY":
                size = risk_decision["size"]

                if PAPER:
                    exec_price = price
                    exec_amount = size
                    safe_print(f"[PAPER BUY] {exec_amount} @ {exec_price}")
                else:
                    res = om.market_buy(SYMBOL, size)
                    exec_price = res.get("price", price)
                    exec_amount = res.get("amount", size)

                tracker.update_from_trade({
                    "side": "buy",
                    "amount": exec_amount,
                    "price": exec_price
                })

                log_trade(conn, "BUY", SYMBOL, exec_amount, exec_price,
                        "paper" if PAPER else "live", note="risk_engine")

            elif action == "SELL":
                size = tracker.position_size()

                if size <= 0:
                    safe_print("[RISK] SELL ignored — no position")
                else:
                    if PAPER:
                        exec_price = price
                        exec_amount = size
                        safe_print(f"[PAPER SELL] {exec_amount} @ {exec_price}")
                    else:
                        res = om.market_sell(SYMBOL, size)
                        exec_price = res.get("price", price)
                        exec_amount = res.get("amount", size)

                    tracker.update_from_trade({
                        "side": "sell",
                        "amount": exec_amount,
                        "price": exec_price
                    })
                    
                    risk_engine.on_position_closed()

                    log_trade(conn, "SELL", SYMBOL, exec_amount, exec_price,
                            "paper" if PAPER else "live", note="risk_engine")

            else:
                safe_print("[RISK] HOLD")

            safe_print(f"Equity: {equity:.2f}", "| Cash:", round(om.state.state["cash"], 2), "| Position:", round(om.state.state["position"], 6))
            safe_print(f"Reward: {reward:.2f}", f"| Portfolio Value: {portfolio_value:.2f}")

            # =========================
            # SAVE EXPERIENCE
            # =========================
            replay.append(
                price=price,
                action=int(raw_action),
                reward=float(reward),
                equity=float(equity)
            )

            last_equity = equity
            step_counter += 1

            # =========================
            # D5 - ONLINE FINE TUNE
            # =========================
            FINE_TUNE_EVERY = int(os.getenv("FINE_TUNE_EVERY", "50"))

            if step_counter % FINE_TUNE_EVERY == 0:
                safe_print("[AI] Online learning triggered...")
                try:
                    fine_tune()
                except Exception as e:
                    safe_print("[AI] Fine-tune failed:", e)

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
