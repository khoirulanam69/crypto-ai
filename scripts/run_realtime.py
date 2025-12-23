# scripts/run_realtime.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import time
import signal
import sqlite3
import traceback
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# load env
load_dotenv()

from executor.order_manager import OrderManager
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
from utils.logger import setup_logger, MetricsCollector

tracker = PositionTracker()
replay = ReplayBuffer()
risk_engine = RiskEngine(max_dd=0.20)

# settings
SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
SLEEP_SECONDS = float(os.getenv("LOOP_INTERVAL", "15"))   # how often to check price
PAPER = os.getenv("MODE", "paper").lower() == "paper"
DB_PATH = os.getenv("TRADES_DB", "trades.db")
MAX_ERRORS_BEFORE_RESTART = int(os.getenv("MAX_ERRORS_BEFORE_RESTART", "10"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

stop_requested = False

def signal_handler(signum, frame):
    global stop_requested
    logger.info(f"Signal {signum} received — stopping gracefully...")
    stop_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Initialize logger
logger = setup_logger(__name__)
metrics = MetricsCollector()

# lightweight DB helper
def init_db(path=DB_PATH):
    conn = sqlite3.connect(path, timeout=10.0)
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
        note TEXT,
        equity REAL,
        reward REAL
    )
    """)
    conn.commit()
    return conn

def log_trade(conn, side, symbol, amount, price, mode, note="", equity=None, reward=None):
    ts = datetime.utcnow().isoformat()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO trades (ts, side, symbol, amount, price, mode, note, equity, reward) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (ts, side, symbol, amount, price, mode, note, equity, reward)
    )
    conn.commit()

def get_price_with_fallback(om, symbol, max_retries=3):
    """Get price with multiple fallback methods"""
    for attempt in range(1, max_retries + 1):
        try:
            # Method 1: get_ticker if available
            if hasattr(om, "get_ticker"):
                ticker = om._safe_request(om.get_ticker, symbol) if hasattr(om, "_safe_request") else om.get_ticker(symbol)
                if isinstance(ticker, dict) and "last" in ticker:
                    return float(ticker["last"])
            
            # Method 2: fetch_ticker directly
            ticker = om.exchange.fetch_ticker(symbol)
            return float(ticker["last"])
            
        except Exception as e:
            if attempt < max_retries:
                logger.warning(f"Price fetch error: {e}, retry {attempt}/{max_retries}")
                time.sleep(2 * attempt)
                continue
    
    # Final fallback: Use OHLCV
    try:
        ohlcv = om.exchange.fetch_ohlcv(symbol, '1m', 2)
        return float(ohlcv[-1][4])
    except:
        # Return last known price or 0
        logger.error("All price fetch methods failed")
        return 0.0

def main_loop():
    last_equity = None
    step_counter = 0
    logger.info(f"Starting realtime bot (paper={PAPER})...")
    
    # create proxy manager (it auto-loads PROXIES from env if set)
    proxy_mgr = ProxyManager()
    
    # create order manager (this should use resolver/proxy inside)
    om = OrderManager()

    # CONNECTION TEST
    logger.info("Testing exchange connection...")
    if not hasattr(om, 'test_connection'):
        logger.warning("test_connection method not available")
    else:
        connection_ok = om.test_connection()
        if not connection_ok:
            logger.error("Connection test failed. Retrying in 30 seconds...")
            time.sleep(30)
            # Optionally restart or exit
            return

    # =========================
    # INIT AI ENSEMBLE
    # =========================
    # Get ensemble configuration from environment
    ENABLE_PPO_TREND = os.getenv("ENABLE_PPO_TREND", "true").lower() == "true"
    ENABLE_PPO_MEAN = os.getenv("ENABLE_PPO_MEAN", "true").lower() == "true"
    ENABLE_LSTM = os.getenv("ENABLE_LSTM", "true").lower() == "true"
    ENABLE_RULE = os.getenv("ENABLE_RULE", "true").lower() == "true"
    
    ensemble_models = []
    if ENABLE_RULE:
        ensemble_models.append(RuleBased())
        logger.info("RuleBased model enabled")
    if ENABLE_PPO_TREND:
        ensemble_models.append(PPOTrend())
        logger.info("PPOTrend model enabled")
    if ENABLE_PPO_MEAN:
        ensemble_models.append(PPOMean())
        logger.info("PPOMean model enabled")
    if ENABLE_LSTM:
        ensemble_models.append(LSTMPrice())
        logger.info("LSTMPrice model enabled")
    
    if not ensemble_models:
        logger.error("No AI models enabled! Check your ENABLE_* environment variables.")
        return
    
    ensemble = EnsembleAggregator(ensemble_models)
    logger.info(f"Ensemble initialized with {len(ensemble_models)} models")

    # ensure safe_request helper exists on om
    if not hasattr(om, "_safe_request"):
        logger.warning("OrderManager._safe_request not found — some requests may not retry properly.")

    conn = init_db()

    error_count = 0
    balance = None

    while not stop_requested:
        try:
            # pick a working proxy and apply if needed (OrderManager may already do this)
            proxy = proxy_mgr.get_working_proxy()
            if proxy:
                logger.debug(f"Using proxy: {proxy}")
            else:
                logger.debug("No proxy (direct connection)")

            # fetch latest price via a safe call
            price = get_price_with_fallback(om, SYMBOL)
            logger.info(f"Current price: {price}")

            # =========================
            # AI ENSEMBLE DECISION
            # =========================

            # ambil OHLCV window untuk AI
            WINDOW = int(os.getenv("AI_WINDOW", "50"))

            try:
                ohlcv = om._safe_request(
                    om.fetch_ohlcv, SYMBOL, '1m', WINDOW
                ) if hasattr(om, "_safe_request") else om.fetch_ohlcv(SYMBOL, '1m', WINDOW)
            except Exception as e:
                logger.error(f"Failed to fetch OHLCV: {e}")
                time.sleep(SLEEP_SECONDS)
                continue

            # build state (samakan dengan ExchangeEnv)
            closes = [c[4] for c in ohlcv]

            window = np.array(closes, dtype=float)
            mean_val = window.mean()
            if mean_val == 0:
                window_norm = window.copy()
            else:
                window_norm = window / mean_val

            if step_counter % 5 == 0 or balance is None:
                try:
                    balance = om.exchange.fetch_balance()
                except Exception as e:
                    logger.error(f"Failed to fetch balance: {e}")
                    # Use last balance if available
                    if balance is None:
                        time.sleep(SLEEP_SECONDS)
                        continue

            usdt = balance['free'].get('USDT', 0) if balance else 0
            btc = balance['free'].get('BTC', 0) if balance else 0

            portfolio_value = usdt + btc * price + 1e-9
            if portfolio_value < 1e-9:
                cash_ratio = 0.0
            else:
                cash_ratio = usdt / portfolio_value
            pos_ratio = btc

            state = np.concatenate([window_norm, [cash_ratio, pos_ratio]]).astype(np.float32)

            # AI decision
            try:
                raw_action = ensemble.decide(state)
                logger.debug(f"Raw AI decision: {raw_action}")
            except Exception as e:
                logger.error(f"AI decision failed: {e}")
                raw_action = 1  # Default to HOLD on error

            # =========================
            # RISK MANAGEMENT GATE
            # =========================

            equity = om.get_equity(SYMBOL) if hasattr(om, 'get_equity') else portfolio_value

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

            logger.info(f"Risk Decision → {action}")

            if action == "BUY":
                size = risk_decision["size"]

                if PAPER:
                    exec_price = price
                    exec_amount = size
                    logger.info(f"[PAPER BUY] {exec_amount} @ {exec_price}")
                else:
                    try:
                        res = om.market_buy(SYMBOL, size)
                        exec_price = res.get("price", price)
                        exec_amount = res.get("amount", size)
                        logger.info(f"[LIVE BUY] {exec_amount} @ {exec_price}")
                    except Exception as e:
                        logger.error(f"Buy execution failed: {e}")
                        exec_price = price
                        exec_amount = 0

                if exec_amount > 0:
                    tracker.update_from_trade({
                        "side": "buy",
                        "amount": exec_amount,
                        "price": exec_price
                    })

                    log_trade(conn, "BUY", SYMBOL, exec_amount, exec_price,
                            "paper" if PAPER else "live", note="risk_engine", 
                            equity=equity, reward=reward)

            elif action == "SELL":
                size = tracker.position_size()

                if size <= 0:
                    logger.warning("SELL ignored — no position")
                else:
                    if PAPER:
                        exec_price = price
                        exec_amount = size
                        logger.info(f"[PAPER SELL] {exec_amount} @ {exec_price}")
                    else:
                        try:
                            res = om.market_sell(SYMBOL, size)
                            exec_price = res.get("price", price)
                            exec_amount = res.get("amount", size)
                            logger.info(f"[LIVE SELL] {exec_amount} @ {exec_price}")
                        except Exception as e:
                            logger.error(f"Sell execution failed: {e}")
                            exec_price = price
                            exec_amount = 0

                    if exec_amount > 0:
                        tracker.update_from_trade({
                            "side": "sell",
                            "amount": exec_amount,
                            "price": exec_price
                        })

                        risk_engine.on_position_closed()

                        log_trade(conn, "SELL", SYMBOL, exec_amount, exec_price,
                                "paper" if PAPER else "live", note="risk_engine",
                                equity=equity, reward=reward)

            else:
                logger.debug("HOLD - No action taken")

            # Log metrics
            cash = om.state.state["cash"] if hasattr(om, 'state') else usdt
            position = om.state.state["position"] if hasattr(om, 'state') else btc
            logger.info(f"Equity: {equity:.2f} | Cash: {cash:.2f} | Position: {position:.6f}")
            logger.info(f"Reward: {reward:.2f} | Portfolio Value: {portfolio_value:.2f}")
            
            # Collect metrics
            metrics.record_trade_step(
                price=price,
                action=action,
                equity=equity,
                reward=reward,
                portfolio_value=portfolio_value
            )

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
            # D5 – SAFE ONLINE LEARNING
            # =========================

            ENABLE_ONLINE = os.getenv("ENABLE_ONLINE_LEARNING", "false").lower() == "true"
            FINE_TUNE_EVERY = int(os.getenv("FINE_TUNE_EVERY", "500"))
            MIN_REPLAY_SIZE = int(os.getenv("MIN_REPLAY_SIZE", "200"))
            MAX_REPLAY_SIZE = int(os.getenv("MAX_REPLAY_SIZE", "10000"))
            MAX_DD_FOR_TRAIN = float(os.getenv("MAX_DD_FOR_TRAIN", "0.10"))

            if ENABLE_ONLINE:
                if step_counter % FINE_TUNE_EVERY == 0:
                    if replay.size() > MAX_REPLAY_SIZE:
                        replay.trim_oldest(MAX_REPLAY_SIZE // 2)
                        logger.info(f"Replay buffer trimmed to {MAX_REPLAY_SIZE // 2} samples")
                    elif replay.size() < MIN_REPLAY_SIZE:
                        logger.info("Skip fine-tune: replay buffer too small")
                    elif not risk_engine.allow_training(equity):
                        logger.warning("Skip fine-tune: drawdown too high")
                    else:
                        logger.info("Online learning triggered...")
                        try:
                            fine_tune()
                            logger.info("Fine-tune completed successfully")
                        except Exception as e:
                            logger.error(f"Fine-tune failed: {e}")

            # reset error counter & sleep
            error_count = 0
            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            error_count += 1
            logger.error(f"Loop error: {type(e).__name__}: {str(e)[:100]}")
            if DEBUG:
                logger.error(traceback.format_exc())
            if error_count >= MAX_ERRORS_BEFORE_RESTART:
                logger.error("Too many errors — exiting to allow supervisor to restart.")
                break
            # exponential backoff
            backoff_time = min(60, 5 * (2 ** (error_count - 1)))
            logger.info(f"Backing off for {backoff_time} seconds...")
            time.sleep(backoff_time)

    logger.info("Realtime bot stopping.")
    try:
        # Save final metrics
        metrics.save_summary()
        conn.close()
    finally:
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    main_loop()