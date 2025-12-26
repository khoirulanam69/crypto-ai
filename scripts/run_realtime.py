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

def create_ai_model(model_class, model_name, **kwargs):
    """Helper function to create AI model with error handling"""
    try:
        model = model_class(**kwargs)
        logger.info(f"{model_name} model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load {model_name}: {e}")
        return None

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
        try:
            connection_ok = om.test_connection()
            if not connection_ok:
                logger.error("Connection test failed. Retrying in 30 seconds...")
                time.sleep(30)
                # Optionally restart or exit
                return
        except Exception as e:
            logger.error(f"Connection test error: {e}")

    # =========================
    # INIT AI ENSEMBLE
    # =========================
    # Get ensemble configuration from environment
    ENABLE_PPO_TREND = os.getenv("ENABLE_PPO_TREND", "false").lower() == "true"
    ENABLE_PPO_MEAN = os.getenv("ENABLE_PPO_MEAN", "false").lower() == "true"
    ENABLE_LSTM = os.getenv("ENABLE_LSTM", "false").lower() == "true"
    ENABLE_RULE = os.getenv("ENABLE_RULE", "true").lower() == "true"
    
    ensemble_models = []
    
    # Load RuleBased model (should always work)
    if ENABLE_RULE:
        try:
            rule_model = RuleBased()
            ensemble_models.append(rule_model)
            logger.info("RuleBased model enabled")
        except Exception as e:
            logger.error(f"Failed to load RuleBased model: {e}")
    
    # Load PPO Trend model
    if ENABLE_PPO_TREND:
        ppo_trend_model = create_ai_model(PPOTrend, "PPOTrend")
        if ppo_trend_model:
            ensemble_models.append(ppo_trend_model)
    
    # Load PPO Mean model
    if ENABLE_PPO_MEAN:
        ppo_mean_model = create_ai_model(PPOMean, "PPOMean")
        if ppo_mean_model:
            ensemble_models.append(ppo_mean_model)
    
    # Load LSTM model
    if ENABLE_LSTM:
        lstm_model = create_ai_model(LSTMPrice, "LSTMPrice")
        if lstm_model:
            ensemble_models.append(lstm_model)
    
    if not ensemble_models:
        logger.error("No AI models could be loaded! Check your ENABLE_* environment variables and model paths.")
        logger.error("At minimum, ensure RuleBased model can load or set ENABLE_RULE=true")
        return
    
    try:
        ensemble = EnsembleAggregator(ensemble_models)
        logger.info(f"Ensemble initialized with {len(ensemble_models)} models")
    except Exception as e:
        logger.error(f"Failed to initialize ensemble: {e}")
        return

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
            if price <= 0:
                logger.error(f"Invalid price received: {price}")
                time.sleep(SLEEP_SECONDS)
                continue
                
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
                if len(ohlcv) < WINDOW:
                    logger.warning(f"OHLCV data insufficient: {len(ohlcv)}/{WINDOW}")
                    time.sleep(SLEEP_SECONDS)
                    continue
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
                # Default to HOLD (1) on error
                raw_action = 1
                logger.info("Using default HOLD action due to AI error")

            # =========================
            # RISK MANAGEMENT GATE
            # =========================

            try:
                equity = om.get_equity(SYMBOL) if hasattr(om, 'get_equity') else portfolio_value
            except Exception as e:
                logger.error(f"Failed to get equity: {e}")
                equity = portfolio_value

            try:
                risk_decision = risk_engine.evaluate(
                    signal=raw_action,
                    candles=ohlcv,
                    equity=equity,
                    price=price,
                    has_position=tracker.has_position()
                )
            except Exception as e:
                logger.error(f"Risk engine evaluation failed: {e}")
                # Default to HOLD on risk engine error
                risk_decision = {"action": "HOLD", "size": 0}

            # =========================
            # MEMORY REWARD
            # =========================
            reward = 0.0
            if last_equity is not None:
                reward = equity - last_equity

            action = risk_decision.get("action", "HOLD")
            size = risk_decision.get("size", 0)

            logger.info(f"Risk Decision → {action} (size: {size})")

            if action == "BUY" and size > 0:
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
                position_size = tracker.position_size()

                if position_size <= 0:
                    logger.warning("SELL ignored — no position")
                else:
                    if PAPER:
                        exec_price = price
                        exec_amount = position_size
                        logger.info(f"[PAPER SELL] {exec_amount} @ {exec_price}")
                    else:
                        try:
                            res = om.market_sell(SYMBOL, position_size)
                            exec_price = res.get("price", price)
                            exec_amount = res.get("amount", position_size)
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
            try:
                cash = om.state.state["cash"] if hasattr(om, 'state') else usdt
                position = om.state.state["position"] if hasattr(om, 'state') else btc
            except:
                cash = usdt
                position = btc
                
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
            try:
                replay.append(
                    price=price,
                    action=int(raw_action),
                    reward=float(reward),
                    equity=float(equity)
                )
            except Exception as e:
                logger.error(f"Failed to save experience to replay buffer: {e}")

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
                    try:
                        if replay.size() > MAX_REPLAY_SIZE:
                            replay.trim_oldest(MAX_REPLAY_SIZE // 2)
                            logger.info(f"Replay buffer trimmed to {MAX_REPLAY_SIZE // 2} samples")
                        elif replay.size() < MIN_REPLAY_SIZE:
                            logger.info("Skip fine-tune: replay buffer too small")
                        elif not risk_engine.allow_training(equity):
                            logger.warning("Skip fine-tune: drawdown too high")
                        else:
                            logger.info("Online learning triggered...")
                            fine_tune()
                            logger.info("Fine-tune completed successfully")
                    except Exception as e:
                        logger.error(f"Online learning error: {e}")

            # reset error counter & sleep
            error_count = 0
            time.sleep(SLEEP_SECONDS)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            break
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
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    finally:
        if 'conn' in locals() and conn:
            try:
                conn.close()
            except:
                pass

if __name__ == "__main__":
    main_loop()