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
        reward REAL,
        cash REAL,
        position REAL
    )
    """)
    conn.commit()
    return conn

def log_trade(conn, side, symbol, amount, price, mode, note="", equity=None, reward=None, cash=None, position=None):
    ts = datetime.utcnow().isoformat()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO trades (ts, side, symbol, amount, price, mode, note, equity, reward, cash, position) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (ts, side, symbol, amount, price, mode, note, equity, reward, cash, position)
    )
    conn.commit()

def get_price_with_fallback(om, symbol, max_retries=3):
    """Get price with multiple fallback methods"""
    for attempt in range(1, max_retries + 1):
        try:
            # Method 1: get_ticker if available
            if hasattr(om, "get_ticker"):
                ticker = om._safe_request(om.get_ticker, symbol) if hasattr(om, "_safe_request") else om.get_ticker(symbol)
                if isinstance(ticker, dict):
                    # Handle missing volume field
                    if "last" in ticker:
                        price = float(ticker["last"])
                        # Log warning if volume is missing but not critical
                        if "volume" not in ticker and attempt == 1:
                            logger.debug(f"Volume field missing in ticker, but price retrieved: {price}")
                        return price
            
            # Method 2: fetch_ticker directly
            ticker = om.exchange.fetch_ticker(symbol)
            if isinstance(ticker, dict) and "last" in ticker:
                price = float(ticker["last"])
                # Log warning if volume is missing but not critical
                if "volume" not in ticker and attempt == 1:
                    logger.debug(f"Volume field missing in ticker, but price retrieved: {price}")
                return price
                
        except Exception as e:
            if attempt < max_retries:
                logger.warning(f"Price fetch error: {e}, retry {attempt}/{max_retries}")
                time.sleep(2 * attempt)
                continue
    
    # Final fallback: Use OHLCV
    try:
        ohlcv = om.exchange.fetch_ohlcv(symbol, '1m', 2)
        if len(ohlcv) >= 2:
            return float(ohlcv[-1][4])
    except Exception as e:
        logger.error(f"OHLCV fallback also failed: {e}")
    
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

def get_balance_with_retry(om, max_retries=3):
    """Get balance with retry logic"""
    for attempt in range(max_retries):
        try:
            balance = om.exchange.fetch_balance()
            if balance and 'free' in balance:
                return balance
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Balance fetch failed (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(2 * (attempt + 1))
            else:
                logger.error(f"Balance fetch failed after {max_retries} attempts: {e}")
    return None

def convert_ai_signal_for_risk_engine(ai_signal):
    """Convert AI signal (0,1,2) to risk engine probability (0-1)."""
    if ai_signal == 0:  # BUY
        return 0.8  # 80% confidence
    elif ai_signal == 2:  # SELL  
        return 0.2  # 20% confidence (bearish)
    else:  # HOLD
        return 0.5  # Neutral

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
                return
        except Exception as e:
            logger.error(f"Connection test error: {e}")
            # Continue anyway, might be a temporary issue

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

    conn = init_db()

    error_count = 0
    balance = None
    usdt_balance = 0.0
    btc_balance = 0.0

    # Get initial balance
    try:
        balance = get_balance_with_retry(om)
        if balance:
            usdt_balance = balance['free'].get('USDT', 0)
            btc_balance = balance['free'].get('BTC', 0)
            logger.info(f"Initial balance - USDT: {usdt_balance:.2f}, BTC: {btc_balance:.6f}")
    except Exception as e:
        logger.error(f"Failed to get initial balance: {e}")

    while not stop_requested:
        try:
            # pick a working proxy and apply if needed
            proxy = proxy_mgr.get_working_proxy()
            if proxy:
                logger.debug(f"Using proxy: {proxy}")

            # fetch latest price via a safe call
            price = get_price_with_fallback(om, SYMBOL)
            if price <= 0:
                logger.error(f"Invalid price received: {price}")
                time.sleep(SLEEP_SECONDS)
                continue
                
            logger.info(f"Current price: ${price:,.2f}")

            # =========================
            # AI ENSEMBLE DECISION
            # =========================

            # Get OHLCV window untuk AI
            WINDOW = int(os.getenv("AI_WINDOW", "50"))

            try:
                if hasattr(om, "_safe_request"):
                    ohlcv = om._safe_request(om.fetch_ohlcv, SYMBOL, '1m', WINDOW)
                else:
                    ohlcv = om.fetch_ohlcv(SYMBOL, '1m', WINDOW)
                    
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
            
            # DEBUG DETAILED
            logger.info(f"=== AI DECISION DEBUG ===")
            logger.info(f"Latest closes (last 10): {[f'{x:.1f}' for x in closes[-10:]]}")
            logger.info(f"Current price: ${price:,.2f}")
            logger.info(f"Data points: {len(closes)}")

            window = np.array(closes, dtype=float)
            mean_val = window.mean()
            std_val = window.std()
            logger.info(f"Window stats - Mean: ${mean_val:,.2f}, Std: ${std_val:,.2f}")
            
            if mean_val == 0:
                window_norm = window.copy()
            else:
                window_norm = window / mean_val
            
            logger.info(f"Normalized last value: {window_norm[-1]:.4f}")
            logger.info(f"Normalized range: min={window_norm.min():.4f}, max={window_norm.max():.4f}")

            # Update balance periodically
            if step_counter % 10 == 0 or balance is None:
                try:
                    balance = get_balance_with_retry(om)
                    if balance:
                        usdt_balance = balance['free'].get('USDT', 0)
                        btc_balance = balance['free'].get('BTC', 0)
                        logger.info(f"Balance - USDT: ${usdt_balance:,.2f}, BTC: {btc_balance:.8f}")
                except Exception as e:
                    logger.warning(f"Balance update failed: {e}")

            portfolio_value = usdt_balance + btc_balance * price + 1e-9
            logger.info(f"Portfolio value: ${portfolio_value:,.2f}")
            
            if portfolio_value < 1e-9:
                cash_ratio = 0.0
            else:
                cash_ratio = usdt_balance / portfolio_value
            
            # Use actual position size from tracker if available
            if tracker.has_position():
                pos_ratio = tracker.position_size()
                logger.info(f"Tracker position: {pos_ratio:.8f} BTC")
            else:
                pos_ratio = btc_balance
                logger.info(f"Balance position: {pos_ratio:.8f} BTC")

            state = np.concatenate([window_norm, [cash_ratio, pos_ratio]]).astype(np.float32)
            
            # DEBUG: Show critical ratios
            logger.info(f"Cash ratio: {cash_ratio:.4f} ({usdt_balance:,.2f}/{portfolio_value:,.2f})")
            logger.info(f"Position ratio: {pos_ratio:.8f}")
            
            # Calculate moving averages for manual check
            if len(window_norm) >= 20:
                short_ma = np.mean(window_norm[-10:]) if len(window_norm) >= 10 else 1.0
                long_ma = np.mean(window_norm[-20:]) if len(window_norm) >= 20 else 1.0
                logger.info(f"MA Analysis - Short MA ({min(10, len(window_norm))}): {short_ma:.4f}, Long MA ({min(20, len(window_norm))}): {long_ma:.4f}")
                logger.info(f"MA Comparison - Short vs Long: {short_ma:.4f} vs {long_ma:.4f}, Diff: {short_ma - long_ma:.4f}")
                logger.info(f"Price vs Short MA: {window_norm[-1]:.4f} vs {short_ma:.4f}, Diff: {window_norm[-1] - short_ma:.4f}")

            # AI decision
            try:
                raw_action = ensemble.decide(state)
                logger.info(f"Raw AI decision: {raw_action} (0=BUY, 1=HOLD, 2=SELL)")
                
                # Get detailed decision info if available
                if ENABLE_RULE and len(ensemble_models) > 0:
                    for i, model in enumerate(ensemble_models):
                        model_name = model.__class__.__name__
                        logger.info(f"Checking model: {model_name}")
                        
                        if hasattr(model, 'get_last_decision_info'):
                            info = model.get_last_decision_info()
                            if info:
                                logger.info(f"Model '{model_name}' decision details:")
                                for key, value in info.items():
                                    logger.info(f"  {key}: {value}")
                        
                        # Also check if model has any debug attributes
                        if hasattr(model, 'debug_info'):
                            logger.info(f"Model '{model_name}' debug: {model.debug_info}")
                            
            except Exception as e:
                logger.error(f"AI decision failed: {e}")
                raw_action = 1
                logger.info("Using default HOLD action due to AI error")

            logger.info(f"=== END AI DEBUG ===")

            # =========================
            # RISK MANAGEMENT GATE
            # =========================

            try:
                if hasattr(om, 'get_equity'):
                    equity = om.get_equity(SYMBOL)
                else:
                    equity = portfolio_value
            except Exception as e:
                logger.error(f"Failed to get equity: {e}")
                equity = portfolio_value

            logger.debug(f"Equity: ${equity:.2f}, Has position: {tracker.has_position()}")

            # Convert AI signal untuk risk engine
            converted_signal = convert_ai_signal_for_risk_engine(raw_action)
            logger.info(f"Signal conversion: AI={raw_action} → RiskEngine={converted_signal:.2f}")

            # DEBUG: Log semua input ke risk engine
            logger.info(f"=== RISK ENGINE INPUT DEBUG ===")
            logger.info(f"Signal: {converted_signal:.2f}")
            logger.info(f"Has position: {tracker.has_position()}")
            logger.info(f"Position size: {tracker.position_size():.8f} BTC")
            logger.info(f"Price: ${price:.2f}")
            logger.info(f"Equity: ${equity:.2f}")
            logger.info(f"Candles count: {len(ohlcv)}")
            logger.info(f"=== END RISK ENGINE DEBUG ===")

            try:
                risk_decision = risk_engine.evaluate(
                    signal=converted_signal,
                    candles=ohlcv,
                    equity=equity,
                    price=price,
                    has_position=tracker.has_position()
                )
                logger.debug(f"Risk decision details: {risk_decision}")
                
                # DEBUG: Log reason dari risk engine
                if "reason" in risk_decision:
                    logger.info(f"Risk Engine Reason: {risk_decision['reason']}")
                    
            except Exception as e:
                logger.error(f"Risk engine evaluation failed: {e}")
                risk_decision = {"action": "HOLD", "size": 0}

            # =========================
            # MEMORY REWARD
            # =========================
            reward = 0.0
            if last_equity is not None and last_equity != 0:
                reward = (equity - last_equity) / last_equity * 100  # Reward as percentage

            action = risk_decision.get("action", "HOLD")
            size = risk_decision.get("size", 0)

            # Tampilkan reason mengapa HOLD jika terjadi
            if action == "HOLD":
                if raw_action == 1:  # Jika AI sudah HOLD
                    logger.info("AI decided to HOLD")
                else:
                    logger.info(f"AI suggested {raw_action} but risk engine decided HOLD")

            logger.info(f"Risk Decision → {action} (size: {size:.6f} BTC)")

            if action == "BUY" and size > 0:
                # Ensure we have enough USDT
                if usdt_balance < size * price * 1.01:  # Include 1% buffer for fees
                    logger.warning(f"Insufficient USDT for BUY. Need: {size * price * 1.01:.2f}, Have: {usdt_balance:.2f}")
                    size = min(size, usdt_balance / price * 0.99)  # Adjust size
                    if size <= 0:
                        logger.warning("Adjusted size is zero or negative, skipping BUY")
                        action = "HOLD"
                
                if action == "BUY" and size > 0:
                    if PAPER:
                        exec_price = price
                        exec_amount = size
                        logger.info(f"[PAPER BUY] {exec_amount:.6f} BTC @ ${exec_price:,.2f} (${exec_amount * exec_price:,.2f})")
                        
                        # Update paper balances
                        usdt_balance -= exec_amount * exec_price
                        btc_balance += exec_amount
                    else:
                        try:
                            res = om.market_buy(SYMBOL, size)
                            exec_price = res.get("price", price)
                            exec_amount = res.get("amount", size)
                            logger.info(f"[LIVE BUY] {exec_amount:.6f} BTC @ ${exec_price:,.2f} (${exec_amount * exec_price:,.2f})")
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
                                equity=equity, reward=reward, cash=usdt_balance, position=btc_balance)

            elif action == "SELL":
                position_size = tracker.position_size()

                if position_size <= 0:
                    logger.warning("SELL ignored — no position")
                else:
                    if PAPER:
                        exec_price = price
                        exec_amount = position_size
                        logger.info(f"[PAPER SELL] {exec_amount:.6f} BTC @ ${exec_price:,.2f} (${exec_amount * exec_price:,.2f})")
                        
                        # Update paper balances
                        usdt_balance += exec_amount * exec_price
                        btc_balance -= exec_amount
                    else:
                        try:
                            res = om.market_sell(SYMBOL, position_size)
                            exec_price = res.get("price", price)
                            exec_amount = res.get("amount", position_size)
                            logger.info(f"[LIVE SELL] {exec_amount:.6f} BTC @ ${exec_price:,.2f} (${exec_amount * exec_price:,.2f})")
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
                                equity=equity, reward=reward, cash=usdt_balance, position=btc_balance)

            else:
                logger.debug("HOLD - No action taken")

            # Log metrics
            try:
                cash = om.state.state["cash"] if hasattr(om, 'state') and hasattr(om.state, 'state') else usdt_balance
                position = om.state.state["position"] if hasattr(om, 'state') and hasattr(om.state, 'state') else btc_balance
            except:
                cash = usdt_balance
                position = btc_balance
                
            logger.info(f"Equity: ${equity:,.2f} | Cash: ${cash:,.2f} | Position: {position:.6f} BTC (${position * price:,.2f})")
            logger.info(f"Reward: {reward:+.2f}% | Portfolio Value: ${portfolio_value:,.2f}")
            
            # Collect metrics - PERBAIKAN: hanya gunakan parameter yang diperlukan
            try:
                metrics.record_trade_step(
                    price=price,
                    action=action,
                    equity=equity,
                    reward=reward,
                    portfolio_value=portfolio_value
                )
            except Exception as e:
                logger.error(f"Failed to record metrics: {e}")

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
            logger.error(f"Loop error: {type(e).__name__}: {str(e)[:200]}")
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
        logger.info(f"Final balance - USDT: {usdt_balance:.2f}, BTC: {btc_balance:.6f}")
        logger.info(f"Total steps executed: {step_counter}")
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