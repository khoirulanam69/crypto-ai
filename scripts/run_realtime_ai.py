import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import time
import traceback
import pandas as pd
from datetime import datetime

from executor.order_manager import OrderManager
from ai.infer import AIDecisionEngine
from risk.risk_manager import RiskManager
from risk.indicators import ATR

risk = RiskManager()

SYMBOL = "BTC/USDT"
TIMEFRAME = "1m"
CANDLE_LIMIT = 100

SLEEP_SECONDS = 5      # polling interval (AMAN utk Binance)
MAX_RETRIES = 5


def log(msg):
    ts = datetime.utcnow().isoformat()
    print(f"[{ts}] {msg}")


def main():
    log("Starting realtime AI trading bot...")

    order_manager = OrderManager()
    ai_engine = AIDecisionEngine(symbol=SYMBOL)

    last_candle_ts = None
    consecutive_errors = 0

    while True:
        try:
            # ======================================
            # FETCH MARKET DATA
            # ======================================
            candles = order_manager.fetch_ohlcv(
                SYMBOL,
                timeframe=TIMEFRAME,
                limit=CANDLE_LIMIT,
            )

            if not candles or len(candles) < 50:
                log("Not enough candle data, waiting...")
                time.sleep(SLEEP_SECONDS)
                continue

            # OHLCV format:
            # [timestamp, open, high, low, close, volume]
            latest = candles[-1]
            candle_ts = latest[0]

            # ======================================
            # PREVENT DOUBLE PROCESSING
            # ======================================
            if last_candle_ts == candle_ts:
                time.sleep(SLEEP_SECONDS)
                continue

            last_candle_ts = candle_ts

            # ======================================
            # AI DECISION
            # ======================================

            action = ai_engine.decide(candles)

            if action is None:
                logger.warning("AI returned no decision")
                continue

            # ======================================
            # RISK MANAGEMENT LAYER (WAJIB DI SINI)
            # ======================================

            equity = order_manager.get_equity()
            candles = order_manager.exchange.fetch_ohlcv(
                SYMBOL, timeframe="1m", limit=100
            )

            df = pd.DataFrame(
                candles,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

            last_price = float(
                order_manager.exchange.fetch_ticker(SYMBOL)["last"]
            )

            allowed, reason = risk.allow_trade(equity)
            if not allowed:
                logger.warning(f"[RISK BLOCKED] {reason}")
                continue

            price = last_price
            atr = ATR(df).iloc[-1]

            if atr is None or atr != atr:
                logger.warning("ATR invalid, skip trade")
                continue

            stop_loss = price - atr * 1.5
            take_profit = price + atr * 1.5 * risk.min_rr

            qty = risk.position_size(equity, price, stop_loss)
            if qty <= 0:
                logger.warning("Position size <= 0, skip trade")
                continue

            # ======================================
            # EXECUTION
            # ======================================

            if action == "BUY":
                order_manager.create_limit_buy(
                    SYMBOL,
                    price,
                    qty
                )
                risk.register_trade()

            elif action == "SELL":
                order_manager.create_market_sell(
                    SYMBOL,
                    decision["base_amount"],
                )

            else:
                log("HOLD")

            consecutive_errors = 0

        except KeyboardInterrupt:
            log("Bot stopped manually")
            break

        except Exception as e:
            consecutive_errors += 1
            log(f"ERROR: {e}")
            traceback.print_exc()

            if consecutive_errors >= MAX_RETRIES:
                log("Too many errors, cooling down 60s")
                time.sleep(60)
                consecutive_errors = 0

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
