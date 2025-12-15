import time
import traceback
from datetime import datetime

from executor.order_manager import OrderManager
from ai.infer import AIDecisionEngine


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
            decision = ai_engine.decide(candles)

            if not decision:
                log("AI returned no decision")
                continue

            action = decision["action"]
            confidence = decision.get("confidence", 0)

            log(f"AI decision: {action} (confidence={confidence:.2f})")

            # ======================================
            # EXECUTION
            # ======================================
            if action == "BUY":
                order_manager.create_market_buy(
                    SYMBOL,
                    decision["quote_amount"],
                )

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
