from .volatility import atr
from .position_sizer import calc_position_size
from .drawdown_guard import DrawdownGuard
from executor.position_state import PositionState
from risk.indicators import ATR
import pandas as pd

class RiskEngine:
    def __init__(self, max_dd=0.2, risk_per_trade=0.01, trail_pct=0.005):
        self.max_dd = max_dd
        self.risk_per_trade = risk_per_trade
        self.trail_pct = trail_pct
        self.position = PositionState()
        self.peak_equity = None

    def evaluate(self, signal, candles, equity, price, has_position):
        # update drawdown
        self.peak_equity = equity if self.peak_equity is None else max(self.peak_equity, equity)
        dd = (self.peak_equity - equity) / self.peak_equity

        if dd >= self.max_dd:
            return {"action": "HOLD", "reason": "MAX_DRAWDOWN"}

        # =====================
        # EXIT MANAGEMENT
        # =====================
        if has_position:
            self.position.update_price(price, self.trail_pct)
            exit_now, reason = self.position.should_exit(price)

            if exit_now:
                return {
                    "action": "SELL",
                    "reason": reason,
                    "size": self.position.size
                }

            return {"action": "HOLD", "reason": "IN_POSITION"}

        # =====================
        # ENTRY MANAGEMENT
        # =====================
        if signal != 1:  # bukan BUY
            return {"action": "HOLD", "reason": "NO_ENTRY_SIGNAL"}

        df = pd.DataFrame(
            candles,
            columns=["ts", "open", "high", "low", "close", "volume"]
        )
        atr = ATR(df).iloc[-1]

        if atr is None or atr != atr:
            return {"action": "HOLD", "reason": "ATR_INVALID"}

        stop_loss = price - atr * 1.5
        take_profit = price + atr * 3.0

        risk_amount = equity * self.risk_per_trade
        size = risk_amount / (price - stop_loss)

        if size <= 0:
            return {"action": "HOLD", "reason": "SIZE_ZERO"}

        # register position
        self.position.open(price, size, stop_loss, take_profit)

        return {
            "action": "BUY",
            "size": size,
            "stop_loss": stop_loss,
            "take_profit": take_profit
        }

    def on_position_closed(self):
        self.position.reset()