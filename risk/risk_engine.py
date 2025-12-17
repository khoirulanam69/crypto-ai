from .volatility import atr
from .position_sizer import calc_position_size
from .drawdown_guard import DrawdownGuard

class RiskEngine:
    def __init__(self, max_dd=0.20):
        self.dd_guard = DrawdownGuard(max_dd)

    def evaluate(
        self,
        signal: int,
        candles: list,
        equity: float,
        price: float,
        has_position: bool
    ):
        self.dd_guard.update(equity)

        if self.dd_guard.breached(equity):
            return {
                "action": "HOLD",
                "reason": "MAX_DRAWDOWN"
            }

        atr_val = atr(candles)

        if signal == 1 and not has_position:  # BUY
            size = calc_position_size(equity, price, atr_val)
            if size <= 0:
                return {"action": "HOLD", "reason": "SIZE_ZERO"}

            return {
                "action": "BUY",
                "size": size,
                "atr": atr_val
            }

        if signal == 2 and has_position:  # SELL
            return {"action": "SELL"}

        return {"action": "HOLD"}
