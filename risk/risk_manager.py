import time

class RiskManager:
    def __init__(
        self,
        max_risk_per_trade=0.01,   # 1% equity
        max_drawdown=0.15,         # 15% hard stop
        min_rr=1.5,                # risk:reward
        cooldown_sec=60
    ):
        self.max_risk_per_trade = max_risk_per_trade
        self.max_drawdown = max_drawdown
        self.min_rr = min_rr
        self.cooldown_sec = cooldown_sec

        self.peak_equity = None
        self.last_trade_ts = 0

    def update_equity(self, equity):
        if self.peak_equity is None:
            self.peak_equity = equity
        self.peak_equity = max(self.peak_equity, equity)

    def allow_trade(self, equity):
        self.update_equity(equity)

        drawdown = (self.peak_equity - equity) / self.peak_equity
        if drawdown >= self.max_drawdown:
            return False, "MAX_DRAWDOWN"

        if time.time() - self.last_trade_ts < self.cooldown_sec:
            return False, "COOLDOWN"

        return True, "OK"

    def position_size(self, equity, entry_price, stop_price):
        risk_amount = equity * self.max_risk_per_trade
        risk_per_unit = abs(entry_price - stop_price)

        if risk_per_unit <= 0:
            return 0

        qty = risk_amount / risk_per_unit
        return qty

    def register_trade(self):
        self.last_trade_ts = time.time()
