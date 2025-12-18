class PositionState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.entry_price = None
        self.size = 0.0
        self.stop_loss = None
        self.take_profit = None
        self.trailing_stop = None
        self.max_price = None

    def open(self, price, size, stop_loss, take_profit):
        self.entry_price = price
        self.size = size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_price = price
        self.trailing_stop = stop_loss

    def update_price(self, price, trail_pct):
        if self.entry_price is None:
            return

        if price > self.max_price:
            self.max_price = price
            self.trailing_stop = max(
                self.trailing_stop,
                self.max_price * (1 - trail_pct)
            )

    def should_exit(self, price):
        if self.entry_price is None:
            return False, None

        if price <= self.trailing_stop:
            return True, "TRAILING_STOP"

        if price <= self.stop_loss:
            return True, "STOP_LOSS"

        if price >= self.take_profit:
            return True, "TAKE_PROFIT"

        return False, None
