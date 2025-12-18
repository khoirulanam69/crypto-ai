import time

class PositionTracker:
    def __init__(self):
        self.position = 0.0

    def update_from_trade(self, trade):
        if trade['side'] == 'buy':
            self.position += trade['amount']
        elif trade['side'] == 'sell':
            self.position -= trade['amount']
        self.state.state["last_update"] = time.time()

    def has_position(self):
        return self.position > 0
