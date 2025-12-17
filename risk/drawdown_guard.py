class DrawdownGuard:
    def __init__(self, max_dd=0.20):
        self.peak = None
        self.max_dd = max_dd

    def update(self, equity):
        if self.peak is None or equity > self.peak:
            self.peak = equity

    def breached(self, equity):
        if self.peak is None:
            return False
        dd = (self.peak - equity) / self.peak
        return dd >= self.max_dd
