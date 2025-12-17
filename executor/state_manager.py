# executor/state_manager.py
import ccxt
import time

class StateManager:
    def __init__(self, exchange: ccxt.Exchange, symbol: str):
        self.exchange = exchange
        self.symbol = symbol
        self.base, self.quote = symbol.split("/")

        self.last_sync = 0
        self.cache_ttl = 3  # detik

        self.state = {
            "cash": 0.0,
            "position": 0.0,
            "avg_entry": 0.0,
            "equity": 0.0
        }

    def sync(self, force=False):
        now = time.time()
        if not force and now - self.last_sync < self.cache_ttl:
            return self.state

        balance = self.exchange.fetch_balance()

        cash = float(balance["free"].get(self.quote, 0.0))
        position = float(balance["free"].get(self.base, 0.0))

        price = self.exchange.fetch_ticker(self.symbol)["last"]

        equity = cash + position * price

        self.state.update({
            "cash": cash,
            "position": position,
            "equity": equity
        })

        self.last_sync = now
        return self.state

    def has_position(self) -> bool:
        self.sync()
        return self.state["position"] > 0

    def can_buy(self, min_quote: float = 5.0) -> bool:
        self.sync()
        return self.state["cash"] >= min_quote

    def can_sell(self, min_base: float = 0.00001) -> bool:
        self.sync()
        return self.state["position"] >= min_base
