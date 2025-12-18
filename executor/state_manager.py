# executor/state_manager.py
import time
from typing import Dict


class StateManager:
    """
    Menyimpan state akun lokal (shadow state) agar:
    - risk engine bisa membaca equity
    - AI punya feedback posisi
    - tidak tergantung fetch balance terus-menerus
    """

    def __init__(self, initial_cash: float = 0.0):
        self.state: Dict[str, float] = {
            "cash": float(initial_cash),
            "position": 0.0,
            "avg_price": 0.0,
            "equity": float(initial_cash),
            "last_update": time.time(),
        }

    # =====================================================
    # UPDATE DARI BALANCE EXCHANGE
    # =====================================================

    def sync_from_exchange(self, balance: dict, price: float):
        """
        balance: hasil ccxt.fetch_balance()
        price: harga terakhir
        """
        usdt = balance.get("free", {}).get("USDT", 0.0)
        btc = balance.get("free", {}).get("BTC", 0.0)

        equity = usdt + btc * price

        self.state.update({
            "cash": float(usdt),
            "position": float(btc),
            "equity": float(equity),
            "last_update": time.time(),
        })

    # =====================================================
    # UPDATE DARI TRADE
    # =====================================================

    def update_from_trade(self, trade: dict):
        """
        trade = {
            side: "buy" | "sell",
            amount: float,
            price: float
        }
        """
        side = trade.get("side")
        amount = float(trade.get("amount", 0))
        price = float(trade.get("price", 0))

        if amount <= 0 or price <= 0:
            return

        if side == "buy":
            cost = amount * price
            self.state["cash"] -= cost

            prev_pos = self.state["position"]
            prev_avg = self.state["avg_price"]

            new_pos = prev_pos + amount
            if new_pos > 0:
                self.state["avg_price"] = (
                    (prev_pos * prev_avg + amount * price) / new_pos
                )

            self.state["position"] = new_pos

        elif side == "sell":
            revenue = amount * price
            self.state["cash"] += revenue
            self.state["position"] -= amount

            if self.state["position"] <= 0:
                self.state["position"] = 0.0
                self.state["avg_price"] = 0.0

        self.state["last_update"] = time.time()

    # =====================================================
    # METRICS
    # =====================================================

    def update_equity(self, price: float):
        self.state["equity"] = (
            self.state["cash"] + self.state["position"] * price
        )

    def get_equity(self) -> float:
        return float(self.state.get("equity", 0.0))

    def has_position(self) -> bool:
        return self.state.get("position", 0.0) > 0

    def get_position(self) -> float:
        return float(self.state.get("position", 0.0))

    def get_cash(self) -> float:
        return float(self.state.get("cash", 0.0))

    def snapshot(self) -> dict:
        return dict(self.state)
