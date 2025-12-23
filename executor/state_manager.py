import time
import logging
from typing import Dict, Tuple, Optional, TypedDict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class Trade(TypedDict):
    """Trade data structure"""
    side: str  # "buy" or "sell"
    amount: float
    price: float
    timestamp: Optional[float]

@dataclass
class AccountState:
    """Data class for account state"""
    cash: float = 0.0
    position: float = 0.0
    avg_price: float = 0.0
    equity: float = 0.0
    last_update: float = 0.0
    last_sync: float = 0.0

class StateManager:
    """
    Manages local account state with validation and consistency checks.
    Acts as a shadow state to reduce API calls to exchange.
    """
    
    def __init__(self, initial_cash: float = 0.0):
        self.state = AccountState(
            cash=float(initial_cash),
            equity=float(initial_cash),
            last_update=time.time(),
            last_sync=time.time()
        )
        self._initial_cash = float(initial_cash)
        
    def sync_from_exchange(self, balance: dict, price: float) -> Tuple[bool, str]:
        """
        Safely sync state from exchange balance.
        
        Args:
            balance: ccxt.fetch_balance() result
            price: Current market price
            
        Returns:
            (success, message)
        """
        try:
            # Safely extract balance data
            free_balance = balance.get("free", {})
            if not isinstance(free_balance, dict):
                free_balance = {}
            
            # Try multiple key formats
            usdt_keys = ["USDT", "usdt"]
            btc_keys = ["BTC", "btc"]
            
            usdt = 0.0
            for key in usdt_keys:
                if key in free_balance:
                    try:
                        usdt = float(free_balance[key])
                        break
                    except (ValueError, TypeError):
                        continue
            
            btc = 0.0
            for key in btc_keys:
                if key in free_balance:
                    try:
                        btc = float(free_balance[key])
                        break
                    except (ValueError, TypeError):
                        continue
            
            # Ensure non-negative
            usdt = max(0.0, usdt)
            btc = max(0.0, btc)
            
            # Calculate equity
            equity = usdt + btc * price
            
            # Update average price logic
            avg_price = self.state.avg_price  # Keep existing by default
            
            if btc <= 1e-10:  # No position
                avg_price = 0.0
            elif abs(self.state.position - btc) / max(abs(btc), 1e-10) > 0.1:
                # Position changed significantly, need to adjust avg_price
                if self.state.position > 0 and btc > 0:
                    # Approximate based on previous state
                    if self.state.cash > usdt:
                        # Cash decreased, likely bought more
                        spent = self.state.cash - usdt
                        if spent > 0:
                            # Estimate new average
                            old_value = self.state.position * self.state.avg_price
                            new_value = old_value + spent
                            avg_price = new_value / btc if btc > 0 else price
                    else:
                        # Cash increased, likely sold
                        avg_price = self.state.avg_price  # Keep same for remaining
                else:
                    avg_price = price  # Fresh position
            
            # Update state
            self.state.cash = usdt
            self.state.position = btc
            self.state.avg_price = avg_price
            self.state.equity = equity
            self.state.last_update = time.time()
            self.state.last_sync = time.time()
            
            logger.debug(f"Synced from exchange: USDT={usdt:.2f}, BTC={btc:.6f}, Equity=${equity:.2f}")
            return True, "SYNC_SUCCESS"
            
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            return False, f"SYNC_FAILED: {str(e)}"
    
    def update_from_trade(self, trade: Trade) -> Tuple[bool, str, float]:
        """
        Update state from a trade execution.
        
        Args:
            trade: Trade dictionary with side, amount, price
            
        Returns:
            (success, message, executed_amount)
        """
        try:
            side = trade.get("side", "").lower()
            amount = float(trade.get("amount", 0))
            price = float(trade.get("price", 0))
            
            # Validate inputs
            if side not in ["buy", "sell"]:
                return False, f"INVALID_SIDE: {side}", 0.0
            
            if amount <= 1e-10:
                return False, "INVALID_AMOUNT", 0.0
            
            if price <= 1e-10:
                return False, "INVALID_PRICE", 0.0
            
            executed_amount = amount
            
            if side == "buy":
                cost = amount * price
                
                # Check if we have enough cash
                if self.state.cash < cost:
                    # Adjust amount to what we can afford
                    max_amount = self.state.cash / price if price > 0 else 0
                    if max_amount <= 1e-10:
                        return False, "INSUFFICIENT_CASH", 0.0
                    
                    amount = max_amount
                    cost = amount * price
                    executed_amount = amount
                    logger.warning(f"Adjusted buy amount to {amount:.6f} due to cash limits")
                
                self.state.cash -= cost
                
                # Update average price (weighted average)
                prev_pos = self.state.position
                prev_avg = self.state.avg_price
                
                new_pos = prev_pos + amount
                if new_pos > 1e-10:
                    self.state.avg_price = (
                        (prev_pos * prev_avg + amount * price) / new_pos
                    )
                else:
                    self.state.avg_price = 0.0
                
                self.state.position = new_pos
                message = f"BUY_EXECUTED_{executed_amount:.6f}"
                
            else:  # sell
                # Cannot sell more than we have
                if amount > self.state.position:
                    amount = self.state.position
                    if amount <= 1e-10:
                        return False, "NO_POSITION_TO_SELL", 0.0
                    
                    executed_amount = amount
                    logger.warning(f"Adjusted sell amount to {amount:.6f}")
                
                revenue = amount * price
                self.state.cash += revenue
                self.state.position -= amount
                
                # Reset average price if position is zero
                if self.state.position <= 1e-10:
                    self.state.position = 0.0
                    self.state.avg_price = 0.0
                
                message = f"SELL_EXECUTED_{executed_amount:.6f}"
            
            # Update equity
            self.state.equity = self.state.cash + self.state.position * price
            self.state.last_update = time.time()
            
            logger.debug(f"Updated from trade: {side} {executed_amount:.6f} @ ${price:.2f}")
            return True, message, executed_amount
            
        except Exception as e:
            logger.error(f"Trade update failed: {e}")
            return False, f"UPDATE_FAILED: {str(e)}", 0.0
    
    # =====================================================
    # METRICS & VALIDATION
    # =====================================================
    
    def update_equity(self, price: float) -> float:
        """Recalculate and update equity based on current price."""
        self.state.equity = self.state.cash + self.state.position * price
        self.state.last_update = time.time()
        return self.state.equity
    
    def get_equity(self, price: Optional[float] = None) -> float:
        """Get equity, optionally recalculating with current price."""
        if price is not None:
            return self.state.cash + self.state.position * price
        return float(self.state.equity)
    
    def has_position(self, threshold: float = 1e-10) -> bool:
        """Check if we have a position above threshold."""
        return self.state.position > threshold
    
    def get_position(self) -> float:
        return float(self.state.position)
    
    def get_cash(self) -> float:
        return float(self.state.cash)
    
    def get_avg_price(self) -> float:
        return float(self.state.avg_price)
    
    def get_unrealized_pnl(self, current_price: float) -> Tuple[float, float]:
        """Calculate unrealized P&L in value and percentage."""
        if self.state.position <= 1e-10 or self.state.avg_price <= 1e-10:
            return 0.0, 0.0
        
        pnl_value = (current_price - self.state.avg_price) * self.state.position
        pnl_percent = ((current_price / self.state.avg_price) - 1) * 100
        
        return pnl_value, pnl_percent
    
    def validate_state(self) -> Tuple[bool, str]:
        """Validate state consistency."""
        issues = []
        
        # Basic validations
        if self.state.cash < -1e-6:  # Allow tiny negative due to floating point
            issues.append(f"Negative cash: {self.state.cash:.2f}")
        
        if self.state.position < -1e-6:
            issues.append(f"Negative position: {self.state.position:.6f}")
        
        if self.state.avg_price < -1e-6 and self.state.position > 1e-6:
            issues.append(f"Negative avg_price with position: {self.state.avg_price:.2f}")
        
        # Equity consistency check
        calculated_equity = self.state.cash + self.state.position * max(self.state.avg_price, 1.0)
        equity_diff = abs(calculated_equity - self.state.equity)
        
        if equity_diff > 1.0:  # $1 tolerance
            issues.append(f"Equity mismatch: {equity_diff:.2f}")
        
        # Check if state is stale
        stale_time = time.time() - self.state.last_update
        if stale_time > 300:  # 5 minutes
            issues.append(f"State stale: {stale_time:.0f}s")
        
        valid = len(issues) == 0
        message = "STATE_VALID" if valid else "; ".join(issues)
        
        return valid, message
    
    def reset(self, cash: Optional[float] = None) -> None:
        """Reset state to initial or specified cash."""
        reset_cash = cash if cash is not None else self._initial_cash
        self.state = AccountState(
            cash=float(reset_cash),
            equity=float(reset_cash),
            last_update=time.time(),
            last_sync=time.time()
        )
        logger.info(f"State reset to cash=${reset_cash:.2f}")
    
    def snapshot(self) -> Dict:
        """Get complete state snapshot."""
        return {
            "cash": self.state.cash,
            "position": self.state.position,
            "avg_price": self.state.avg_price,
            "equity": self.state.equity,
            "last_update": self.state.last_update,
            "last_sync": self.state.last_sync,
            "initial_cash": self._initial_cash,
            "has_position": self.has_position(),
        }
    
    def __str__(self) -> str:
        """String representation."""
        pnl_val, pnl_pct = self.get_unrealized_pnl(self.state.avg_price or 1.0)
        
        return (f"StateManager("
                f"Cash=${self.state.cash:.2f}, "
                f"Pos={self.state.position:.6f}, "
                f"Avg=${self.state.avg_price:.2f}, "
                f"Equity=${self.state.equity:.2f}, "
                f"P&L=${pnl_val:.2f} ({pnl_pct:.2f}%))")