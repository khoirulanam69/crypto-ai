class PositionState:
    """Manage state of a single trading position with stop loss and take profit."""
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> None:
        """Reset all position state to empty."""
        self.entry_price: float = None
        self.size: float = 0.0
        self.stop_loss: float = None
        self.take_profit: float = None
        self.trailing_stop: float = None
        self.max_price: float = None
        self.entry_time: float = None
        import time
        self.entry_time = time.time()
    
    def open(self, price: float, size: float, stop_loss: float, take_profit: float) -> None:
        """
        Open a new position.
        
        Args:
            price: Entry price (must be > 0)
            size: Position size in base asset (must be > 0)
            stop_loss: Stop loss price (must be < price for long)
            take_profit: Take profit price (must be > price for long)
        
        Raises:
            ValueError: If any parameter is invalid
        """
        # Validation
        if price <= 0:
            raise ValueError(f"Invalid entry price: {price}")
        if size <= 0:
            raise ValueError(f"Invalid position size: {size}")
        if stop_loss >= price:  # For long positions
            raise ValueError(f"Stop loss {stop_loss} must be less than entry price {price}")
        if take_profit <= price:  # For long positions
            raise ValueError(f"Take profit {take_profit} must be greater than entry price {price}")
        if stop_loss <= 0:
            raise ValueError(f"Invalid stop loss: {stop_loss}")
        
        self.entry_price = price
        self.size = size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_price = price
        self.trailing_stop = stop_loss  # Initial trailing stop = fixed stop loss
        self.entry_time = time.time()
    
    def update_price(self, price: float, trail_pct: float = 0.005) -> None:
        """
        Update position with current market price for trailing stop calculation.
        
        Args:
            price: Current market price
            trail_pct: Trailing stop percentage (e.g., 0.005 for 0.5%)
        """
        if self.entry_price is None:
            return
        
        # Update max price seen
        if price > self.max_price:
            self.max_price = price
            
            # Calculate new trailing stop based on max price
            new_trailing_stop = self.max_price * (1 - trail_pct)
            
            # Trailing stop hanya naik, tidak turun
            if self.trailing_stop is None or new_trailing_stop > self.trailing_stop:
                self.trailing_stop = new_trailing_stop
            
            # Also update fixed stop loss if trailing is higher
            if self.trailing_stop > self.stop_loss:
                self.stop_loss = self.trailing_stop
    
    def should_exit(self, price: float, tolerance: float = 0.0001) -> tuple[bool, str]:
        """
        Check if position should be exited based on current price.
        
        Args:
            price: Current market price
            tolerance: Price tolerance to account for spread/slippage (0.01% default)
        
        Returns:
            Tuple of (should_exit: bool, reason: str)
        """
        if self.entry_price is None or self.size == 0:
            return False, "NO_POSITION"
        
        # Adjust price with tolerance to prevent false triggers
        sell_price = price * (1 - tolerance)  # Slightly lower for sell checks
        buy_price = price * (1 + tolerance)   # Slightly higher for buy checks
        
        # Determine effective stop (highest of fixed or trailing)
        effective_stop = self.stop_loss
        if self.trailing_stop and self.trailing_stop > effective_stop:
            effective_stop = self.trailing_stop
        
        # Check stop loss (trailing or fixed)
        if sell_price <= effective_stop:
            # Determine which stop was hit
            if sell_price <= self.stop_loss:
                return True, "STOP_LOSS"
            elif self.trailing_stop and sell_price <= self.trailing_stop:
                return True, "TRAILING_STOP"
        
        # Check take profit
        if buy_price >= self.take_profit:
            return True, "TAKE_PROFIT"
        
        return False, "HOLDING"
    
    def calculate_pnl(self, current_price: float) -> tuple[float, float]:
        """
        Calculate unrealized profit/loss.
        
        Args:
            current_price: Current market price
        
        Returns:
            Tuple of (pnl_value, pnl_percentage)
        """
        if self.entry_price is None or self.size == 0:
            return 0.0, 0.0
        
        pnl_value = (current_price - self.entry_price) * self.size
        pnl_percent = ((current_price / self.entry_price) - 1) * 100
        
        return pnl_value, pnl_percent
    
    def calculate_pnl_percentage(self, current_price: float) -> float:
        """Calculate P&L as percentage."""
        if self.entry_price is None or self.entry_price == 0:
            return 0.0
        return ((current_price / self.entry_price) - 1) * 100
    
    def position_value(self, current_price: float = None) -> float:
        """Calculate current position value."""
        if self.entry_price is None or self.size == 0:
            return 0.0
        
        if current_price is None:
            current_price = self.entry_price
        
        return self.size * current_price
    
    def risk_reward_ratio(self) -> float:
        """Calculate risk/reward ratio of the position."""
        if self.entry_price is None:
            return 0.0
        
        risk = self.entry_price - self.stop_loss
        reward = self.take_profit - self.entry_price
        
        if risk <= 0:
            return 0.0
        
        return reward / risk
    
    def time_in_position(self) -> float:
        """Get time in position in seconds."""
        if self.entry_time is None:
            return 0.0
        return time.time() - self.entry_time
    
    def to_dict(self) -> dict:
        """Serialize position state to dictionary."""
        pnl_value, pnl_percent = self.calculate_pnl(self.max_price or self.entry_price or 0)
        
        return {
            "entry_price": self.entry_price,
            "size": self.size,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "trailing_stop": self.trailing_stop,
            "max_price": self.max_price,
            "entry_time": self.entry_time,
            "has_position": self.entry_price is not None,
            "current_pnl_value": pnl_value,
            "current_pnl_percent": pnl_percent,
            "position_value": self.position_value(self.max_price or self.entry_price),
            "risk_reward_ratio": self.risk_reward_ratio(),
            "time_in_position": self.time_in_position()
        }
    
    def __str__(self) -> str:
        """String representation of position."""
        if self.entry_price is None:
            return "PositionState(No Position)"
        
        pnl_val, pnl_pct = self.calculate_pnl(self.max_price or self.entry_price)
        
        return (f"PositionState("
                f"Entry=${self.entry_price:.2f}, "
                f"Size={self.size:.6f}, "
                f"SL=${self.stop_loss:.2f}, "
                f"TP=${self.take_profit:.2f}, "
                f"P&L=${pnl_val:.2f} ({pnl_pct:.2f}%), "
                f"Trail=${self.trailing_stop or 0:.2f})")