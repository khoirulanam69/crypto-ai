from executor.position_state import PositionState
import numpy as np
import math
import logging
from typing import Dict, Tuple, Optional, Union, List

logger = logging.getLogger(__name__)

class RiskEngine:
    """
    Risk management engine for trading with proper position sizing.
    
    Features:
    - Correct position sizing based on risk percentage
    - ATR-based stop loss and take profit
    - Drawdown protection
    - Maximum position size limits
    - Comprehensive validation
    """
    
    def __init__(self, 
                 max_dd: float = 0.2, 
                 risk_per_trade: float = 0.01, 
                 trail_pct: float = 0.005,
                 sl_multiplier: float = 1.5, 
                 tp_multiplier: float = 3.0, 
                 atr_period: int = 14,
                 max_position_size_pct: float = 0.1,  # Max 10% of equity
                 buy_threshold: float = 0.7,
                 min_position_value: float = 10.0,    # $10 minimum
                 max_position_duration_hours: int = 24):  # Close after 24h
        
        self.max_dd = max_dd
        self.risk_per_trade = risk_per_trade
        self.trail_pct = trail_pct
        self.sl_multiplier = sl_multiplier
        self.tp_multiplier = tp_multiplier
        self.atr_period = atr_period
        self.max_position_size_pct = max_position_size_pct
        self.buy_threshold = buy_threshold
        self.min_position_value = min_position_value
        self.max_position_duration_seconds = max_position_duration_hours * 3600
        
        self.position = PositionState()
        self.peak_equity = None
        self.initial_equity = None
        self.last_trade_time = None
        
        logger.info(
            f"RiskEngine initialized: "
            f"max_dd={max_dd*100:.0f}%, "
            f"risk_per_trade={risk_per_trade*100:.1f}%, "
            f"max_position={max_position_size_pct*100:.0f}%"
        )
    
    def calculate_atr(self, candles: List[List[float]]) -> Optional[float]:
        """
        Calculate Average True Range (ATR) from OHLCV data.
        
        Args:
            candles: List of [timestamp, open, high, low, close, volume]
            
        Returns:
            ATR value or None if insufficient data
        """
        if len(candles) < self.atr_period + 1:
            logger.debug(f"Insufficient data for ATR: {len(candles)} < {self.atr_period + 1}")
            return None
        
        true_ranges = []
        for i in range(1, len(candles)):
            try:
                high = float(candles[i][2])
                low = float(candles[i][3])
                prev_close = float(candles[i-1][4])
                
                tr1 = high - low
                tr2 = abs(high - prev_close)
                tr3 = abs(low - prev_close)
                true_ranges.append(max(tr1, tr2, tr3))
            except (IndexError, ValueError, TypeError) as e:
                logger.warning(f"Error calculating ATR at index {i}: {e}")
                continue
        
        if len(true_ranges) < self.atr_period:
            return None
        
        # Simple Moving Average of True Ranges
        atr_values = []
        for i in range(len(true_ranges) - self.atr_period + 1):
            window = true_ranges[i:i + self.atr_period]
            atr_values.append(sum(window) / self.atr_period)
        
        return atr_values[-1] if atr_values else None
    
    def validate_candles(self, candles: List[List[float]], min_candles: int = 20) -> Tuple[bool, str]:
        """Validate OHLCV data format and quality."""
        if not candles:
            return False, "NO_CANDLE_DATA"
        
        if len(candles) < min_candles:
            return False, f"INSUFFICIENT_DATA_{len(candles)}"
        
        for i, candle in enumerate(candles[-min_candles:]):
            # Check length
            if len(candle) != 6:
                return False, f"INVALID_FORMAT_AT_{i}"
            
            # Check data types and values
            for j, val in enumerate(candle[1:5]):  # Only check OHLC (skip volume)
                try:
                    float_val = float(val)
                    if float_val <= 0:
                        return False, f"NON_POSITIVE_PRICE_AT_{i}_COL_{j+1}"
                    if math.isnan(float_val) or math.isinf(float_val):
                        return False, f"INVALID_NUMERIC_AT_{i}_COL_{j+1}"
                except (ValueError, TypeError):
                    return False, f"NON_NUMERIC_AT_{i}_COL_{j+1}"
        
        return True, "OK"
    
    def calculate_position_size(self, equity: float, price: float, stop_loss_price: float) -> Tuple[float, str]:
        """
        CORRECT position size calculation for risk management.
        
        Formula: size = (risk_amount / risk_per_coin)
        Where:
          - risk_amount = equity * risk_per_trade
          - risk_per_coin = abs(price - stop_loss_price)
        
        Args:
            equity: Account equity in quote currency
            price: Current price
            stop_loss_price: Stop loss price
            
        Returns:
            (size_in_base_currency, status_message)
        """
        # Validate inputs
        if equity <= 0:
            return 0.0, "INVALID_EQUITY"
        
        if price <= 0:
            return 0.0, "INVALID_PRICE"
        
        if stop_loss_price <= 0:
            return 0.0, "INVALID_STOP_LOSS"
        
        # 1. Calculate risk amount in quote currency
        risk_amount = equity * self.risk_per_trade
        
        # 2. Calculate risk per coin (distance to stop in quote currency)
        risk_per_coin = abs(price - stop_loss_price)
        
        if risk_per_coin <= 0:
            return 0.0, "ZERO_RISK_PER_COIN"
        
        # 3. Calculate number of coins we can buy with our risk budget
        # Example: Risk $100, each coin risks $50 → can buy 2 coins
        num_coins = risk_amount / risk_per_coin
        
        # 4. Calculate position value
        position_value = num_coins * price
        
        # 5. Apply maximum position size limit (% of equity)
        max_position_value = equity * self.max_position_size_pct
        if position_value > max_position_value:
            logger.debug(f"Position value ${position_value:.2f} > max ${max_position_value:.2f}, capping")
            num_coins = max_position_value / price
            position_value = max_position_value
        
        # 6. Check minimum position value
        if position_value < self.min_position_value:
            return 0.0, f"BELOW_MIN_POSITION_VALUE_{position_value:.2f}"
        
        logger.debug(
            f"Position size calc: "
            f"equity=${equity:.2f}, "
            f"price=${price:.2f}, "
            f"risk_amount=${risk_amount:.2f}, "
            f"risk_per_coin=${risk_per_coin:.2f}, "
            f"size={num_coins:.6f}, "
            f"value=${position_value:.2f}"
        )
        
        return num_coins, "OK"
    
    def evaluate(self, 
                 signal: Union[int, float], 
                 candles: List[List[float]], 
                 equity: float, 
                 price: float, 
                 has_position: bool) -> Dict:
        """
        Evaluate trading decision with risk management.
        
        Args:
            signal: AI signal (higher = more bullish)
            candles: OHLCV data
            equity: Current account equity
            price: Current market price
            has_position: Whether we currently have a position
            
        Returns:
            Dictionary with action and details
        """
        # =====================
        # INITIALIZATION & VALIDATION
        # =====================
        result_template = {
            "action": "HOLD",
            "reason": "",
            "size": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "atr": 0.0,
            "emergency": False
        }
        
        # Track initial equity
        if self.initial_equity is None and equity > 0:
            self.initial_equity = equity
            logger.info(f"Initial equity set: ${equity:.2f}")
        
        # Validate inputs
        if not isinstance(signal, (int, float)):
            result_template["reason"] = "INVALID_SIGNAL_TYPE"
            return result_template
        
        if price <= 0:
            result_template["reason"] = "INVALID_PRICE"
            return result_template
        
        if equity <= 0:
            result_template["reason"] = "ZERO_EQUITY"
            return result_template
        
        # Validate candles
        candles_valid, candles_reason = self.validate_candles(candles)
        if not candles_valid:
            result_template["reason"] = candles_reason
            return result_template
        
        # =====================
        # DRAWDOWN PROTECTION
        # =====================
        # Initialize peak equity
        if self.peak_equity is None:
            self.peak_equity = equity
        
        # Update peak equity
        if equity > self.peak_equity:
            self.peak_equity = equity
            logger.debug(f"New peak equity: ${self.peak_equity:.2f}")
        
        # Calculate current drawdown
        if self.peak_equity > 0:
            current_dd = (self.peak_equity - equity) / self.peak_equity
        else:
            current_dd = 0.0
        
        logger.debug(f"Drawdown: {current_dd*100:.1f}% (max: {self.max_dd*100:.0f}%)")
        
        # Check max drawdown breach
        if current_dd >= self.max_dd:
            logger.warning(f"Max drawdown breached: {current_dd*100:.1f}% >= {self.max_dd*100:.0f}%")
            
            # Force close position if any
            if has_position:
                result_template.update({
                    "action": "SELL",
                    "reason": "MAX_DRAWDOWN_BREACH",
                    "size": self.position.size if hasattr(self.position, 'size') else 0.0,
                    "emergency": True
                })
                return result_template
            
            result_template["reason"] = "MAX_DRAWDOWN_BREACH"
            return result_template
        
        # =====================
        # EXIT MANAGEMENT
        # =====================
        if has_position:
            # Check if position state is consistent
            if not hasattr(self.position, 'size') or self.position.size <= 0:
                logger.warning("Position state inconsistent with has_position flag")
                result_template["reason"] = "POSITION_STATE_INCONSISTENT"
                return result_template
            
            # Update position price for trailing stop
            self.position.update_price(price, self.trail_pct)
            
            # Check for exit signals
            exit_now, exit_reason = self.position.should_exit(price)
            
            # Check position timeout
            if hasattr(self.position, 'time_in_position'):
                time_in_pos = self.position.time_in_position()
                if time_in_pos > self.max_position_duration_seconds:
                    exit_now = True
                    exit_reason = "POSITION_TIMEOUT"
                    logger.info(f"Position timeout after {time_in_pos/3600:.1f}h")
            
            if exit_now:
                result_template.update({
                    "action": "SELL",
                    "reason": exit_reason,
                    "size": self.position.size,
                    "emergency": False
                })
                return result_template
            
            # Still in position, hold
            result_template["reason"] = "IN_POSITION"
            
            # Log position status
            if hasattr(self.position, 'calculate_pnl'):
                pnl_value, pnl_percent = self.position.calculate_pnl(price)
                logger.debug(f"In position: P&L ${pnl_value:.2f} ({pnl_percent:.2f}%)")
            
            return result_template
        
        # =====================
        # ENTRY MANAGEMENT (No position)
        # =====================
        # Check if we have an open position in PositionState (safety check)
        if hasattr(self.position, 'has_position') and self.position.has_position():
            logger.warning("PositionState indicates open position but has_position=False")
            result_template["reason"] = "POSITION_STATE_MISMATCH"
            return result_template
        
        # Check buy signal threshold
        if signal < self.buy_threshold:
            result_template["reason"] = f"SIGNAL_BELOW_THRESHOLD_{signal:.2f}"
            return result_template
        
        # Calculate ATR for volatility-based stops
        atr = self.calculate_atr(candles)
        if atr is None or math.isnan(atr) or atr <= 0:
            result_template["reason"] = "INVALID_ATR"
            return result_template
        
        logger.debug(f"ATR: ${atr:.2f} ({atr/price*100:.1f}%)")
        
        # Calculate stop loss and take profit (for LONG position)
        stop_loss = price - (atr * self.sl_multiplier)
        take_profit = price + (atr * self.tp_multiplier)
        
        # Ensure stop loss is reasonable
        min_sl_distance = price * 0.005  # Minimum 0.5% stop loss
        if (price - stop_loss) < min_sl_distance:
            stop_loss = price - min_sl_distance
            logger.debug(f"Adjusted stop loss to minimum: ${stop_loss:.2f}")
        
        # Ensure stop loss is not below zero
        if stop_loss <= 0:
            stop_loss = price * 0.95  # 5% stop loss as fallback
            logger.warning(f"Stop loss adjusted to 5%: ${stop_loss:.2f}")
        
        # Calculate position size
        size, size_reason = self.calculate_position_size(equity, price, stop_loss)
        
        if size <= 0:
            result_template["reason"] = f"INVALID_SIZE_{size_reason}"
            return result_template
        
        # Check if position value meets minimum
        position_value = size * price
        if position_value < self.min_position_value:
            result_template["reason"] = f"BELOW_MIN_POSITION_VALUE_{position_value:.2f}"
            return result_template
        
        # Prepare entry decision
        result_template.update({
            "action": "BUY",
            "reason": "ENTRY_SIGNAL",
            "size": size,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "atr": atr,
            "risk_per_trade": self.risk_per_trade,
            "position_value": position_value
        })
        
        logger.info(
            f"BUY Signal: "
            f"size={size:.6f}, "
            f"price=${price:.2f}, "
            f"value=${position_value:.2f}, "
            f"SL=${stop_loss:.2f}, "
            f"TP=${take_profit:.2f}, "
            f"risk={self.risk_per_trade*100:.1f}%"
        )
        
        # Note: We don't call self.position.open() here because:
        # 1. The trade might not execute
        # 2. Position should be opened by OrderManager after successful trade
        # 3. This prevents state inconsistency
        
        return result_template
    
    def on_position_opened(self, price: float, size: float, stop_loss: float, take_profit: float) -> bool:
        """
        Update risk engine when a position is successfully opened.
        
        Args:
            price: Entry price
            size: Position size
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            True if position registered successfully
        """
        try:
            if hasattr(self.position, 'open'):
                self.position.open(price, size, stop_loss, take_profit)
                self.last_trade_time = time.time() if 'time' in globals() else None
                logger.info(f"Position opened: {size:.6f} @ ${price:.2f}")
                return True
            else:
                logger.error("PositionState does not have 'open' method")
                return False
        except Exception as e:
            logger.error(f"Failed to register position opening: {e}")
            return False
    
    def on_position_closed(self) -> bool:
        """Update risk engine when position is closed."""
        try:
            if hasattr(self.position, 'reset'):
                # Log closing info if available
                if hasattr(self.position, 'size') and self.position.size > 0:
                    logger.info(f"Position closed: size={self.position.size:.6f}")
                
                self.position.reset()
                self.last_trade_time = None
                return True
            else:
                logger.error("PositionState does not have 'reset' method")
                return False
        except Exception as e:
            logger.error(f"Failed to reset position: {e}")
            return False
    
    def allow_training(self, equity: float) -> bool:
        """Check if conditions allow for model training."""
        if self.peak_equity is None or self.peak_equity <= 0:
            return True
        
        if equity <= 0:
            return False
        
        current_dd = (self.peak_equity - equity) / self.peak_equity
        allow = current_dd < (self.max_dd * 0.5)  # Only train if DD less than half of max
        
        logger.debug(f"Training allowed: {allow} (DD: {current_dd*100:.1f}%)")
        return allow
    
    def get_stats(self) -> Dict:
        """Get risk engine statistics."""
        stats = {
            "max_drawdown": self.max_dd,
            "risk_per_trade": self.risk_per_trade,
            "peak_equity": self.peak_equity,
            "initial_equity": self.initial_equity,
            "has_position": False,
            "position_size": 0.0
        }
        
        # Add position info if available
        if hasattr(self.position, 'has_position'):
            stats["has_position"] = self.position.has_position()
            
            if stats["has_position"] and hasattr(self.position, 'size'):
                stats["position_size"] = self.position.size
        
        # Calculate current drawdown if we have peak equity
        if self.peak_equity and self.peak_equity > 0 and hasattr(self, '_last_equity'):
            stats["current_drawdown"] = (self.peak_equity - self._last_equity) / self.peak_equity
        
        return stats
    
    def __str__(self) -> str:
        """String representation."""
        stats = self.get_stats()
        return (f"RiskEngine("
                f"max_dd={self.max_dd*100:.0f}%, "
                f"risk={self.risk_per_trade*100:.1f}%, "
                f"peak_equity=${stats.get('peak_equity', 0):.2f}, "
                f"has_position={stats.get('has_position', False)})")