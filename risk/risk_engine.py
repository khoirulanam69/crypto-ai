from executor.position_state import PositionState
import numpy as np
import math

class RiskEngine:
    def __init__(self, max_dd=0.2, risk_per_trade=0.01, trail_pct=0.005,
                 sl_multiplier=1.5, tp_multiplier=3.0, atr_period=14,
                 max_position_size=0.1, buy_threshold=0.7):
        
        self.max_dd = max_dd
        self.risk_per_trade = risk_per_trade
        self.trail_pct = trail_pct
        self.sl_multiplier = sl_multiplier
        self.tp_multiplier = tp_multiplier
        self.atr_period = atr_period
        self.max_position_size = max_position_size  # Max % of equity in one position
        self.buy_threshold = buy_threshold
        
        self.position = PositionState()
        self.peak_equity = None
        self.initial_equity = None

    def calculate_atr(self, candles):
        """Calculate ATR without pandas dependency"""
        if len(candles) < self.atr_period + 1:
            return None
        
        true_ranges = []
        for i in range(1, len(candles)):
            high = candles[i][2]
            low = candles[i][3]
            prev_close = candles[i-1][4]
            
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            true_ranges.append(max(tr1, tr2, tr3))
        
        # Simple Moving Average of TR
        atr_values = []
        for i in range(len(true_ranges) - self.atr_period + 1):
            window = true_ranges[i:i + self.atr_period]
            atr_values.append(sum(window) / self.atr_period)
        
        return atr_values[-1] if atr_values else None

    def validate_candles(self, candles, min_candles=20):
        """Validate OHLCV data format"""
        if not candles:
            return False, "NO_CANDLE_DATA"
        
        if len(candles) < min_candles:
            return False, f"INSUFFICIENT_DATA_{len(candles)}"
        
        for i, candle in enumerate(candles[-min_candles:]):
            if len(candle) != 6:
                return False, f"INVALID_FORMAT_AT_{i}"
            
            # Check numeric values (skip timestamp)
            for val in candle[1:]:
                if not isinstance(val, (int, float)):
                    return False, f"NON_NUMERIC_AT_{i}"
                
                # Check for negative prices (impossible)
                if val < 0 and candle.index(val) in [1, 2, 3, 4]:  # OHLC
                    return False, f"NEGATIVE_PRICE_AT_{i}"
        
        return True, "OK"

    def calculate_position_size(self, equity, price, stop_loss_price):
        """Calculate position size with proper risk management"""
        # 1. Risk amount based on equity
        risk_amount = equity * self.risk_per_trade
        
        # 2. Risk per coin (distance from entry to stop)
        if stop_loss_price <= 0:
            return 0, "INVALID_STOP_LOSS"
        
        risk_distance = abs(price - stop_loss_price)
        
        # 3. Risk per coin as percentage
        if price <= 0:
            return 0, "INVALID_PRICE"
        
        risk_percentage = risk_distance / price
        
        # 4. Position size calculation (CORRECTED)
        if risk_percentage > 0:
            position_value = risk_amount / risk_percentage
            size = position_value / price
        else:
            size = 0
        
        # 5. Apply max position size limit
        max_size = (equity * self.max_position_size) / price
        size = min(size, max_size)
        
        return size, "OK"

    def evaluate(self, signal, candles, equity, price, has_position):
        # =====================
        # INITIALIZATION & VALIDATION
        # =====================
        if self.initial_equity is None and equity > 0:
            self.initial_equity = equity
        
        # Validate inputs
        if not isinstance(signal, (int, float)):
            return {"action": "HOLD", "reason": "INVALID_SIGNAL_TYPE"}
        
        if price <= 0:
            return {"action": "HOLD", "reason": "INVALID_PRICE"}
        
        if equity <= 0:
            return {"action": "HOLD", "reason": "ZERO_EQUITY"}
        
        # Validate candles
        candles_valid, candles_reason = self.validate_candles(candles)
        if not candles_valid:
            return {"action": "HOLD", "reason": candles_reason}
        
        # =====================
        # DRAWDOWN CHECK
        # =====================
        if self.peak_equity is None:
            self.peak_equity = equity
        
        self.peak_equity = max(self.peak_equity, equity)
        
        if self.peak_equity > 0:
            current_dd = (self.peak_equity - equity) / self.peak_equity
        else:
            current_dd = 0
        
        if current_dd >= self.max_dd:
            # Force close position if any
            if has_position:
                return {
                    "action": "SELL",
                    "reason": "MAX_DRAWDOWN_BREACH",
                    "size": self.position.size if hasattr(self.position, 'size') else None,
                    "emergency": True
                }
            return {"action": "HOLD", "reason": "MAX_DRAWDOWN_BREACH"}
        
        # =====================
        # EXIT MANAGEMENT
        # =====================
        if has_position and hasattr(self.position, 'should_exit'):
            self.position.update_price(price, self.trail_pct)
            exit_now, reason = self.position.should_exit(price)
            
            if exit_now:
                return {
                    "action": "SELL",
                    "reason": reason,
                    "size": self.position.size,
                    "emergency": False
                }
            
            return {"action": "HOLD", "reason": "IN_POSITION"}
        
        # =====================
        # ENTRY MANAGEMENT
        # =====================
        # Check buy signal threshold
        if signal < self.buy_threshold:
            return {"action": "HOLD", "reason": f"SIGNAL_BELOW_THRESHOLD_{signal:.2f}"}
        
        # Calculate ATR for stop loss
        atr = self.calculate_atr(candles)
        if atr is None or math.isnan(atr) or atr <= 0:
            return {"action": "HOLD", "reason": "INVALID_ATR"}
        
        # Calculate stop loss and take profit
        stop_loss = price - (atr * self.sl_multiplier)
        take_profit = price + (atr * self.tp_multiplier)
        
        # Ensure stop loss is reasonable (not too close)
        min_sl_distance = price * 0.005  # Min 0.5% stop loss
        if (price - stop_loss) < min_sl_distance:
            stop_loss = price - min_sl_distance
        
        # Calculate position size
        size, size_reason = self.calculate_position_size(equity, price, stop_loss)
        
        if size <= 0:
            return {"action": "HOLD", "reason": f"INVALID_SIZE_{size_reason}"}
        
        # Check minimum position size
        min_position_value = 10  # $10 minimum
        if size * price < min_position_value:
            return {"action": "HOLD", "reason": "BELOW_MIN_POSITION_VALUE"}
        
        # Register position (if PositionState supports it)
        if hasattr(self.position, 'open'):
            self.position.open(price, size, stop_loss, take_profit)
        
        return {
            "action": "BUY",
            "size": size,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "atr": atr,
            "risk_per_trade": self.risk_per_trade,
            "position_value": size * price
        }
    
    def on_position_closed(self):
        """Reset position state"""
        if hasattr(self.position, 'reset'):
            self.position.reset()
        # Reset peak equity to current for new position cycle
        # self.peak_equity = None  # Optional
    
    def allow_training(self, equity):
        """Check if conditions allow for model training"""
        if self.peak_equity is None or self.peak_equity <= 0:
            return True
        
        if equity <= 0:
            return False
        
        current_dd = (self.peak_equity - equity) / self.peak_equity
        return current_dd < (self.max_dd * 0.5)  # Only train if DD less than half of max