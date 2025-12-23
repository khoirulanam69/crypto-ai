import time
from typing import Dict, Tuple, Optional, List
import threading

class PositionTracker:
    """
    Thread-safe position tracker with trade history.
    
    Tracks position size and maintains trade history for audit/debugging.
    """
    
    def __init__(self, initial_position: float = 0.0):
        """
        Initialize position tracker.
        
        Args:
            initial_position: Starting position size (default 0)
        """
        self._position = float(initial_position)
        self._last_update = time.time()
        self._trade_history: List[Dict] = []
        self._lock = threading.RLock()  # For thread safety
        
    def update_from_trade(self, trade: Dict[str, float]) -> Tuple[bool, str, float]:
        """
        Update position from a trade execution.
        
        Args:
            trade: Dictionary with 'side' ('buy'/'sell') and 'amount'
            
        Returns:
            (success, message, executed_amount)
            
        Raises:
            ValueError: If trade data is invalid
        """
        with self._lock:  # Thread safety
            try:
                # Safe extraction and validation
                side = trade.get('side', '').lower()
                amount = trade.get('amount', 0)
                
                # Validation
                if side not in ['buy', 'sell']:
                    return False, f"INVALID_SIDE: {side}", 0.0
                
                # Convert amount to float safely
                try:
                    amount = float(amount)
                except (ValueError, TypeError):
                    return False, f"INVALID_AMOUNT_TYPE: {type(amount)}", 0.0
                
                if amount <= 1e-10:  # Tiny threshold
                    return False, f"INVALID_AMOUNT_VALUE: {amount}", 0.0
                
                executed_amount = amount
                
                if side == 'buy':
                    self._position += amount
                    action = "BUY"
                    
                else:  # sell
                    # Cannot sell more than we have
                    if amount > self._position:
                        amount = self._position
                        if amount <= 1e-10:
                            return False, "NO_POSITION_TO_SELL", 0.0
                        
                        executed_amount = amount
                    
                    self._position -= amount
                    action = "SELL"
                
                # Ensure position is non-negative (floating point safety)
                if self._position < -1e-10:
                    # This shouldn't happen but just in case
                    self._position = 0.0
                    return False, "POSITION_WENT_NEGATIVE", 0.0
                
                # Record trade in history
                trade_record = {
                    'timestamp': time.time(),
                    'side': side,
                    'amount': executed_amount,
                    'position_after': self._position,
                    'trade_data': trade.copy()  # Keep original trade data
                }
                self._trade_history.append(trade_record)
                
                # Keep history manageable (last 1000 trades)
                if len(self._trade_history) > 1000:
                    self._trade_history = self._trade_history[-1000:]
                
                self._last_update = time.time()
                
                return True, f"{action}_EXECUTED", executed_amount
                
            except Exception as e:
                return False, f"UPDATE_ERROR: {str(e)}", 0.0
    
    def has_position(self, threshold: float = 1e-10) -> bool:
        """
        Check if we have a position above threshold.
        
        Args:
            threshold: Minimum position size to consider as 'having position'
            
        Returns:
            True if position > threshold
        """
        with self._lock:
            return self._position > threshold
    
    def position_size(self) -> float:
        """Get current position size."""
        with self._lock:
            return float(self._position)
    
    def reset(self, new_position: float = 0.0) -> None:
        """Reset position to specified value (default 0)."""
        with self._lock:
            self._position = float(new_position)
            self._last_update = time.time()
            # Optionally clear history or keep it
            # self._trade_history.clear()
    
    def get_trade_history(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Get trade history.
        
        Args:
            limit: Maximum number of trades to return (most recent first)
            
        Returns:
            List of trade records
        """
        with self._lock:
            if limit is None:
                return self._trade_history.copy()
            return self._trade_history[-limit:].copy()
    
    def get_last_trade(self) -> Optional[Dict]:
        """Get the most recent trade."""
        with self._lock:
            if not self._trade_history:
                return None
            return self._trade_history[-1].copy()
    
    def get_stats(self) -> Dict:
        """Get position statistics."""
        with self._lock:
            total_buys = sum(1 for t in self._trade_history if t['side'] == 'buy')
            total_sells = sum(1 for t in self._trade_history if t['side'] == 'sell')
            buy_volume = sum(t['amount'] for t in self._trade_history if t['side'] == 'buy')
            sell_volume = sum(t['amount'] for t in self._trade_history if t['side'] == 'sell')
            
            return {
                'current_position': self._position,
                'last_update': self._last_update,
                'total_trades': len(self._trade_history),
                'total_buys': total_buys,
                'total_sells': total_sells,
                'total_buy_volume': buy_volume,
                'total_sell_volume': sell_volume,
                'net_volume': buy_volume - sell_volume,
                'age_seconds': time.time() - self._last_update if self._last_update else 0
            }
    
    def validate(self) -> Tuple[bool, str]:
        """Validate position consistency with trade history."""
        with self._lock:
            if not self._trade_history:
                if abs(self._position) > 1e-10:
                    return False, f"POSITION_WITHOUT_HISTORY: {self._position}"
                return True, "EMPTY_VALID"
            
            # Recalculate position from history
            calculated_position = 0.0
            for trade in self._trade_history:
                if trade['side'] == 'buy':
                    calculated_position += trade['amount']
                else:  # sell
                    calculated_position -= trade['amount']
            
            diff = abs(calculated_position - self._position)
            if diff > 1e-8:  # Allow tiny floating point difference
                return False, f"POSITION_MISMATCH: calculated={calculated_position}, current={self._position}, diff={diff}"
            
            return True, "CONSISTENT"
    
    def __str__(self) -> str:
        """String representation."""
        with self._lock:
            stats = self.get_stats()
            return (f"PositionTracker("
                    f"position={self._position:.6f}, "
                    f"trades={stats['total_trades']}, "
                    f"last_update={time.ctime(self._last_update)})")