import time
import logging
import json
from typing import Dict, Tuple, Optional, List, TypedDict
import threading

logger = logging.getLogger(__name__)

# Type definitions
class TradeRecord(TypedDict, total=False):
    timestamp: float
    side: str
    amount: float
    position_after: float
    trade_data: Dict[str, float]

class PositionTracker:
    """
    Thread-safe position tracker with trade history and persistence.
    
    Features:
    - Thread-safe operations
    - Trade history tracking
    - Position validation
    - Serialization for persistence
    - Performance metrics
    """
    
    def __init__(self, initial_position: float = 0.0):
        """
        Initialize position tracker.
        
        Args:
            initial_position: Starting position size (default 0)
        """
        self._position = float(initial_position)
        self._last_update = time.time()
        self._trade_history: List[TradeRecord] = []
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        
        logger.debug(f"PositionTracker initialized with position={initial_position}")
    
    def update_from_trade(self, trade: Dict[str, float]) -> Tuple[bool, str, float]:
        """
        Update position from a trade execution.
        
        Args:
            trade: Dictionary with 'side' ('buy'/'sell') and 'amount'
                   Can include optional fields: 'price', 'symbol', 'order_id'
            
        Returns:
            (success, message, executed_amount)
        """
        with self._lock:
            try:
                # Safe extraction and validation
                side = trade.get('side', '').lower()
                amount = trade.get('amount', 0)
                
                # Validation
                if side not in ['buy', 'sell']:
                    logger.warning(f"Invalid trade side: {side}")
                    return False, f"INVALID_SIDE: {side}", 0.0
                
                # Convert amount to float safely
                try:
                    amount = float(amount)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid amount type: {type(amount)}")
                    return False, f"INVALID_AMOUNT_TYPE: {type(amount)}", 0.0
                
                if amount <= 1e-10:  # Tiny threshold for floating point
                    logger.warning(f"Invalid amount value: {amount}")
                    return False, f"INVALID_AMOUNT_VALUE: {amount}", 0.0
                
                executed_amount = amount
                action = "BUY" if side == 'buy' else "SELL"
                
                if side == 'buy':
                    self._position += amount
                else:  # sell
                    # Cannot sell more than we have
                    if amount > self._position:
                        amount = self._position
                        if amount <= 1e-10:
                            logger.warning("No position to sell")
                            return False, "NO_POSITION_TO_SELL", 0.0
                        
                        executed_amount = amount
                        logger.info(f"Adjusted sell amount to {amount:.6f}")
                    
                    self._position -= amount
                
                # Ensure position is non-negative (floating point safety)
                if self._position < -1e-10:
                    # This shouldn't happen but just in case
                    logger.error(f"Position went negative: {self._position}")
                    self._position = 0.0
                    return False, "POSITION_WENT_NEGATIVE", 0.0
                
                # Record trade in history
                trade_record: TradeRecord = {
                    'timestamp': time.time(),
                    'side': side,
                    'amount': executed_amount,
                    'position_after': self._position,
                    'trade_data': trade.copy()  # Keep original trade data
                }
                self._trade_history.append(trade_record)
                
                # Keep history manageable (last 1000 trades)
                if len(self._trade_history) > 1000:
                    removed = len(self._trade_history) - 1000
                    self._trade_history = self._trade_history[-1000:]
                    logger.debug(f"Trimmed {removed} old trades from history")
                
                self._last_update = time.time()
                
                logger.info(
                    f"Position updated: {action} {executed_amount:.6f}, "
                    f"new_position={self._position:.6f}"
                )
                
                return True, f"{action}_EXECUTED", executed_amount
                
            except Exception as e:
                logger.error(f"Position update failed: {e}", exc_info=True)
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
        """
        Reset position to specified value (default 0).
        
        Args:
            new_position: New position value
        """
        with self._lock:
            old_position = self._position
            self._position = float(new_position)
            self._last_update = time.time()
            
            logger.info(
                f"Position reset: {old_position:.6f} -> {new_position:.6f}"
            )
    
    def clear_history(self) -> None:
        """Clear all trade history."""
        with self._lock:
            count = len(self._trade_history)
            self._trade_history.clear()
            logger.info(f"Cleared {count} trade records from history")
    
    def get_trade_history(self, limit: Optional[int] = None) -> List[TradeRecord]:
        """
        Get trade history.
        
        Args:
            limit: Maximum number of trades to return (most recent first)
            
        Returns:
            List of trade records
        """
        with self._lock:
            if limit is None:
                return [trade.copy() for trade in self._trade_history]
            return [trade.copy() for trade in self._trade_history[-limit:]]
    
    def get_last_trade(self) -> Optional[TradeRecord]:
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
            
            stats = {
                'current_position': self._position,
                'last_update': self._last_update,
                'total_trades': len(self._trade_history),
                'total_buys': total_buys,
                'total_sells': total_sells,
                'total_buy_volume': buy_volume,
                'total_sell_volume': sell_volume,
                'net_volume': buy_volume - sell_volume,
                'age_seconds': time.time() - self._last_update if self._last_update else 0,
                'has_position': self._position > 1e-10
            }
            
            # Add trade frequency if we have enough history
            if len(self._trade_history) >= 2:
                first_timestamp = self._trade_history[0]['timestamp']
                last_timestamp = self._trade_history[-1]['timestamp']
                duration = last_timestamp - first_timestamp
                
                if duration > 0:
                    stats['trades_per_hour'] = len(self._trade_history) / (duration / 3600)
                    stats['avg_time_between_trades'] = duration / len(self._trade_history)
            
            return stats
    
    def get_performance_metrics(self) -> Dict:
        """
        Calculate trading performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        with self._lock:
            if len(self._trade_history) < 2:
                return {'status': 'INSUFFICIENT_DATA'}
            
            stats = self.get_stats()
            
            # Calculate holding period metrics if we have position
            metrics = {
                'total_trades': stats['total_trades'],
                'buy_sell_ratio': stats['total_buys'] / max(1, stats['total_sells']),
                'avg_trade_size': (stats['total_buy_volume'] + stats['total_sell_volume']) / max(1, stats['total_trades']),
                'net_exposure': stats['net_volume'],
                'current_position': self._position
            }
            
            # Add trade frequency metrics
            if 'trades_per_hour' in stats:
                metrics.update({
                    'trades_per_hour': stats['trades_per_hour'],
                    'avg_time_between_trades': stats['avg_time_between_trades']
                })
            
            return metrics
    
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
                logger.error(
                    f"Position mismatch: calculated={calculated_position:.6f}, "
                    f"current={self._position:.6f}, diff={diff:.6f}"
                )
                return False, f"POSITION_MISMATCH: diff={diff:.6f}"
            
            return True, "CONSISTENT"
    
    def to_dict(self) -> Dict:
        """Serialize position tracker state to dictionary."""
        with self._lock:
            return {
                'position': self._position,
                'last_update': self._last_update,
                'trade_history': self._trade_history.copy(),
                'stats': self.get_stats(),
                'metadata': {
                    'class_name': self.__class__.__name__,
                    'timestamp': time.time()
                }
            }
    
    def save_to_file(self, filepath: str) -> bool:
        """
        Save position tracker state to JSON file.
        
        Args:
            filepath: Path to save file
            
        Returns:
            True if successful
        """
        try:
            data = self.to_dict()
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Position tracker saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save position tracker: {e}")
            return False
    
    @classmethod
    def load_from_file(cls, filepath: str) -> Optional['PositionTracker']:
        """
        Load position tracker from JSON file.
        
        Args:
            filepath: Path to load file
            
        Returns:
            PositionTracker instance or None if failed
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            tracker = cls(initial_position=float(data.get('position', 0.0)))
            tracker._last_update = float(data.get('last_update', time.time()))
            tracker._trade_history = data.get('trade_history', [])
            
            logger.info(f"Position tracker loaded from {filepath}")
            return tracker
            
        except Exception as e:
            logger.error(f"Failed to load position tracker: {e}")
            return None
    
    def should_stop_trading(self, 
                           max_trades_per_hour: float = 60.0,
                           min_time_between_trades: float = 5.0) -> Tuple[bool, str]:
        """
        Check if trading should be paused based on recent activity.
        
        Args:
            max_trades_per_hour: Maximum allowed trades per hour
            min_time_between_trades: Minimum seconds between trades
            
        Returns:
            (should_stop, reason)
        """
        with self._lock:
            if len(self._trade_history) < 3:
                return False, "INSUFFICIENT_HISTORY"
            
            # Check recent trade frequency
            recent_trades = self._trade_history[-3:]
            timestamps = [t['timestamp'] for t in recent_trades]
            
            # Check time between recent trades
            for i in range(len(timestamps) - 1):
                time_diff = timestamps[i+1] - timestamps[i]
                if time_diff < min_time_between_trades:
                    return True, f"TRADES_TOO_FAST: {time_diff:.1f}s < {min_time_between_trades}s"
            
            # Check overall trade rate
            stats = self.get_stats()
            if 'trades_per_hour' in stats and stats['trades_per_hour'] > max_trades_per_hour:
                return True, f"TRADE_RATE_TOO_HIGH: {stats['trades_per_hour']:.1f}/h > {max_trades_per_hour}/h"
            
            return False, "OK"
    
    def __str__(self) -> str:
        """String representation."""
        with self._lock:
            stats = self.get_stats()
            has_pos = "✓" if self._position > 1e-10 else "✗"
            
            return (f"PositionTracker("
                    f"position={self._position:.6f}, "
                    f"has_position={has_pos}, "
                    f"trades={stats['total_trades']})")