# ai/ensemble/rule.py
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RuleBased:
    """Simple rule-based trading model."""
    
    def __init__(self):
        self.name = "RuleBased"
        self.last_decision_info = {}
        
    def predict(self, state):
        """
        Make a trading decision based on state.
        Compatible with EnsembleAggregator's expected interface.
        
        Args:
            state: Input state array (normalized prices + cash_ratio + position_ratio)
            
        Returns:
            tuple: (action, confidence) where:
                action: 0=BUY, 1=HOLD, 2=SELL
                confidence: float between 0.0 and 1.0
        """
        try:
            # Extract components from state
            # state = [normalized_prices..., cash_ratio, position_ratio]
            prices = state[:-2]  # Normalized prices
            cash_ratio = state[-2]
            position_ratio = state[-1]
            
            # Need minimum data
            if len(prices) < 20:
                self.last_decision_info = {
                    'decision': 1,
                    'reason': f'Insufficient data: {len(prices)} < 20',
                    'confidence': 0.1
                }
                return 1, 0.1  # HOLD with low confidence
            
            # Calculate simple indicators
            current_price = prices[-1]
            
            # Short-term momentum (last 5 vs last 10)
            if len(prices) >= 10:
                momentum_5 = np.mean(prices[-5:])
                momentum_10 = np.mean(prices[-10:])
                momentum_signal = 1 if momentum_5 > momentum_10 else -1
            else:
                momentum_signal = 0
            
            # Simple trend (price vs 20-period MA)
            ma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else 1.0
            trend_signal = current_price - ma_20
            
            # Calculate confidence based on trend strength
            confidence = min(0.9, abs(trend_signal) * 10)  # Convert to 0-0.9 range
            
            # Decision logic
            action = 1  # Default HOLD
            reason = ""
            
            # BUY conditions
            if trend_signal > 0.002 and cash_ratio > 0.05:
                action = 0  # BUY
                confidence = max(confidence, 0.6)  # Minimum confidence for BUY
                reason = f"Uptrend (+{trend_signal:.4f}), cash available ({cash_ratio:.2f})"
            
            # SELL conditions
            elif trend_signal < -0.002 and position_ratio > 0.000001:
                action = 2  # SELL
                confidence = max(confidence, 0.6)  # Minimum confidence for SELL
                reason = f"Downtrend ({trend_signal:.4f}), has position ({position_ratio:.8f})"
            
            else:
                confidence = 0.3  # Low confidence for HOLD
                if cash_ratio <= 0.05:
                    reason = f"Not enough cash ({cash_ratio:.2f} <= 0.05)"
                elif position_ratio <= 0.000001:
                    reason = f"No significant position ({position_ratio:.8f})"
                else:
                    reason = f"No clear trend (trend={trend_signal:.4f})"
            
            # Ensure confidence is valid
            confidence = max(0.1, min(0.9, confidence))
            
            self.last_decision_info = {
                'decision': action,
                'reason': reason,
                'confidence': confidence,
                'current_price': float(current_price),
                'ma_20': float(ma_20),
                'trend_signal': float(trend_signal),
                'momentum_signal': momentum_signal,
                'cash_ratio': float(cash_ratio),
                'position_ratio': float(position_ratio),
                'prices_length': len(prices)
            }
            
            action_name = ['BUY', 'HOLD', 'SELL'][action]
            logger.info(f"RuleBased: {action_name} (conf={confidence:.2f}) - {reason}")
            
            return action, confidence
            
        except Exception as e:
            error_msg = f"RuleBased error: {str(e)}"
            logger.error(error_msg)
            self.last_decision_info = {
                'decision': 1,
                'reason': error_msg,
                'confidence': 0.1
            }
            return 1, 0.1  # HOLD with low confidence on error
    
    def decide(self, state):
        """
        Alias for predict() to maintain backward compatibility.
        """
        action, confidence = self.predict(state)
        return action
    
    def get_last_decision_info(self):
        """Get debug info about last decision."""
        return self.last_decision_info