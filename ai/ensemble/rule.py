# ai/ensemble/rule.py
import numpy as np
from .base import BaseModel
import logging

logger = logging.getLogger(__name__)

class RuleBased(BaseModel):
    def __init__(self):
        super().__init__()
        self.name = "RuleBased"
        self.last_decision_info = {}
        
    def decide(self, state):
        """
        Simple but effective rule-based strategy
        state: normalized prices + cash_ratio + position_ratio
        """
        try:
            # Extract components from state
            prices = state[:-2]  # Normalized prices
            cash_ratio = state[-2]
            position_ratio = state[-1]
            
            # Need minimum data
            if len(prices) < 20:
                self.last_decision_info = {
                    'decision': 1,
                    'reason': f'Insufficient data: {len(prices)} < 20'
                }
                return 1  # HOLD
            
            # Calculate simple indicators
            current_price = prices[-1]
            
            # Short-term momentum (last 5 vs last 10)
            if len(prices) >= 10:
                momentum_5 = np.mean(prices[-5:])
                momentum_10 = np.mean(prices[-10:])
                momentum_signal = 1 if momentum_5 > momentum_10 else -1
            else:
                momentum_signal = 0
            
            # Volatility indicator
            recent_volatility = np.std(prices[-10:]) if len(prices) >= 10 else 0
            
            # Simple trend (price vs 20-period MA)
            ma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else 1.0
            trend_signal = current_price - ma_20
            
            # Decision logic - SIMPLE but EFFECTIVE
            decision = 1  # Default HOLD
            
            # BUY conditions (aggressive)
            # 1. Price above MA (uptrend)
            # 2. Positive momentum
            # 3. Have cash
            if trend_signal > 0.005 and momentum_signal > 0 and cash_ratio > 0.05:
                decision = 0  # BUY
                reason = f"Uptrend (+{trend_signal:.3f}), positive momentum, cash available"
            
            # SELL conditions
            # 1. Price below MA (downtrend)
            # 2. Have position
            elif trend_signal < -0.005 and position_ratio > 0.00001:
                decision = 2  # SELL
                reason = f"Downtrend ({trend_signal:.3f}), has position"
            
            else:
                reason = f"No clear signal (trend={trend_signal:.3f}, momentum={momentum_signal}, cash={cash_ratio:.2f})"
            
            self.last_decision_info = {
                'decision': decision,
                'reason': reason,
                'current_price': float(current_price),
                'ma_20': float(ma_20),
                'trend_signal': float(trend_signal),
                'momentum_signal': momentum_signal,
                'cash_ratio': float(cash_ratio),
                'position_ratio': float(position_ratio)
            }
            
            logger.info(f"RuleBased: {reason}")
            
            return decision
            
        except Exception as e:
            error_msg = f"RuleBased error: {str(e)}"
            logger.error(error_msg)
            self.last_decision_info = {
                'decision': 1,
                'reason': error_msg
            }
            return 1  # HOLD on error
    
    def get_last_decision_info(self):
        return self.last_decision_info