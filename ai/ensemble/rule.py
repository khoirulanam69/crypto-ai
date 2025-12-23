import numpy as np
from typing import Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)

class RuleBased:
    """
    Advanced rule-based model combining multiple technical indicators.
    """
    
    def __init__(self):
        self.indicators = {
            'ma_crossover': {'weight': 0.4, 'fast': 10, 'slow': 30},
            'rsi': {'weight': 0.3, 'period': 14, 'oversold': 30, 'overbought': 70},
            'bollinger': {'weight': 0.3, 'period': 20, 'std_dev': 2},
            'momentum': {'weight': 0.2, 'period': 10}
        }
        self.action_history = []
    
    def calculate_indicators(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate all technical indicators."""
        indicators = {}
        
        # MA Crossover
        fast_ma = np.mean(prices[-self.indicators['ma_crossover']['fast']:])
        slow_ma = np.mean(prices[-self.indicators['ma_crossover']['slow']:])
        indicators['ma_signal'] = 1 if fast_ma > slow_ma else -1
        
        # RSI
        if len(prices) >= self.indicators['rsi']['period'] + 1:
            deltas = np.diff(prices[-self.indicators['rsi']['period']-1:])
            gains = deltas[deltas > 0]
            losses = -deltas[deltas < 0]
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 1e-10
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            indicators['rsi'] = rsi
        else:
            indicators['rsi'] = 50
        
        # Bollinger Bands
        if len(prices) >= self.indicators['bollinger']['period']:
            ma = np.mean(prices[-self.indicators['bollinger']['period']:])
            std = np.std(prices[-self.indicators['bollinger']['period']:])
            upper = ma + (std * self.indicators['bollinger']['std_dev'])
            lower = ma - (std * self.indicators['bollinger']['std_dev'])
            current_price = prices[-1]
            
            if current_price > upper:
                indicators['bollinger_signal'] = -1  # Overbought
            elif current_price < lower:
                indicators['bollinger_signal'] = 1   # Oversold
            else:
                indicators['bollinger_signal'] = 0   # Neutral
        
        return indicators
    
    def predict(self, state: np.ndarray) -> Tuple[int, float]:
        """Make prediction using multiple indicators."""
        if len(state) < 30:  # Need minimum data
            return 0, 0.1
        
        price_data = state[:-2]
        
        # Calculate indicators
        indicators = self.calculate_indicators(price_data)
        
        # Weighted voting
        buy_score = 0
        sell_score = 0
        total_weight = 0
        
        # MA Crossover rule
        if 'ma_signal' in indicators:
            weight = self.indicators['ma_crossover']['weight']
            total_weight += weight
            if indicators['ma_signal'] > 0:
                buy_score += weight
            else:
                sell_score += weight
        
        # RSI rule
        if 'rsi' in indicators:
            weight = self.indicators['rsi']['weight']
            total_weight += weight
            rsi = indicators['rsi']
            
            if rsi < self.indicators['rsi']['oversold']:
                buy_score += weight * (1 - rsi/self.indicators['rsi']['oversold'])
            elif rsi > self.indicators['rsi']['overbought']:
                sell_score += weight * (rsi/self.indicators['rsi']['overbought'] - 1)
        
        # Bollinger Bands rule
        if 'bollinger_signal' in indicators:
            weight = self.indicators['bollinger']['weight']
            total_weight += weight
            signal = indicators['bollinger_signal']
            
            if signal > 0:
                buy_score += weight
            elif signal < 0:
                sell_score += weight
        
        # Determine action
        if total_weight > 0:
            buy_ratio = buy_score / total_weight
            sell_ratio = sell_score / total_weight
            
            # Threshold for action
            threshold = 0.3
            
            if buy_ratio > sell_ratio and buy_ratio > threshold:
                action = 1  # BUY
                confidence = min(0.9, buy_ratio)
            elif sell_ratio > buy_ratio and sell_ratio > threshold:
                action = 2  # SELL
                confidence = min(0.9, sell_ratio)
            else:
                action = 0  # HOLD
                confidence = 0.3
        else:
            action = 0
            confidence = 0.1
        
        # Log decision
        logger.debug(f"RuleBased: action={action}, confidence={confidence:.2f}, "
                    f"buy_score={buy_score:.2f}, sell_score={sell_score:.2f}")
        
        return action, confidence