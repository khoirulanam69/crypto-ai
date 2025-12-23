import os
import numpy as np
from typing import Tuple
import logging
from stable_baselines3 import PPO

logger = logging.getLogger(__name__)

class PPOTrend:
    """PPO model specialized for trend following."""
    
    def __init__(self):
        model_path = os.getenv("PPO_TREND_MODEL")
        if not model_path:
            raise ValueError("PPO_TREND_MODEL environment variable not set")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"PPO_TREND_MODEL not found: {model_path}")
        
        try:
            self.model = PPO.load(model_path, verbose=0)
            logger.info(f"PPOTrend model loaded: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load PPOTrend model: {e}")
            raise
    
    def predict(self, state) -> Tuple[int, float]:
        """
        Predict with trend-specific preprocessing.
        
        Args:
            state: np.ndarray shape (window+2,) - normalized prices + cash_ratio + position
            
        Returns:
            (action, confidence)
        """
        try:
            # Preprocess with trend-specific logic
            state_processed = self._preprocess_for_trend(state)
            
            # Validate
            if np.any(np.isnan(state_processed)) or np.any(np.isinf(state_processed)):
                logger.warning("PPOTrend: Invalid state, returning HOLD")
                return 0, 0.1
            
            # Predict
            action, _ = self.model.predict(state_processed, deterministic=True)
            
            # Convert to scalar
            if isinstance(action, np.ndarray):
                action = int(action.item())
            
            # Get trend-specific confidence
            confidence = self._get_trend_confidence(state_processed, action)
            
            logger.debug(f"PPOTrend prediction: action={action}, confidence={confidence:.3f}")
            
            return int(action), float(confidence)
            
        except Exception as e:
            logger.error(f"PPOTrend prediction failed: {e}")
            return 0, 0.1  # Safe fallback
    
    def _preprocess_for_trend(self, state):
        """
        Add trend-specific features before prediction.
        
        Could add:
        - Trend indicators (moving average slopes)
        - Rate of change
        - Volatility measures
        """
        state_arr = np.array(state, dtype=np.float32)
        
        # Basic preprocessing
        if state_arr.ndim == 1:
            state_arr = state_arr.reshape(1, -1)
        
        # Extract price data (assuming last 2 elements are cash_ratio and position)
        if state_arr.shape[1] > 2:
            price_data = state_arr[0, :-2]
            
            # Calculate trend features (example)
            if len(price_data) >= 10:
                # 1. Short-term vs long-term trend
                short_window = min(5, len(price_data))
                long_window = min(20, len(price_data))
                
                short_ma = np.mean(price_data[-short_window:])
                long_ma = np.mean(price_data[-long_window:])
                trend_strength = short_ma - long_ma
                
                # 2. Rate of change
                if len(price_data) >= 2:
                    roc = (price_data[-1] - price_data[-2]) / price_data[-2]
                else:
                    roc = 0.0
                
                # Append trend features to state
                # Note: This changes state dimension - model must be trained with this!
                # For now, just use original state
                pass
        
        return state_arr
    
    def _get_trend_confidence(self, state, action) -> float:
        """Get confidence based on trend strength."""
        try:
            # Method 1: Actual model confidence
            if hasattr(self.model.policy, 'get_distribution'):
                import torch
                state_tensor = torch.FloatTensor(state)
                dist = self.model.policy.get_distribution(state_tensor)
                
                if hasattr(dist, 'probs'):
                    probs = dist.probs.detach().numpy()
                    base_confidence = float(np.max(probs))
                    
                    # Adjust based on trend clarity
                    trend_clarity = self._calculate_trend_clarity(state)
                    adjusted_confidence = base_confidence * (0.5 + 0.5 * trend_clarity)
                    
                    return min(1.0, adjusted_confidence)
        
        except Exception as e:
            logger.debug(f"Trend confidence estimation failed: {e}")
        
        # Fallback: calculate confidence from trend strength
        trend_clarity = self._calculate_trend_clarity(state)
        confidence = 0.3 + 0.4 * trend_clarity  # 0.3-0.7 range based on trend
        
        return confidence
    
    def _calculate_trend_clarity(self, state) -> float:
        """Calculate how clear the trend is (0=no trend, 1=strong trend)."""
        try:
            # Extract price data
            if state.shape[1] > 2:
                price_data = state[0, :-2]
                
                if len(price_data) >= 10:
                    # Calculate multiple moving averages
                    windows = [5, 10, 20]
                    ma_values = []
                    
                    for window in windows:
                        if len(price_data) >= window:
                            ma = np.mean(price_data[-window:])
                            ma_values.append(ma)
                    
                    # Check alignment of MAs (trend clarity)
                    if len(ma_values) >= 2:
                        # All MAs should be in same order for clear trend
                        is_uptrend = all(ma_values[i] <= ma_values[i+1] for i in range(len(ma_values)-1))
                        is_downtrend = all(ma_values[i] >= ma_values[i+1] for i in range(len(ma_values)-1))
                        
                        if is_uptrend or is_downtrend:
                            # Calculate strength
                            trend_strength = abs(ma_values[-1] - ma_values[0]) / (ma_values[0] + 1e-10)
                            return min(1.0, trend_strength * 5)  # Scale to 0-1
            
        except:
            pass
        
        return 0.5  # Neutral
    
    def get_model_type(self) -> str:
        """Get model type description."""
        return "PPO Trend Follower"