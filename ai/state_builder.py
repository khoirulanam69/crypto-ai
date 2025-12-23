# import numpy as np
# import pandas as pd
# from typing import Dict, Tuple, Optional, Union, List
# import logging
# from dataclasses import dataclass

# logger = logging.getLogger(__name__)

# @dataclass
# class StateConfig:
#     """Configuration for state building."""
#     window_size: int = 20
#     volatility_window: int = 10
#     normalize: bool = True
#     clip_returns: float = 0.5  # Clip returns at ±50%
#     add_time_features: bool = True
#     add_technical_indicators: bool = True
#     price_column: str = "close"
#     volume_column: Optional[str] = "volume"

# def validate_dataframe(df: pd.DataFrame, 
#                       config: StateConfig) -> Tuple[bool, str]:
#     """
#     Validate input DataFrame.
    
#     Returns:
#         (is_valid, error_message)
#     """
#     if df is None:
#         return False, "DataFrame cannot be None"
    
#     if not isinstance(df, pd.DataFrame):
#         return False, f"Expected DataFrame, got {type(df)}"
    
#     if len(df) < config.window_size:
#         return False, f"Insufficient rows: {len(df)} < {config.window_size}"
    
#     # Check required columns
#     if config.price_column not in df.columns:
#         available = list(df.columns)
#         return False, f"Price column '{config.price_column}' not found. Available: {available}"
    
#     if config.volume_column and config.volume_column not in df.columns:
#         logger.warning(f"Volume column '{config.volume_column}' not found")
#         config.volume_column = None
    
#     # Check for NaN/inf in price
#     price_data = df[config.price_column]
#     if price_data.isnull().any():
#         return False, f"Price column contains NaN values"
    
#     if np.isinf(price_data).any():
#         return False, f"Price column contains infinite values"
    
#     # Check for zero or negative prices
#     if (price_data <= 0).any():
#         return False, f"Price column contains non-positive values"
    
#     return True, "OK"

# def build_state(df: pd.DataFrame, 
#                 config: Optional[StateConfig] = None) -> np.ndarray:
#     """
#     Build state vector from price data.
    
#     Args:
#         df: DataFrame with OHLCV data
#         config: State building configuration
        
#     Returns:
#         State vector as numpy array
#     """
#     # Default config
#     if config is None:
#         config = StateConfig()
    
#     # Validate input
#     is_valid, error_msg = validate_dataframe(df, config)
#     if not is_valid:
#         raise ValueError(f"Invalid DataFrame: {error_msg}")
    
#     try:
#         # Extract price data
#         price_series = df[config.price_column]
        
#         # 1. Calculate returns
#         returns = price_series.pct_change().fillna(0)
        
#         # Clip extreme returns (e.g., >50% moves are outliers)
#         returns_clipped = np.clip(returns, -config.clip_returns, config.clip_returns)
        
#         # 2. Calculate volatility
#         # Use at least min(volatility_window, available_data)
#         vol_window = min(config.volatility_window, len(returns))
#         volatility = returns.rolling(window=vol_window, min_periods=1).std().fillna(0)
        
#         # 3. Get last window_size values
#         returns_window = returns_clipped.values[-config.window_size:]
#         vol_window_vals = volatility.values[-config.window_size:]
        
#         # Ensure correct length
#         if len(returns_window) < config.window_size:
#             returns_window = np.pad(
#                 returns_window,
#                 (config.window_size - len(returns_window), 0),
#                 'constant',
#                 constant_values=0
#             )
        
#         if len(vol_window_vals) < config.window_size:
#             vol_window_vals = np.pad(
#                 vol_window_vals,
#                 (config.window_size - len(vol_window_vals), 0),
#                 'constant', 
#                 constant_values=0
#             )
        
#         # Start building state
#         state_parts = []
        
#         # 4. Basic features
#         state_parts.append(returns_window)
#         state_parts.append(vol_window_vals)
        
#         # 5. Normalize if requested
#         if config.normalize:
#             # Returns already clipped to [-clip_returns, clip_returns]
#             returns_normalized = returns_window / config.clip_returns
            
#             # Normalize volatility (0 to 1)
#             vol_max = vol_window_vals.max()
#             if vol_max > 0:
#                 vol_normalized = vol_window_vals / vol_max
#             else:
#                 vol_normalized = vol_window_vals
            
#             # Replace in state parts
#             state_parts[0] = returns_normalized
#             state_parts[1] = vol_normalized
        
#         # 6. Add technical indicators if requested
#         if config.add_technical_indicators:
#             tech_features = _calculate_technical_indicators(
#                 df, config.window_size, config.price_column
#             )
#             state_parts.extend(tech_features)
        
#         # 7. Add time features if requested
#         if config.add_time_features and 'timestamp' in df.columns:
#             time_features = _extract_time_features(df, config.window_size)
#             state_parts.extend(time_features)
        
#         # 8. Add volume features if available
#         if config.volume_column and config.volume_column in df.columns:
#             volume_features = _calculate_volume_features(df, config)
#             state_parts.extend(volume_features)
        
#         # 9. Concatenate all features
#         state = np.concatenate(state_parts)
        
#         # Final validation
#         if np.any(np.isnan(state)):
#             logger.warning("State contains NaN values, replacing with 0")
#             state = np.nan_to_num(state, nan=0.0)
        
#         if np.any(np.isinf(state)):
#             logger.warning("State contains infinite values, replacing with 0")
#             state = np.nan_to_num(state, posinf=0.0, neginf=0.0)
        
#         logger.debug(f"Built state: shape={state.shape}, range=[{state.min():.3f}, {state.max():.3f}]")
        
#         return state.astype(np.float32)
        
#     except Exception as e:
#         logger.error(f"Failed to build state: {e}")
#         raise

# def _calculate_technical_indicators(df: pd.DataFrame, 
#                                    window_size: int,
#                                    price_col: str) -> List[np.ndarray]:
#     """Calculate basic technical indicators."""
#     features = []
#     price = df[price_col].values
    
#     # Need enough data for indicators
#     if len(price) >= window_size:
#         price_window = price[-window_size:]
        
#         # 1. Moving averages (normalized by current price)
#         for ma_period in [5, 10, 20]:
#             if len(price) >= ma_period:
#                 ma = np.mean(price[-ma_period:])
#                 # Normalize: (MA - current) / current
#                 current = price[-1]
#                 if current > 0:
#                     ma_normalized = (ma - current) / current
#                 else:
#                     ma_normalized = 0.0
                
#                 # Repeat for window (simplified - in reality should calculate for each point)
#                 ma_feature = np.full(window_size, ma_normalized, dtype=np.float32)
#                 features.append(ma_feature)
        
#         # 2. Price position in recent range
#         recent_prices = price[-window_size:]
#         if len(recent_prices) > 0:
#             price_min = np.min(recent_prices)
#             price_max = np.max(recent_prices)
#             price_range = price_max - price_min
            
#             if price_range > 0:
#                 # Current price position in range (0=bottom, 1=top)
#                 current_pos = (price[-1] - price_min) / price_range
#                 pos_feature = np.full(window_size, current_pos, dtype=np.float32)
#                 features.append(pos_feature)
        
#         # 3. Rate of Change (ROC)
#         if len(price) >= 5:
#             roc_5 = (price[-1] - price[-5]) / (price[-5] + 1e-10)
#             roc_feature = np.full(window_size, roc_5, dtype=np.float32)
#             features.append(roc_feature)
    
#     return features

# def _extract_time_features(df: pd.DataFrame, window_size: int) -> List[np.ndarray]:
#     """Extract time-based features."""
#     features = []
    
#     if 'timestamp' in df.columns:
#         # Get last window timestamps
#         timestamps = df['timestamp'].values[-window_size:]
        
#         if len(timestamps) > 0:
#             try:
#                 # Convert to datetime if not already
#                 if not np.issubdtype(timestamps.dtype, np.datetime64):
#                     import pandas as pd
#                     timestamps = pd.to_datetime(timestamps)
                
#                 # Extract hour of day (sine/cosine for circular nature)
#                 hours = np.array([ts.hour for ts in timestamps])
#                 hour_sin = np.sin(2 * np.pi * hours / 24)
#                 hour_cos = np.cos(2 * np.pi * hours / 24)
                
#                 features.append(hour_sin)
#                 features.append(hour_cos)
                
#                 # Day of week
#                 day_of_week = np.array([ts.weekday() for ts in timestamps])
#                 dow_sin = np.sin(2 * np.pi * day_of_week / 7)
#                 dow_cos = np.cos(2 * np.pi * day_of_week / 7)
                
#                 features.append(dow_sin)
#                 features.append(dow_cos)
                
#             except Exception as e:
#                 logger.debug(f"Failed to extract time features: {e}")
    
#     return features

# def _calculate_volume_features(df: pd.DataFrame, 
#                               config: StateConfig) -> List[np.ndarray]:
#     """Calculate volume-based features."""
#     features = []
    
#     if config.volume_column and config.volume_column in df.columns:
#         volume = df[config.volume_column].values
        
#         if len(volume) >= config.window_size:
#             volume_window = volume[-config.window_size:]
            
#             # 1. Volume normalized by recent average
#             vol_mean = np.mean(volume_window)
#             if vol_mean > 0:
#                 volume_normalized = volume_window / vol_mean
#                 features.append(volume_normalized)
            
#             # 2. Volume rate of change
#             if len(volume) >= 5:
#                 vol_roc = np.diff(volume[-5:]) / (volume[-6:-1] + 1e-10)
#                 # Pad to window size
#                 vol_roc_padded = np.zeros(config.window_size, dtype=np.float32)
#                 vol_roc_padded[-len(vol_roc):] = vol_roc[:config.window_size]
#                 features.append(vol_roc_padded)
    
#     return features