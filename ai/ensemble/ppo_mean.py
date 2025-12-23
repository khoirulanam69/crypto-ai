import os
import numpy as np
from typing import Tuple, Optional, Union, List
import logging
from stable_baselines3 import PPO

logger = logging.getLogger(__name__)

class PPOMean:
    """
    PPO model wrapper for ensemble trading.
    
    Loads a pre-trained PPO model and provides predictions with confidence estimates.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 deterministic: bool = True,
                 default_confidence: float = 0.7):
        """
        Initialize PPO model wrapper.
        
        Args:
            model_path: Path to .zip model file (default: from PPO_MEAN_MODEL env)
            deterministic: Whether to use deterministic actions
            default_confidence: Default confidence if cannot estimate
        """
        # Get model path
        if model_path is None:
            model_path = os.getenv("PPO_MEAN_MODEL")
        
        if not model_path:
            raise ValueError(
                "Model path not provided and PPO_MEAN_MODEL environment variable not set"
            )
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Store parameters
        self.deterministic = deterministic
        self.default_confidence = min(1.0, max(0.0, default_confidence))
        self.model_path = model_path
        
        # Load model
        self._load_model()
        
        # Warm-up: make a dummy prediction to initialize
        self._warm_up()
        
        logger.info(f"PPOMean initialized: {model_path}, deterministic={deterministic}")
    
    def _load_model(self) -> None:
        """Load PPO model with error handling."""
        try:
            self.model = PPO.load(self.model_path, verbose=0)
            
            # Check if model has required attributes
            if not hasattr(self.model, 'predict'):
                raise AttributeError("Loaded model does not have predict method")
            
            logger.debug(f"Model loaded successfully: {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load PPO model from {self.model_path}: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _warm_up(self) -> None:
        """Make a dummy prediction to initialize model (JIT compilation, etc.)"""
        try:
            # Create dummy state (same shape as expected)
            dummy_state = np.zeros((1, 50), dtype=np.float32)  # Adjust based on your state size
            
            # Warm-up prediction
            action, _ = self.model.predict(dummy_state, deterministic=self.deterministic)
            logger.debug(f"Model warm-up successful, action shape: {np.shape(action)}")
            
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")
    
    def _validate_state(self, state: np.ndarray) -> bool:
        """
        Validate input state.
        
        Returns:
            True if state is valid
        """
        # Check for NaN/Inf
        if np.any(np.isnan(state)):
            logger.warning("State contains NaN values")
            return False
        
        if np.any(np.isinf(state)):
            logger.warning("State contains infinite values")
            return False
        
        # Check shape
        if state.ndim not in [1, 2]:
            logger.warning(f"Invalid state dimensions: {state.ndim}")
            return False
        
        # Check for reasonable values (assuming normalized)
        if np.any(np.abs(state) > 10):  # Arbitrary threshold
            logger.warning(f"State values out of expected range: min={state.min():.2f}, max={state.max():.2f}")
            # Don't return False for this, just warn
        
        return True
    
    def _estimate_confidence(self, 
                            state: np.ndarray, 
                            action: Union[int, np.ndarray]) -> float:
        """
        Estimate prediction confidence.
        
        Args:
            state: Input state
            action: Predicted action
            
        Returns:
            Confidence estimate between 0 and 1
        """
        try:
            # Method 1: Try to get action probabilities from policy
            if hasattr(self.model.policy, 'get_distribution'):
                # Convert state to tensor
                import torch
                state_tensor = torch.FloatTensor(state)
                
                # Get distribution
                dist = self.model.policy.get_distribution(state_tensor)
                
                if hasattr(dist, 'probs'):
                    # Categorical distribution
                    probs = dist.probs.detach().numpy()
                    confidence = np.max(probs)
                    return float(confidence)
                elif hasattr(dist, 'mean'):
                    # Gaussian distribution - estimate confidence from std
                    std = dist.stddev.detach().numpy()
                    # Lower std = higher confidence
                    confidence = 1.0 / (1.0 + np.mean(std))
                    return float(confidence)
            
            # Method 2: Use model's built-in method if available
            if hasattr(self.model, 'predict_proba'):
                action_probs = self.model.predict_proba(state)
                confidence = np.max(action_probs)
                return float(confidence)
            
            # Method 3: Multiple predictions to check consistency
            if not self.deterministic:
                predictions = []
                for _ in range(5):
                    pred, _ = self.model.predict(state, deterministic=False)
                    predictions.append(pred)
                
                consistency = np.mean([p == action for p in predictions])
                return float(consistency)
            
        except Exception as e:
            logger.debug(f"Confidence estimation failed: {e}")
        
        # Fallback: return default confidence
        return self.default_confidence
    
    def predict(self, 
                state: Union[np.ndarray, List[float]]) -> Tuple[int, float]:
        """
        Make prediction from state.
        
        Args:
            state: Input state array or list
            
        Returns:
            Tuple of (action, confidence)
            
        Raises:
            ValueError: If state is invalid
            RuntimeError: If prediction fails
        """
        try:
            # Convert to numpy array
            state_arr = np.asarray(state, dtype=np.float32)
            
            # Ensure correct shape for single sample
            if state_arr.ndim == 1:
                state_arr = state_arr.reshape(1, -1)
            elif state_arr.ndim == 2:
                if state_arr.shape[0] != 1:
                    logger.warning(f"Multiple states provided ({state_arr.shape[0]}), using first")
                    state_arr = state_arr[0:1, :]
            else:
                raise ValueError(f"Invalid state dimensions: {state_arr.ndim}")
            
            # Validate state
            if not self._validate_state(state_arr):
                logger.warning("State validation failed, returning safe prediction")
                return 0, 0.1  # HOLD with low confidence
            
            # Make prediction
            action, _ = self.model.predict(
                state_arr, 
                deterministic=self.deterministic
            )
            
            # Convert action to scalar if needed
            if isinstance(action, np.ndarray):
                if action.size == 1:
                    action = int(action.item())
                else:
                    # Multi-dimensional action, take first
                    action = int(action.flat[0])
            
            # Estimate confidence
            confidence = self._estimate_confidence(state_arr, action)
            
            # Ensure confidence is in valid range
            confidence = max(0.0, min(1.0, confidence))
            
            logger.debug(f"PPOMean prediction: action={action}, confidence={confidence:.3f}")
            
            return int(action), float(confidence)
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            
            # Safe fallback: return HOLD with low confidence
            return 0, 0.1
    
    def batch_predict(self, 
                     states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions for multiple states.
        
        Args:
            states: Array of shape (n_samples, n_features)
            
        Returns:
            Tuple of (actions, confidences)
        """
        if states.ndim != 2:
            raise ValueError(f"States must be 2D, got shape {states.shape}")
        
        actions = []
        confidences = []
        
        for i in range(states.shape[0]):
            state = states[i:i+1, :]  # Keep as 2D
            action, confidence = self.predict(state)
            actions.append(action)
            confidences.append(confidence)
        
        return np.array(actions), np.array(confidences)
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        info = {
            "model_type": "PPO",
            "model_path": self.model_path,
            "deterministic": self.deterministic,
            "policy_type": type(self.model.policy).__name__ if hasattr(self.model, 'policy') else "Unknown",
            "action_space": str(self.model.action_space) if hasattr(self.model, 'action_space') else "Unknown",
            "observation_space": str(self.model.observation_space) if hasattr(self.model, 'observation_space') else "Unknown",
        }
        
        # Try to get more info
        try:
            if hasattr(self.model, 'num_timesteps'):
                info["training_steps"] = int(self.model.num_timesteps)
        except:
            pass
        
        return info
    
    def __str__(self) -> str:
        """String representation."""
        return f"PPOMean(model={os.path.basename(self.model_path)})"