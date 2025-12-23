from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)

class EnsembleAggregator:
    """
    Robust ensemble voting system for combining multiple AI trading models.
    
    Features:
    - Weighted voting
    - Confidence thresholds
    - Tie-breaking mechanisms
    - Comprehensive error handling
    - Detailed logging
    """
    
    # Action mappings (convention)
    ACTION_HOLD = 0
    ACTION_BUY = 1
    ACTION_SELL = 2
    
    ACTION_NAMES = {
        ACTION_HOLD: "HOLD",
        ACTION_BUY: "BUY",
        ACTION_SELL: "SELL"
    }
    
    def __init__(self, 
                 models: List[Any], 
                 weights: Optional[List[float]] = None,
                 default_action: int = ACTION_HOLD,
                 confidence_threshold: float = 0.0,
                 tie_preference: str = "conservative"):
        """
        Initialize ensemble aggregator.
        
        Args:
            models: List of model objects with predict() method
            weights: Optional weights for each model (default: equal weights)
            default_action: Action to return if voting fails
            confidence_threshold: Minimum average confidence to accept vote
            tie_preference: How to handle ties - "conservative" (prefer HOLD), 
                           "aggressive" (prefer BUY/SELL), or "random"
        """
        if not models:
            raise ValueError("Models list cannot be empty")
        
        self.models = models
        self.weights = weights or [1.0] * len(models)
        self.default_action = default_action
        self.confidence_threshold = max(0.0, min(1.0, confidence_threshold))
        self.tie_preference = tie_preference
        
        # Validate weights
        if len(self.weights) != len(self.models):
            raise ValueError(f"Weights length {len(self.weights)} != models length {len(self.models)}")
        
        # Normalize weights
        weight_sum = sum(self.weights)
        if weight_sum > 0:
            self.weights = [w / weight_sum for w in self.weights]
        
        logger.info(f"Ensemble initialized with {len(models)} models")
    
    def decide(self, state: np.ndarray) -> int:
        """
        Make decision based on ensemble voting.
        
        Args:
            state: Input state array for models
            
        Returns:
            Integer action code
        """
        if not isinstance(state, (np.ndarray, list)):
            logger.error(f"Invalid state type: {type(state)}")
            return self.default_action
        
        votes: Dict[int, float] = {}
        model_results = []
        total_confidence = 0.0
        active_models = 0
        
        # Collect votes from all models
        for i, (model, weight) in enumerate(zip(self.models, self.weights)):
            try:
                # Get model prediction
                result = model.predict(state)
                
                # Parse result (support multiple formats)
                action, confidence = self._parse_model_result(result, i)
                
                # Validate action
                if not isinstance(action, int):
                    logger.warning(f"Model {i} returned non-integer action: {type(action)}")
                    continue
                
                # Validate confidence
                confidence = max(0.0, min(1.0, float(confidence)))
                
                # Apply model weight
                weighted_confidence = confidence * weight
                
                # Record vote
                votes[action] = votes.get(action, 0.0) + weighted_confidence
                total_confidence += confidence
                active_models += 1
                
                # Store for debugging
                model_results.append({
                    'model_index': i,
                    'action': action,
                    'action_name': self.ACTION_NAMES.get(action, f"UNKNOWN({action})"),
                    'confidence': confidence,
                    'weight': weight,
                    'weighted_confidence': weighted_confidence
                })
                
            except Exception as e:
                logger.warning(f"Model {i} failed: {e}")
                model_results.append({
                    'model_index': i,
                    'error': str(e)
                })
                continue
        
        # Log model results for debugging
        if logger.isEnabledFor(logging.DEBUG):
            for result in model_results:
                if 'error' in result:
                    logger.debug(f"Model {result['model_index']}: ERROR - {result['error']}")
                else:
                    logger.debug(
                        f"Model {result['model_index']}: {result['action_name']} "
                        f"(conf={result['confidence']:.3f}, weight={result['weight']:.3f})"
                    )
        
        # Check if we have any valid votes
        if not votes:
            logger.warning("No valid votes from any model")
            return self.default_action
        
        # Check confidence threshold
        if active_models > 0:
            avg_confidence = total_confidence / active_models
            if avg_confidence < self.confidence_threshold:
                logger.info(f"Average confidence {avg_confidence:.3f} below threshold {self.confidence_threshold}")
                return self.default_action
        
        # Find winning action(s)
        max_vote = max(votes.values())
        winning_actions = [action for action, vote in votes.items() if vote == max_vote]
        
        # Log voting summary
        vote_summary = ", ".join([f"{self.ACTION_NAMES.get(a, a)}: {votes[a]:.3f}" for a in sorted(votes.keys())])
        logger.info(f"Vote results: {vote_summary}")
        
        # Handle single winner
        if len(winning_actions) == 1:
            winner = winning_actions[0]
            logger.info(f"Winner: {self.ACTION_NAMES.get(winner, winner)} with {votes[winner]:.3f} votes")
            return winner
        
        # Handle tie
        logger.info(f"Tie between actions: {winning_actions}")
        winner = self._break_tie(winning_actions, votes)
        logger.info(f"Tie broken: {self.ACTION_NAMES.get(winner, winner)}")
        
        return winner
    
    def _parse_model_result(self, result: Any, model_index: int) -> Tuple[int, float]:
        """
        Parse model result in various formats.
        
        Args:
            result: Model prediction result
            model_index: Index of model for logging
            
        Returns:
            (action, confidence)
        """
        # Format 1: Tuple (action, confidence)
        if isinstance(result, tuple) and len(result) >= 2:
            return int(result[0]), float(result[1])
        
        # Format 2: Single action (assume confidence=1.0)
        elif isinstance(result, (int, float, np.integer, np.floating)):
            return int(result), 1.0
        
        # Format 3: Dict with action and confidence keys
        elif isinstance(result, dict):
            action = result.get('action', result.get('prediction', 0))
            confidence = result.get('confidence', result.get('prob', 1.0))
            return int(action), float(confidence)
        
        # Format 4: Numpy array (argmax for classification)
        elif isinstance(result, np.ndarray):
            if result.ndim == 1:
                action = np.argmax(result)
                confidence = np.max(result)
                return int(action), float(confidence)
            elif result.ndim == 2 and result.shape[0] == 1:
                action = np.argmax(result[0])
                confidence = np.max(result[0])
                return int(action), float(confidence)
        
        # Unknown format
        raise ValueError(f"Unknown result format from model {model_index}: {type(result)}")
    
    def _break_tie(self, tied_actions: List[int], votes: Dict[int, float]) -> int:
        """
        Break tie between actions with equal votes.
        
        Args:
            tied_actions: List of tied actions
            votes: Complete votes dictionary
            
        Returns:
            Selected action
        """
        # Remove default action from tie if possible
        if len(tied_actions) > 1 and self.default_action in tied_actions:
            tied_actions.remove(self.default_action)
        
        if len(tied_actions) == 1:
            return tied_actions[0]
        
        # Apply tie-breaking preference
        if self.tie_preference == "conservative":
            # Prefer HOLD, then SELL, then BUY (safest to most aggressive)
            for preferred in [self.ACTION_HOLD, self.ACTION_SELL, self.ACTION_BUY]:
                if preferred in tied_actions:
                    return preferred
        
        elif self.tie_preference == "aggressive":
            # Prefer BUY, then SELL, then HOLD (most to least aggressive)
            for preferred in [self.ACTION_BUY, self.ACTION_SELL, self.ACTION_HOLD]:
                if preferred in tied_actions:
                    return preferred
        
        # "random" or default: random choice
        import random
        return random.choice(tied_actions)
    
    def get_model_count(self) -> int:
        """Get number of models in ensemble."""
        return len(self.models)
    
    def get_weights(self) -> List[float]:
        """Get current model weights."""
        return self.weights.copy()
    
    def update_weight(self, model_index: int, new_weight: float) -> bool:
        """Update weight for a specific model."""
        if 0 <= model_index < len(self.weights):
            self.weights[model_index] = max(0.0, new_weight)
            # Re-normalize
            weight_sum = sum(self.weights)
            if weight_sum > 0:
                self.weights = [w / weight_sum for w in self.weights]
            return True
        return False
    
    def __str__(self) -> str:
        """String representation."""
        return f"EnsembleAggregator(models={len(self.models)}, threshold={self.confidence_threshold:.2f})"