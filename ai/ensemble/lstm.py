import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
import os
import logging

logger = logging.getLogger(__name__)

class SimpleLSTM(nn.Module):
    """Simple LSTM for time series classification."""
    
    def __init__(self, input_dim=50, hidden_dim=32, output_dim=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]  # Take last timestep
        logits = self.fc(last_out)
        return logits

class LSTMPrice:
    """Wrapper for LSTM trading model."""
    
    def __init__(self):
        model_path = os.getenv("LSTM_MODEL_PATH", "models/lstm_model.pth")
        
        # Create model architecture
        self.model = SimpleLSTM(
            input_dim=int(os.getenv("LSTM_INPUT_DIM", "50")),
            hidden_dim=int(os.getenv("LSTM_HIDDEN_DIM", "32")),
            output_dim=3  # HOLD, BUY, SELL
        )
        
        # Load weights if available
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                logger.info(f"LSTM model loaded from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load LSTM weights: {e}")
        else:
            logger.warning(f"LSTM model not found at {model_path}, using random weights")
        
        # Set to evaluation mode
        self.model.eval()
    
    def predict(self, state: np.ndarray) -> Tuple[int, float]:
        """Make prediction."""
        try:
            # Prepare input
            state_tensor = self._prepare_input(state)
            
            # Forward pass
            with torch.no_grad():
                logits = self.model(state_tensor)
                probs = torch.softmax(logits, dim=-1)
            
            # Get prediction
            action = torch.argmax(probs, dim=-1).item()
            confidence = torch.max(probs).item()
            
            return int(action), float(confidence)
            
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            return 0, 0.1
    
    def _prepare_input(self, state: np.ndarray) -> torch.Tensor:
        """Prepare state for LSTM input."""
        state_arr = np.array(state, dtype=np.float32)
        
        # Reshape to (batch=1, seq_len=1, features)
        if state_arr.ndim == 1:
            state_arr = state_arr.reshape(1, 1, -1)
        elif state_arr.ndim == 2:
            state_arr = state_arr.reshape(1, state_arr.shape[0], state_arr.shape[1])
        
        return torch.FloatTensor(state_arr)