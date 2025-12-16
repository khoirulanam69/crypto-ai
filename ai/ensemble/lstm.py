import numpy as np
from .base import BaseModel

class LSTMPrice(BaseModel):
    name = "lstm"
    weight = 0.8

    def predict(self, state):
        trend = np.mean(state[-10:])

        if trend > 0:
            return 1, abs(trend)
        elif trend < 0:
            return 2, abs(trend)
        return 0, 0.1
    
    def decide(self, state):
        # netral dulu
        return 0
