import numpy as np
from .base import BaseModel

class RuleBased(BaseModel):
    name = "rule"
    weight = 0.6

    def predict(self, state):
        vol = np.std(state)
        if vol > 0.05:
            return 0, 0.9  # stay out
        return 0, 0.2
