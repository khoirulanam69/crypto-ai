class BaseModel:
    name = "base"
    weight = 1.0

    def predict(self, state):
        """
        return: (action, confidence)
        action: 0 hold | 1 buy | 2 sell
        confidence: 0.0 – 1.0
        """
        raise NotImplementedError
