# ai/ensemble/aggregator.py
class EnsembleAggregator:
    def __init__(self, models):
        self.models = models

    def decide(self, state):
        votes = {0: 0.0, 1: 0.0, 2: 0.0}

        for m in self.models:
            action, confidence = m.predict(state)
            votes[action] += confidence

        # ambil action dengan bobot terbesar
        return max(votes, key=votes.get)
