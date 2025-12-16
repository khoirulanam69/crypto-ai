from collections import defaultdict

class EnsembleAggregator:
    def __init__(self, models, threshold=0.6):
        self.models = models
        self.threshold = threshold

    def decide(self, state):
        votes = defaultdict(float)

        for m in self.models:
            action, conf = m.predict(state)
            votes[action] += conf * m.weight

        total = sum(votes.values())
        if total == 0:
            return 0

        best_action = max(votes, key=votes.get)
        confidence = votes[best_action] / total

        if confidence < self.threshold:
            return 0

        return best_action
