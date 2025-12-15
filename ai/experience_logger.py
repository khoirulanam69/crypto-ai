import csv
import os
import time
import json

class ExperienceLogger:
    def __init__(self, path=None):
        if path is None:
            base = os.path.expanduser("~/.crypto_ai")
            path = os.path.join(base, "experience.csv")

        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "ts", "obs", "action", "reward", "next_obs", "done"
                ])

    def log(self, obs, action, reward, next_obs, done):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                int(time.time() * 1000),
                json.dumps(obs.tolist()),
                action,
                reward,
                json.dumps(next_obs.tolist()),
                int(done)
            ])
