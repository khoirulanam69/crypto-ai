# ai/memory/replay_buffer.py
import csv
import os
import time

class ReplayBuffer:
    def __init__(self, path="data/replay_buffer.csv", max_size=50000):
        self.path = path
        self.max_size = max_size
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "ts",
                    "price",
                    "action",
                    "reward",
                    "equity"
                ])

    def append(self, price, action, reward, equity):
        row = [
            int(time.time() * 1000),
            price,
            action,
            reward,
            equity
        ]

        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    def size(self):
        with open(self.path, "r") as f:
            return max(0, sum(1 for _ in f) - 1)
