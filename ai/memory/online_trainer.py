# ai/memory/online_trainer.py
import os
from stable_baselines3 import PPO
from ai.memory.replay_buffer import ReplayBuffer

MODEL = os.getenv("MODEL", "models/ppo_live.zip")
FINE_TUNE_STEPS = int(os.getenv("FINE_TUNE_STEPS", "512"))

def fine_tune():
    if not os.path.exists(MODEL):
        print("[AI] Model not found, skip fine-tune")
        return

    buffer = ReplayBuffer.load()

    if len(buffer) < 100:
        print("[AI] Replay buffer too small, skip fine-tune")
        return

    print("[AI] Online fine-tuning started...")

    model = PPO.load(MODEL)

    model.learn(
        total_timesteps=FINE_TUNE_STEPS,
        reset_num_timesteps=False
    )

    model.save(MODEL)

    print("[AI] Online fine-tuning finished")
