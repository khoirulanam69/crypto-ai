import os
import time
import logging
import tempfile
import shutil
from typing import Optional, Dict, Any
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# Import your environment
# from trading_env import TradingEnv
from ai.memory.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)

class TrainingCallback(BaseCallback):
    """Callback for logging training progress."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.start_time = time.time()
    
    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:
            elapsed = time.time() - self.start_time
            logger.info(f"Training step {self.n_calls}, elapsed: {elapsed:.1f}s")
        return True

def fine_tune(
    model_path: Optional[str] = None,
    buffer_path: Optional[str] = None,
    fine_tune_steps: Optional[int] = None,
    learning_rate: float = 3e-4,
    min_buffer_size: int = 100,
    create_backup: bool = True
) -> Dict[str, Any]:
    """
    Fine-tune PPO model with experience replay.
    
    Args:
        model_path: Path to model file (default: from env var)
        buffer_path: Path to replay buffer (default: from env var)
        fine_tune_steps: Number of training steps (default: from env var)
        learning_rate: Learning rate for fine-tuning
        min_buffer_size: Minimum buffer size required
        create_backup: Whether to backup original model
        
    Returns:
        Dictionary with training results
    """
    # Configuration
    model_path = model_path or os.getenv("MODEL", "models/ppo_live.zip")
    buffer_path = buffer_path or os.getenv("REPLAY_BUFFER_PATH", "data/replay_buffer.csv")
    fine_tune_steps = fine_tune_steps or int(os.getenv("FINE_TUNE_STEPS", "512"))
    
    results = {
        "success": False,
        "error": None,
        "steps_trained": 0,
        "training_time": 0,
        "model_path": model_path,
        "backup_created": False
    }
    
    start_time = time.time()
    backup_path = None
    
    try:
        # =====================
        # 1. VALIDATE INPUTS
        # =====================
        if not os.path.exists(model_path):
            error_msg = f"Model not found: {model_path}"
            logger.error(f"[AI] {error_msg}")
            results["error"] = error_msg
            return results
        
        # =====================
        # 2. LOAD REPLAY BUFFER
        # =====================
        logger.info(f"[AI] Loading replay buffer from {buffer_path}")
        
        try:
            buffer = ReplayBuffer(path=buffer_path)
            buffer_size = buffer.size()
        except Exception as e:
            error_msg = f"Failed to load replay buffer: {e}"
            logger.error(f"[AI] {error_msg}")
            results["error"] = error_msg
            return results
        
        if buffer_size < min_buffer_size:
            error_msg = f"Replay buffer too small: {buffer_size} < {min_buffer_size}"
            logger.warning(f"[AI] {error_msg}")
            results["error"] = error_msg
            return results
        
        logger.info(f"[AI] Replay buffer size: {buffer_size}")
        
        # =====================
        # 3. LOAD MODEL
        # =====================
        logger.info(f"[AI] Loading model from {model_path}")
        
        try:
            model = PPO.load(model_path, verbose=0)
            
            # Update learning rate for fine-tuning
            model.learning_rate = learning_rate
            logger.info(f"[AI] Set learning rate to {learning_rate}")
            
        except Exception as e:
            error_msg = f"Failed to load model: {e}"
            logger.error(f"[AI] {error_msg}")
            results["error"] = error_msg
            return results
        
        # =====================
        # 4. CREATE BACKUP
        # =====================
        if create_backup:
            backup_path = f"{model_path}.backup_{int(time.time())}"
            try:
                shutil.copy2(model_path, backup_path)
                results["backup_created"] = True
                results["backup_path"] = backup_path
                logger.info(f"[AI] Created backup at {backup_path}")
            except Exception as e:
                logger.warning(f"[AI] Failed to create backup: {e}")
        
        # =====================
        # 5. SETUP ENVIRONMENT
        # =====================
        # NOTE: This requires your trading environment
        # Uncomment and implement based on your setup
        
        # from trading_env import TradingEnv
        # env = TradingEnv(...)
        # model.set_env(env)
        
        # For now, we'll check if environment is set
        if not hasattr(model, 'env') or model.env is None:
            error_msg = "Model does not have environment set. Cannot train."
            logger.error(f"[AI] {error_msg}")
            results["error"] = error_msg
            
            # Cleanup backup
            if backup_path and os.path.exists(backup_path):
                os.remove(backup_path)
            return results
        
        # =====================
        # 6. FINE-TUNE
        # =====================
        logger.info(f"[AI] Starting fine-tuning for {fine_tune_steps} steps")
        
        callback = TrainingCallback()
        
        try:
            model.learn(
                total_timesteps=fine_tune_steps,
                callback=callback,
                reset_num_timesteps=False,
                log_interval=50,
                tb_log_name="online_fine_tune"
            )
            
            results["steps_trained"] = fine_tune_steps
            
        except Exception as e:
            error_msg = f"Training failed: {e}"
            logger.error(f"[AI] {error_msg}")
            results["error"] = error_msg
            
            # Restore from backup if training failed
            if backup_path and os.path.exists(backup_path):
                logger.info(f"[AI] Restoring from backup: {backup_path}")
                shutil.copy2(backup_path, model_path)
            
            return results
        
        # =====================
        # 7. SAVE MODEL
        # =====================
        logger.info(f"[AI] Saving fine-tuned model to {model_path}")
        
        # Save to temp file first
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.temp') as tmp:
            temp_path = tmp.name
        
        try:
            model.save(temp_path)
            
            # Replace original
            os.replace(temp_path, model_path)
            logger.info(f"[AI] Model saved successfully")
            
        except Exception as e:
            error_msg = f"Failed to save model: {e}"
            logger.error(f"[AI] {error_msg}")
            results["error"] = error_msg
            
            # Try to restore from backup
            if backup_path and os.path.exists(backup_path):
                shutil.copy2(backup_path, model_path)
            
            return results
        
        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        # =====================
        # 8. CLEANUP
        # =====================
        if backup_path and os.path.exists(backup_path):
            os.remove(backup_path)
            logger.info(f"[AI] Removed backup: {backup_path}")
        
        # =====================
        # 9. RETURN RESULTS
        # =====================
        training_time = time.time() - start_time
        results.update({
            "success": True,
            "training_time": training_time,
            "buffer_size": buffer_size,
            "learning_rate": learning_rate,
            "final_model_size": os.path.getsize(model_path) if os.path.exists(model_path) else 0
        })
        
        logger.info(
            f"[AI] Fine-tuning completed: "
            f"{fine_tune_steps} steps in {training_time:.1f}s, "
            f"buffer={buffer_size}"
        )
        
        return results
        
    except Exception as e:
        error_msg = f"Unexpected error in fine_tune: {e}"
        logger.error(f"[AI] {error_msg}")
        results["error"] = error_msg
        
        # Final cleanup
        if backup_path and os.path.exists(backup_path):
            try:
                os.remove(backup_path)
            except:
                pass
        
        return results

# Backward-compatible wrapper
def fine_tune_simple():
    """Simple wrapper for backward compatibility."""
    results = fine_tune()
    
    if results["success"]:
        print("[AI] Online fine-tuning finished")
    else:
        print(f"[AI] Online fine-tuning failed: {results.get('error', 'Unknown error')}")
    
    return results["success"]