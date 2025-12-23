# import csv
# import os
# import time
# import json
# import threading
# import gzip
# import shutil
# from typing import Any, Union, Optional, Dict, List
# import logging
# from datetime import datetime
# import numpy as np

# logger = logging.getLogger(__name__)

# class ExperienceLogger:
#     """
#     Thread-safe experience logger with compression and rotation.
    
#     Efficiently logs (state, action, reward, next_state, done) tuples.
#     Supports batch writing, file rotation, and multiple formats.
#     """
    
#     # Supported data types for serialization
#     SUPPORTED_TYPES = (np.ndarray, list, tuple, int, float, str, bool, type(None))
    
#     def __init__(self, 
#                  path: Optional[str] = None,
#                  max_file_size_mb: int = 100,
#                  compress_old: bool = True,
#                  buffer_size: int = 100,
#                  format: str = "csv"):
#         """
#         Initialize experience logger.
        
#         Args:
#             path: Log file path (default: ~/.crypto_ai/experience_YYYYMMDD_HHMMSS.csv)
#             max_file_size_mb: Maximum file size before rotation (MB)
#             compress_old: Whether to compress rotated files
#             buffer_size: Number of experiences to buffer before writing
#             format: Output format - "csv" or "jsonl"
#         """
#         # Determine path
#         if path is None:
#             base_dir = os.getenv("EXPERIENCE_LOG_DIR", os.path.expanduser("~/.crypto_ai"))
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             path = os.path.join(base_dir, f"experience_{timestamp}.{format}")
        
#         self.path = path
#         self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
#         self.compress_old = compress_old
#         self.buffer_size = buffer_size
#         self.format = format.lower()
        
#         if self.format not in ["csv", "jsonl"]:
#             raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'jsonl'")
        
#         # Thread safety
#         self._lock = threading.RLock()
#         self._buffer: List[Dict] = []
        
#         # Create directory
#         os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        
#         # Initialize file
#         self._init_file()
        
#         # Statistics
#         self.stats = {
#             "total_logged": 0,
#             "total_rotations": 0,
#             "last_log_time": None,
#             "start_time": time.time()
#         }
        
#         logger.info(
#             f"ExperienceLogger initialized: "
#             f"path={path}, "
#             f"max_size={max_file_size_mb}MB, "
#             f"buffer={buffer_size}, "
#             f"format={format}"
#         )
    
#     def _init_file(self) -> None:
#         """Initialize log file with headers if needed."""
#         with self._lock:
#             if not os.path.exists(self.path):
#                 if self.format == "csv":
#                     with open(self.path, "w", newline="") as f:
#                         writer = csv.writer(f)
#                         writer.writerow([
#                             "timestamp",
#                             "episode_id",
#                             "step",
#                             "observation",
#                             "action", 
#                             "reward",
#                             "next_observation",
#                             "done",
#                             "info"
#                         ])
#                 else:  # jsonl
#                     # JSONL doesn't need header
#                     with open(self.path, "w") as f:
#                         pass  # Empty file
    
#     def _safe_serialize(self, data: Any) -> Any:
#         """
#         Safely serialize data for JSON storage.
        
#         Args:
#             data: Input data (numpy array, list, scalar, etc.)
            
#         Returns:
#             JSON-serializable data
#         """
#         # Handle None
#         if data is None:
#             return None
        
#         # Handle numpy arrays
#         if hasattr(data, 'tolist'):
#             try:
#                 return data.tolist()
#             except:
#                 pass
        
#         # Handle PyTorch tensors
#         if hasattr(data, 'detach'):
#             try:
#                 return data.detach().cpu().numpy().tolist()
#             except:
#                 pass
        
#         # Handle TensorFlow tensors
#         if hasattr(data, 'numpy'):
#             try:
#                 return data.numpy().tolist()
#             except:
#                 pass
        
#         # Handle scalars and lists
#         if isinstance(data, (int, float, str, bool)):
#             return data
#         elif isinstance(data, (list, tuple)):
#             return [self._safe_serialize(item) for item in data]
#         elif isinstance(data, dict):
#             return {str(k): self._safe_serialize(v) for k, v in data.items()}
#         else:
#             # Last resort: string representation
#             logger.warning(f"Cannot serialize type {type(data)}, converting to string")
#             return str(data)
    
#     def log(self, 
#             observation: Any,
#             action: Any,
#             reward: float,
#             next_observation: Any,
#             done: bool,
#             episode_id: Optional[int] = None,
#             step: Optional[int] = None,
#             info: Optional[Dict] = None) -> bool:
#         """
#         Log an experience tuple.
        
#         Args:
#             observation: Current state/observation
#             action: Action taken
#             reward: Reward received
#             next_observation: Next state/observation
#             done: Whether episode is done
#             episode_id: Optional episode identifier
#             step: Optional step number in episode
#             info: Optional additional info dictionary
            
#         Returns:
#             True if logged successfully
#         """
#         try:
#             # Generate timestamp
#             timestamp = int(time.time() * 1000)
            
#             # Default values
#             if episode_id is None:
#                 episode_id = int(time.time() // 3600)  # Hour-based episode
            
#             if step is None:
#                 step = self.stats["total_logged"]
            
#             if info is None:
#                 info = {}
            
#             # Prepare record
#             record = {
#                 "timestamp": timestamp,
#                 "episode_id": episode_id,
#                 "step": step,
#                 "observation": self._safe_serialize(observation),
#                 "action": self._safe_serialize(action),
#                 "reward": float(reward),
#                 "next_observation": self._safe_serialize(next_observation),
#                 "done": bool(done),
#                 "info": self._safe_serialize(info)
#             }
            
#             # Add to buffer
#             with self._lock:
#                 self._buffer.append(record)
                
#                 # Check if buffer needs flushing
#                 if len(self._buffer) >= self.buffer_size:
#                     self._flush_buffer()
                
#                 # Update stats
#                 self.stats["total_logged"] += 1
#                 self.stats["last_log_time"] = time.time()
            
#             return True
            
#         except Exception as e:
#             logger.error(f"Failed to log experience: {e}")
#             return False
    
#     def log_batch(self, experiences: List[tuple]) -> int:
#         """
#         Log multiple experiences at once.
        
#         Args:
#             experiences: List of (obs, action, reward, next_obs, done, episode_id, step, info) tuples
            
#         Returns:
#             Number of successfully logged experiences
#         """
#         success_count = 0
        
#         for exp in experiences:
#             # Unpack with defaults
#             if len(exp) >= 5:
#                 obs, action, reward, next_obs, done = exp[:5]
#                 episode_id = exp[5] if len(exp) > 5 else None
#                 step = exp[6] if len(exp) > 6 else None
#                 info = exp[7] if len(exp) > 7 else None
                
#                 if self.log(obs, action, reward, next_obs, done, episode_id, step, info):
#                     success_count += 1
        
#         return success_count
    
#     def _flush_buffer(self) -> None:
#         """Write buffered experiences to disk."""
#         if not self._buffer:
#             return
        
#         with self._lock:
#             try:
#                 # Check if file needs rotation
#                 if os.path.exists(self.path):
#                     file_size = os.path.getsize(self.path)
#                     if file_size > self.max_file_size_bytes:
#                         self._rotate_file()
                
#                 # Write buffer
#                 if self.format == "csv":
#                     self._write_csv_buffer()
#                 else:  # jsonl
#                     self._write_jsonl_buffer()
                
#                 # Clear buffer
#                 self._buffer.clear()
                
#                 logger.debug(f"Flushed {len(self._buffer)} experiences to disk")
                
#             except Exception as e:
#                 logger.error(f"Failed to flush buffer: {e}")
#                 # Keep buffer for retry
    
#     def _write_csv_buffer(self) -> None:
#         """Write buffer in CSV format."""
#         with open(self.path, "a", newline="", encoding="utf-8") as f:
#             writer = csv.writer(f)
            
#             for record in self._buffer:
#                 writer.writerow([
#                     record["timestamp"],
#                     record["episode_id"],
#                     record["step"],
#                     json.dumps(record["observation"]),
#                     json.dumps(record["action"]),
#                     record["reward"],
#                     json.dumps(record["next_observation"]),
#                     int(record["done"]),
#                     json.dumps(record["info"])
#                 ])
    
#     def _write_jsonl_buffer(self) -> None:
#         """Write buffer in JSON Lines format."""
#         with open(self.path, "a", encoding="utf-8") as f:
#             for record in self._buffer:
#                 json_line = json.dumps(record, separators=(",", ":"))
#                 f.write(json_line + "\n")
    
#     def _rotate_file(self) -> None:
#         """Rotate log file when it reaches max size."""
#         try:
#             # Create new filename with timestamp
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             base, ext = os.path.splitext(self.path)
#             archive_path = f"{base}_{timestamp}{ext}"
            
#             # Rename current file
#             if os.path.exists(self.path):
#                 shutil.move(self.path, archive_path)
#                 self.stats["total_rotations"] += 1
                
#                 logger.info(f"Rotated log file: {self.path} -> {archive_path}")
                
#                 # Compress if enabled
#                 if self.compress_old and os.path.exists(archive_path):
#                     self._compress_file(archive_path)
            
#             # Reinitialize file
#             self._init_file()
            
#         except Exception as e:
#             logger.error(f"Failed to rotate log file: {e}")
    
#     def _compress_file(self, filepath: str) -> None:
#         """Compress file with gzip."""
#         try:
#             compressed_path = filepath + ".gz"
            
#             with open(filepath, 'rb') as f_in:
#                 with gzip.open(compressed_path, 'wb') as f_out:
#                     shutil.copyfileobj(f_in, f_out)
            
#             # Remove original
#             os.remove(filepath)
            
#             logger.debug(f"Compressed log file: {compressed_path}")
            
#         except Exception as e:
#             logger.warning(f"Failed to compress file {filepath}: {e}")
    
#     def flush(self) -> None:
#         """Force flush buffer to disk."""
#         with self._lock:
#             if self._buffer:
#                 self._flush_buffer()
    
#     def get_stats(self) -> Dict:
#         """Get logger statistics."""
#         with self._lock:
#             stats = self.stats.copy()
            
#             # Current file size
#             if os.path.exists(self.path):
#                 stats["current_file_size_mb"] = os.path.getsize(self.path) / (1024 * 1024)
#             else:
#                 stats["current_file_size_mb"] = 0
            
#             # Buffer status
#             stats["buffer_size"] = len(self._buffer)
#             stats["buffer_capacity"] = self.buffer_size
            
#             # Uptime
#             stats["uptime_seconds"] = time.time() - stats["start_time"]
            
#             # Throughput
#             if stats["uptime_seconds"] > 0:
#                 stats["experiences_per_second"] = stats["total_logged"] / stats["uptime_seconds"]
#             else:
#                 stats["experiences_per_second"] = 0
            
#             return stats
    
#     def __del__(self):
#         """Ensure buffer is flushed on destruction."""
#         try:
#             self.flush()
#         except:
#             pass  # Avoid errors during cleanup
    
#     def __str__(self) -> str:
#         """String representation."""
#         stats = self.get_stats()
#         return (f"ExperienceLogger("
#                 f"logged={stats['total_logged']}, "
#                 f"buffer={stats['buffer_size']}/{stats['buffer_capacity']}, "
#                 f"rotations={stats['total_rotations']})")