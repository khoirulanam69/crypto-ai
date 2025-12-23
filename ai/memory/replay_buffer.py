import csv
import os
import time
import threading
import math
import json
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ReplayBuffer:
    """
    Thread-safe, efficient experience replay buffer for RL training.
    
    Features:
    - Batch writing for performance
    - Size limiting with automatic trimming
    - Data validation
    - File locking for multi-process safety
    - Optional compression
    - Metadata tracking
    """
    
    def __init__(self, 
                 path: str = "data/replay_buffer.csv",
                 max_size: int = 50000,
                 buffer_size: int = 100,
                 compress: bool = False):
        """
        Initialize replay buffer.
        
        Args:
            path: Path to CSV file
            max_size: Maximum number of rows to keep
            buffer_size: Number of rows to buffer before writing to disk
            compress: Whether to compress old files (not implemented in CSV)
        """
        self.path = path
        self.max_size = max_size
        self.buffer_size = buffer_size
        self.compress = compress
        
        # Thread safety
        self._lock = threading.RLock()
        
        # In-memory buffer for batch writes
        self._buffer: List[List] = []
        
        # Initialize file and directory
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        
        # Load existing size
        self._size = self._load_existing_size()
        
        # Metadata
        self.metadata_path = path.replace('.csv', '_meta.json')
        self._load_metadata()
        
        logger.info(f"ReplayBuffer initialized: path={path}, size={self._size}, max={max_size}")
    
    def _load_existing_size(self) -> int:
        """Load size from existing file efficiently."""
        if not os.path.exists(self.path):
            # Create file with header
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "price",
                    "action",
                    "reward", 
                    "equity",
                    "episode_id",
                    "step"
                ])
            return 0
        
        # Fast line count (skip header)
        try:
            with open(self.path, 'rb') as f:
                # Skip header
                next(f)
                # Count remaining lines
                size = sum(1 for _ in f)
                return size
        except Exception as e:
            logger.error(f"Failed to load buffer size: {e}")
            return 0
    
    def _load_metadata(self) -> None:
        """Load or create metadata file."""
        metadata = {
            "created": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "total_appends": 0,
            "total_trims": 0,
            "max_size": self.max_size,
            "version": "1.0"
        }
        
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    metadata.update(json.load(f))
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        
        self.metadata = metadata
        self._save_metadata()
    
    def _save_metadata(self) -> None:
        """Save metadata to file."""
        self.metadata["last_modified"] = datetime.now().isoformat()
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def append(self, 
               price: float, 
               action: int, 
               reward: float, 
               equity: float,
               episode_id: Optional[int] = None,
               step: Optional[int] = None) -> bool:
        """
        Append experience to buffer.
        
        Args:
            price: Current price
            action: Action taken (0=HOLD, 1=BUY, 2=SELL)
            reward: Reward received
            equity: Current equity
            episode_id: Optional episode identifier
            step: Optional step number in episode
            
        Returns:
            True if successful
        """
        # Validate inputs
        try:
            # Check for NaN/inf
            if not isinstance(price, (int, float)) or math.isnan(price) or math.isinf(price):
                logger.warning(f"Invalid price: {price}")
                price = 0.0
            
            if not isinstance(action, int):
                try:
                    action = int(action)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid action: {action}")
                    action = 0
            
            if not isinstance(reward, (int, float)) or math.isnan(reward) or math.isinf(reward):
                logger.warning(f"Invalid reward: {reward}")
                reward = 0.0
            
            if not isinstance(equity, (int, float)) or math.isnan(equity) or math.isinf(equity):
                logger.warning(f"Invalid equity: {equity}")
                equity = 0.0
            
            # Generate timestamp (milliseconds, safe for 64-bit)
            timestamp = int(time.time() * 1000)
            
            # Default episode/step if not provided
            if episode_id is None:
                episode_id = int(time.time() // 3600)  # Hour-based episode
            
            if step is None:
                step = self.metadata.get("total_appends", 0)
            
            # Create row
            row = [
                timestamp,
                float(price),
                int(action),
                float(reward),
                float(equity),
                int(episode_id),
                int(step)
            ]
            
            # Add to buffer
            with self._lock:
                self._buffer.append(row)
                
                # Check if buffer needs flushing
                if len(self._buffer) >= self.buffer_size:
                    self._flush_buffer()
                
                # Check if we need to trim
                if self._size >= self.max_size:
                    self._trim_oldest(self.max_size // 2)
                
                # Update metadata
                self.metadata["total_appends"] = self.metadata.get("total_appends", 0) + 1
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to append to replay buffer: {e}")
            return False
    
    def _flush_buffer(self) -> None:
        """Flush in-memory buffer to disk."""
        if not self._buffer:
            return
        
        with self._lock:
            try:
                # File locking for multi-process safety
                with open(self.path, "a", newline="") as f:
                    # On Unix systems, use flock
                    if hasattr(os, 'posix'):
                        import fcntl
                        fcntl.flock(f, fcntl.LOCK_EX)
                    
                    writer = csv.writer(f)
                    writer.writerows(self._buffer)
                    
                    if hasattr(os, 'posix'):
                        fcntl.flock(f, fcntl.LOCK_UN)
                
                # Update size and clear buffer
                self._size += len(self._buffer)
                self._buffer.clear()
                
                # Save metadata
                self._save_metadata()
                
                logger.debug(f"Flushed {len(self._buffer)} rows to replay buffer")
                
            except Exception as e:
                logger.error(f"Failed to flush buffer: {e}")
                # Keep buffer for retry
    
    def _trim_oldest(self, keep_size: int) -> None:
        """
        Trim oldest entries, keeping only latest keep_size rows.
        
        Args:
            keep_size: Number of rows to keep
        """
        if self._size <= keep_size:
            return
        
        with self._lock:
            try:
                # Read all data
                rows = []
                with open(self.path, 'r') as f:
                    reader = csv.reader(f)
                    header = next(reader)  # Save header
                    rows = list(reader)
                
                # Keep only newest entries
                if len(rows) > keep_size:
                    rows = rows[-keep_size:]
                    self._size = len(rows)
                    
                    # Write back
                    with open(self.path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(header)
                        writer.writerows(rows)
                    
                    # Update metadata
                    self.metadata["total_trims"] = self.metadata.get("total_trims", 0) + 1
                    self._save_metadata()
                    
                    logger.info(f"Trimmed replay buffer: {len(rows)} rows kept")
                
            except Exception as e:
                logger.error(f"Failed to trim replay buffer: {e}")
    
    def size(self) -> int:
        """Get current buffer size (in-memory + on-disk)."""
        with self._lock:
            return self._size + len(self._buffer)
    
    def load_batch(self, 
                   batch_size: int = 32,
                   recent_first: bool = True) -> List[Dict[str, Any]]:
        """
        Load a batch of experiences for training.
        
        Args:
            batch_size: Number of experiences to load
            recent_first: If True, load most recent experiences first
            
        Returns:
            List of experience dictionaries
        """
        with self._lock:
            self._flush_buffer()  # Ensure all data is on disk
            
            if self._size == 0:
                return []
            
            try:
                experiences = []
                with open(self.path, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                
                # Select rows (most recent first if requested)
                if recent_first:
                    rows = rows[-batch_size:]
                else:
                    # Random sampling (better for training)
                    import random
                    if len(rows) > batch_size:
                        rows = random.sample(rows, batch_size)
                    else:
                        rows = rows
                
                # Convert to dictionaries
                for row in rows:
                    exp = {
                        'timestamp': int(row['timestamp']),
                        'price': float(row['price']),
                        'action': int(row['action']),
                        'reward': float(row['reward']),
                        'equity': float(row['equity']),
                        'episode_id': int(row.get('episode_id', 0)),
                        'step': int(row.get('step', 0))
                    }
                    experiences.append(exp)
                
                return experiences
                
            except Exception as e:
                logger.error(f"Failed to load batch: {e}")
                return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        with self._lock:
            stats = self.metadata.copy()
            stats.update({
                'current_size': self.size(),
                'buffer_pending': len(self._buffer),
                'on_disk_size': self._size,
                'max_size': self.max_size,
                'usage_percent': (self.size() / self.max_size * 100) if self.max_size > 0 else 0
            })
            return stats
    
    def clear(self) -> None:
        """Clear all data from buffer."""
        with self._lock:
            self._buffer.clear()
            self._size = 0
            
            # Reset file
            with open(self.path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "price",
                    "action",
                    "reward", 
                    "equity",
                    "episode_id",
                    "step"
                ])
            
            # Update metadata
            self.metadata["total_appends"] = 0
            self.metadata["total_trims"] = 0
            self._save_metadata()
            
            logger.info("Replay buffer cleared")
    
    def backup(self, backup_path: Optional[str] = None) -> bool:
        """Create backup of replay buffer."""
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{self.path}.backup_{timestamp}"
        
        try:
            import shutil
            shutil.copy2(self.path, backup_path)
            logger.info(f"Replay buffer backed up to: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to backup replay buffer: {e}")
            return False
    
    def __del__(self):
        """Ensure buffer is flushed on destruction."""
        try:
            self._flush_buffer()
        except:
            pass  # Avoid errors during cleanup
    
    def __str__(self) -> str:
        """String representation."""
        stats = self.get_stats()
        return (f"ReplayBuffer(size={stats['current_size']}/{stats['max_size']}, "
                f"pending={stats['buffer_pending']}, "
                f"appends={stats['total_appends']})")