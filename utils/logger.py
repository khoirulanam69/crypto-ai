# utils/logger.py
import os
import sys
import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path

class MetricsCollector:
    """Collect and save trading metrics"""
    
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.metrics = {
            'start_time': datetime.now().isoformat(),
            'trades': [],
            'equity_history': [],
            'reward_history': [],
            'portfolio_values': [],
            'actions': []
        }
        
    def record_trade_step(self, price, action, equity, reward, portfolio_value):
        """Record a single trading step"""
        self.metrics['trades'].append({
            'timestamp': datetime.now().isoformat(),
            'price': price,
            'action': action,
            'equity': equity,
            'reward': reward,
            'portfolio_value': portfolio_value
        })
        
        self.metrics['equity_history'].append(equity)
        self.metrics['reward_history'].append(reward)
        self.metrics['portfolio_values'].append(portfolio_value)
        self.metrics['actions'].append(action)
        
    def save_summary(self, filename=None):
        """Save metrics summary to file"""
        if filename is None:
            filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.log_dir / filename
        
        # Calculate summary statistics
        if self.metrics['equity_history']:
            self.metrics['summary'] = {
                'end_time': datetime.now().isoformat(),
                'total_trades': len(self.metrics['trades']),
                'buy_count': self.metrics['actions'].count('BUY'),
                'sell_count': self.metrics['actions'].count('SELL'),
                'hold_count': self.metrics['actions'].count('HOLD'),
                'final_equity': self.metrics['equity_history'][-1] if self.metrics['equity_history'] else 0,
                'max_equity': max(self.metrics['equity_history']) if self.metrics['equity_history'] else 0,
                'min_equity': min(self.metrics['equity_history']) if self.metrics['equity_history'] else 0,
                'total_reward': sum(self.metrics['reward_history']),
                'avg_reward': sum(self.metrics['reward_history']) / len(self.metrics['reward_history']) if self.metrics['reward_history'] else 0
            }
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"Metrics saved to {filepath}")
        
        return self.metrics.get('summary', {})

def setup_logger(name, log_level=logging.INFO, log_to_file=True):
    """Setup a logger with console and file handlers"""
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_to_file:
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create log file with timestamp
        log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

# Convenience function for quick logging setup
def get_logger(name):
    """Get a pre-configured logger"""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_to_file = os.getenv("LOG_TO_FILE", "true").lower() == "true"
    
    return setup_logger(name, getattr(logging, log_level, logging.INFO), log_to_file)

# Example usage
if __name__ == "__main__":
    logger = get_logger("test")
    logger.info("Logger test successful")
    logger.warning("This is a warning")
    logger.error("This is an error")
    
    # Test metrics collector
    metrics = MetricsCollector()
    metrics.record_trade_step(50000, "BUY", 10000, 50, 10500)
    metrics.record_trade_step(51000, "SELL", 11000, 100, 11000)
    summary = metrics.save_summary()
    print(f"Summary: {summary}")