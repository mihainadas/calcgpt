"""
CalcGPT Logger

Comprehensive logging system for CalcGPT with high traceability for terminal and file output.
Provides structured logging with timestamps, module info, and component-specific log files.
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import threading
from functools import wraps


# ANSI color codes for terminal output
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Standard colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to console output"""
    
    COLORS = {
        'DEBUG': Colors.CYAN,
        'INFO': Colors.GREEN,
        'WARNING': Colors.YELLOW,
        'ERROR': Colors.RED,
        'CRITICAL': Colors.BRIGHT_RED + Colors.BOLD,
    }
    
    def format(self, record):
        # Get the original formatted message
        message = super().format(record)
        
        # Add color based on log level
        level_color = self.COLORS.get(record.levelname, Colors.RESET)
        
        # Format with colors
        colored_message = f"{level_color}{message}{Colors.RESET}"
        
        return colored_message


class DetailedFormatter(logging.Formatter):
    """Detailed formatter for file output with maximum traceability"""
    
    def format(self, record):
        # Add extra context information
        record.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        record.thread_id = threading.get_ident()
        
        # Get caller information
        if hasattr(record, 'pathname'):
            record.module_name = Path(record.pathname).stem
        else:
            record.module_name = 'unknown'
            
        return super().format(record)


class CalcGPTLogger:
    """Main logger class for CalcGPT with component-specific logging"""
    
    def __init__(self, logs_dir: str = "logs"):
        """Initialize the CalcGPT logging system
        
        Args:
            logs_dir: Directory to store log files
        """
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Component-specific loggers
        self._loggers: Dict[str, logging.Logger] = {}
        self._handlers_created = set()
        
        # Default configuration
        self.console_level = logging.INFO
        self.file_level = logging.DEBUG
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.backup_count = 5
        
        # Initialize main logger
        self._setup_main_logger()
    
    def _setup_main_logger(self):
        """Setup the main CalcGPT logger"""
        logger = logging.getLogger('calcgpt')
        logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if not logger.handlers:
            # Console handler with colors
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.console_level)
            console_formatter = ColoredFormatter(
                fmt='%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # File handler with detailed formatting
            file_handler = logging.handlers.RotatingFileHandler(
                filename=self.logs_dir / 'calcgpt.log',
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(self.file_level)
            file_formatter = DetailedFormatter(
                fmt='%(timestamp)s | %(thread_id)d | %(module_name)s | %(funcName)s:%(lineno)d | %(levelname)s | %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        self._loggers['main'] = logger
    
    def get_logger(self, component: str) -> logging.Logger:
        """Get a component-specific logger
        
        Args:
            component: Component name (e.g., 'train', 'inference', 'eval')
            
        Returns:
            Logger instance for the component
        """
        if component not in self._loggers:
            self._create_component_logger(component)
        
        return self._loggers[component]
    
    def _create_component_logger(self, component: str):
        """Create a component-specific logger"""
        logger_name = f'calcgpt.{component}'
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        
        # Prevent adding handlers multiple times
        if logger_name not in self._handlers_created:
            # Component-specific file handler
            log_file = self.logs_dir / f'{component}.log'
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(self.file_level)
            file_formatter = DetailedFormatter(
                fmt='%(timestamp)s | %(thread_id)d | %(module_name)s | %(funcName)s:%(lineno)d | %(levelname)s | %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            # Also log to main console (will inherit from parent)
            logger.propagate = True
            
            self._handlers_created.add(logger_name)
        
        self._loggers[component] = logger
    
    def set_console_level(self, level: str):
        """Set console logging level
        
        Args:
            level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        """
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        
        if level.upper() in level_map:
            self.console_level = level_map[level.upper()]
            # Update existing console handlers
            for logger in self._loggers.values():
                for handler in logger.handlers:
                    if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                        handler.setLevel(self.console_level)
    
    def set_file_level(self, level: str):
        """Set file logging level
        
        Args:
            level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        """
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        
        if level.upper() in level_map:
            self.file_level = level_map[level.upper()]
            # Update existing file handlers
            for logger in self._loggers.values():
                for handler in logger.handlers:
                    if isinstance(handler, logging.handlers.RotatingFileHandler):
                        handler.setLevel(self.file_level)
    
    def log_function_entry(self, func_name: str, args: tuple = None, kwargs: dict = None, component: str = 'main'):
        """Log function entry with parameters
        
        Args:
            func_name: Name of the function
            args: Function arguments
            kwargs: Function keyword arguments
            component: Component name
        """
        logger = self.get_logger(component)
        args_str = f"args={args}" if args else ""
        kwargs_str = f"kwargs={kwargs}" if kwargs else ""
        params = ", ".join(filter(None, [args_str, kwargs_str]))
        logger.debug(f"ENTER {func_name}({params})")
    
    def log_function_exit(self, func_name: str, result: Any = None, component: str = 'main'):
        """Log function exit with result
        
        Args:
            func_name: Name of the function
            result: Function result
            component: Component name
        """
        logger = self.get_logger(component)
        result_str = f" -> {result}" if result is not None else ""
        logger.debug(f"EXIT {func_name}{result_str}")
    
    def log_performance(self, operation: str, duration: float, component: str = 'main', **metrics):
        """Log performance metrics
        
        Args:
            operation: Operation name
            duration: Duration in seconds
            component: Component name
            **metrics: Additional metrics to log
        """
        logger = self.get_logger(component)
        metrics_str = " | ".join([f"{k}={v}" for k, v in metrics.items()])
        logger.info(f"PERF {operation} | duration={duration:.3f}s | {metrics_str}")
    
    def log_model_info(self, model_info: Dict[str, Any], component: str = 'main'):
        """Log model information
        
        Args:
            model_info: Dictionary containing model information
            component: Component name
        """
        logger = self.get_logger(component)
        logger.info("MODEL INFO:")
        for key, value in model_info.items():
            logger.info(f"  {key}: {value}")
    
    def log_config(self, config: Dict[str, Any], component: str = 'main'):
        """Log configuration information
        
        Args:
            config: Configuration dictionary
            component: Component name
        """
        logger = self.get_logger(component)
        logger.info("CONFIGURATION:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics
        
        Returns:
            Dictionary with logging statistics
        """
        stats = {
            'logs_directory': str(self.logs_dir),
            'active_loggers': list(self._loggers.keys()),
            'console_level': logging.getLevelName(self.console_level),
            'file_level': logging.getLevelName(self.file_level),
            'log_files': []
        }
        
        # Get log file information
        for log_file in self.logs_dir.glob('*.log'):
            file_stats = log_file.stat()
            stats['log_files'].append({
                'name': log_file.name,
                'size_mb': file_stats.st_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat()
            })
        
        return stats


# Global logger instance
_global_logger: Optional[CalcGPTLogger] = None


def setup_logging(logs_dir: str = "logs", console_level: str = "INFO", file_level: str = "DEBUG") -> CalcGPTLogger:
    """Setup global CalcGPT logging
    
    Args:
        logs_dir: Directory for log files
        console_level: Console logging level
        file_level: File logging level
        
    Returns:
        CalcGPTLogger instance
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = CalcGPTLogger(logs_dir)
        _global_logger.set_console_level(console_level)
        _global_logger.set_file_level(file_level)
    
    return _global_logger


def get_logger(component: str = 'main') -> logging.Logger:
    """Get a logger for the specified component
    
    Args:
        component: Component name
        
    Returns:
        Logger instance
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = setup_logging()
    
    return _global_logger.get_logger(component)


def log_function(component: str = 'main', log_args: bool = False, log_result: bool = False):
    """Decorator to automatically log function entry and exit
    
    Args:
        component: Component name for logging
        log_args: Whether to log function arguments
        log_result: Whether to log function result
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(component)
            func_name = func.__name__
            
            # Log entry
            if log_args:
                logger.debug(f"ENTER {func_name}(args={args}, kwargs={kwargs})")
            else:
                logger.debug(f"ENTER {func_name}")
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Log exit
                if log_result:
                    logger.debug(f"EXIT {func_name} -> {result}")
                else:
                    logger.debug(f"EXIT {func_name}")
                
                return result
                
            except Exception as e:
                logger.error(f"ERROR in {func_name}: {e}", exc_info=True)
                raise
        
        return wrapper
    return decorator


def log_performance(operation: str, component: str = 'main'):
    """Decorator to log function performance
    
    Args:
        operation: Operation name
        component: Component name
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            
            logger = get_logger(component)
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"PERF {operation} completed in {duration:.3f}s")
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"PERF {operation} FAILED after {duration:.3f}s: {e}")
                raise
        
        return wrapper
    return decorator


# Convenience functions for common logging patterns
def log_step(message: str, component: str = 'main'):
    """Log a step in a process"""
    logger = get_logger(component)
    logger.info(f"STEP: {message}")


def log_metric(name: str, value: Any, component: str = 'main'):
    """Log a metric value"""
    logger = get_logger(component)
    logger.info(f"METRIC {name}: {value}")


def log_config_item(key: str, value: Any, component: str = 'main'):
    """Log a configuration item"""
    logger = get_logger(component)
    logger.info(f"CONFIG {key}: {value}")


def log_error(message: str, exception: Exception = None, component: str = 'main'):
    """Log an error with optional exception details"""
    logger = get_logger(component)
    if exception:
        logger.error(f"ERROR: {message}", exc_info=exception)
    else:
        logger.error(f"ERROR: {message}")


def log_warning(message: str, component: str = 'main'):
    """Log a warning message"""
    logger = get_logger(component)
    logger.warning(f"WARNING: {message}")


def log_success(message: str, component: str = 'main'):
    """Log a success message"""
    logger = get_logger(component)
    logger.info(f"SUCCESS: {message}")


if __name__ == "__main__":
    # Demo usage
    setup_logging(console_level="DEBUG")
    
    # Test different components
    train_logger = get_logger('train')
    inference_logger = get_logger('inference')
    
    train_logger.info("Starting training process")
    inference_logger.info("Loading model for inference")
    
    # Test decorators
    @log_function('train', log_args=True, log_result=True)
    @log_performance('model_training', 'train')
    def train_model(epochs=10, batch_size=32):
        import time
        time.sleep(0.1)  # Simulate work
        return {"loss": 0.25, "accuracy": 0.95}
    
    result = train_model(epochs=5)
    
    # Test convenience functions
    log_step("Model training completed", 'train')
    log_metric("final_loss", 0.25, 'train')
    log_success("Training completed successfully", 'train')
