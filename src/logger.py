import logging.config
import os
from pathlib import Path
from typing import Optional

from .config import config

def setup_logging() -> None:
    """Set up logging configuration from global config."""
    try:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        # Configure logging
        logging.config.dictConfig(config.get_logging_config())
    except Exception as e:
        print(f"Error loading logging configuration: {e}")
        logging.basicConfig(level=logging.INFO)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the specified name.
    
    Args:
        name: The name of the logger to get
        
    Returns:
        A configured logger instance
    """
    return logging.getLogger(name)

class LoggerMixin:
    """Mixin class to add logging capabilities to other classes."""
    
    def __init__(self, logger_name: Optional[str] = None):
        """Initialize the logger.
        
        Args:
            logger_name: Optional name for the logger. If not provided,
                        uses the class name.
        """
        if logger_name is None:
            logger_name = self.__class__.__name__
        self.logger = get_logger(logger_name)
    
    def log_error(self, message: str, exc_info: bool = True, **kwargs) -> None:
        """Log an error message with optional exception info.
        
        Args:
            message: The error message
            exc_info: Whether to include exception info
            **kwargs: Additional context to include in the log
        """
        self.logger.error(message, exc_info=exc_info, extra=kwargs)
    
    def log_warning(self, message: str, **kwargs) -> None:
        """Log a warning message.
        
        Args:
            message: The warning message
            **kwargs: Additional context to include in the log
        """
        self.logger.warning(message, extra=kwargs)
    
    def log_info(self, message: str, **kwargs) -> None:
        """Log an info message.
        
        Args:
            message: The info message
            **kwargs: Additional context to include in the log
        """
        self.logger.info(message, extra=kwargs)
    
    def log_debug(self, message: str, **kwargs) -> None:
        """Log a debug message.
        
        Args:
            message: The debug message
            **kwargs: Additional context to include in the log
        """
        self.logger.debug(message, extra=kwargs) 