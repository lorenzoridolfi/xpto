"""
Logger Module

This module provides logging functionality for the multi-agent system. It includes:
- Logging configuration setup
- Logger instance management
- Mixin class for adding logging capabilities
- Structured logging with context

The module supports different log levels (DEBUG, INFO, WARNING, ERROR) and
allows for additional context to be included in log messages.
"""

import logging.config
import os
from pathlib import Path
from typing import Optional, Dict, Any, TypedDict, Union

from .config import config

# Custom Exceptions
class LoggerError(Exception):
    """Base exception for logger-related errors."""
    pass

class LoggerConfigurationError(LoggerError):
    """Raised when logger configuration is invalid."""
    pass

class LoggerOperationError(LoggerError):
    """Raised when logger operations fail."""
    pass

# Type Definitions
class LogContext(TypedDict, total=False):
    """Type definition for log context data."""
    error: Optional[str]
    config: Optional[Dict[str, Any]]
    entry_id: Optional[str]
    query_length: Optional[int]
    response_length: Optional[int]
    has_metadata: Optional[bool]
    has_feedback: Optional[bool]
    older_than: Optional[str]
    purged_count: Optional[int]
    keep_last_n: Optional[int]

def setup_logging() -> None:
    """Set up logging configuration from global config.
    
    This function initializes the logging system with configuration from the
    global config object. It creates a logs directory if it doesn't exist and
    sets up the logging configuration.
    
    Raises:
        LoggerConfigurationError: If logging configuration fails
    """
    try:
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        # Configure logging
        logging.config.dictConfig(config.get_logging_config())
    except Exception as e:
        error_msg = f"Error loading logging configuration: {str(e)}"
        print(error_msg)  # Fallback to print if logging isn't configured
        logging.basicConfig(level=logging.INFO)
        raise LoggerConfigurationError(error_msg)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the specified name.
    
    Args:
        name: The name of the logger to get
        
    Returns:
        A configured logger instance
        
    Raises:
        LoggerOperationError: If logger creation fails
    """
    try:
        return logging.getLogger(name)
    except Exception as e:
        raise LoggerOperationError(f"Failed to get logger {name}: {str(e)}")

class LoggerMixin:
    """Mixin class to add logging capabilities to other classes.
    
    This class provides methods for logging at different levels (ERROR, WARNING,
    INFO, DEBUG) with support for additional context data. It automatically
    creates a logger instance using the class name or a provided name.
    """
    
    def __init__(self, logger_name: Optional[str] = None) -> None:
        """Initialize the logger.
        
        Args:
            logger_name: Optional name for the logger. If not provided,
                        uses the class name.
                        
        Raises:
            LoggerOperationError: If logger initialization fails
        """
        try:
            if logger_name is None:
                logger_name = self.__class__.__name__
            self.logger = get_logger(logger_name)
        except Exception as e:
            raise LoggerOperationError(f"Failed to initialize logger: {str(e)}")
    
    def log_error(self, message: str, exc_info: bool = True, **kwargs: Any) -> None:
        """Log an error message with optional exception info.
        
        Args:
            message: The error message
            exc_info: Whether to include exception info
            **kwargs: Additional context to include in the log
            
        Raises:
            LoggerOperationError: If logging fails
        """
        try:
            self.logger.error(message, exc_info=exc_info, extra=kwargs)
        except Exception as e:
            raise LoggerOperationError(f"Failed to log error: {str(e)}")
    
    def log_warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning message.
        
        Args:
            message: The warning message
            **kwargs: Additional context to include in the log
            
        Raises:
            LoggerOperationError: If logging fails
        """
        try:
            self.logger.warning(message, extra=kwargs)
        except Exception as e:
            raise LoggerOperationError(f"Failed to log warning: {str(e)}")
    
    def log_info(self, message: str, **kwargs: Any) -> None:
        """Log an info message.
        
        Args:
            message: The info message
            **kwargs: Additional context to include in the log
            
        Raises:
            LoggerOperationError: If logging fails
        """
        try:
            self.logger.info(message, extra=kwargs)
        except Exception as e:
            raise LoggerOperationError(f"Failed to log info: {str(e)}")
    
    def log_debug(self, message: str, **kwargs: Any) -> None:
        """Log a debug message.
        
        Args:
            message: The debug message
            **kwargs: Additional context to include in the log
            
        Raises:
            LoggerOperationError: If logging fails
        """
        try:
            self.logger.debug(message, extra=kwargs)
        except Exception as e:
            raise LoggerOperationError(f"Failed to log debug: {str(e)}") 