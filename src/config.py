"""
Configuration settings for the multi-agent system.

This module contains default configuration settings and configuration management utilities
for the agent system. It provides a centralized location for all configurable parameters
and their default values.
"""

import os
import json
from typing import Dict, Any, Optional

# Default configuration settings
DEFAULT_CONFIG = {
    "file_manifest": [],
    "max_retries": 3,
    "retry_delay": 1,
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s [%(levelname)s] %(message)s",
        "file": "agent_system.log",
        "console": True
    },
    "cache": {
        "enabled": False
    }
}

def load_config(config_path: str = "agent_config.json") -> Dict[str, Any]:
    """
    Load configuration from a JSON file, falling back to defaults if file doesn't exist.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dict containing the configuration settings
    """
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return DEFAULT_CONFIG.copy()

def save_config(config: Dict[str, Any], config_path: str = "agent_config.json") -> None:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary to save
        config_path: Path where to save the configuration
    """
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

class Config:
    """Global configuration manager."""
    
    _instance = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._config:
            self.load_config()
    
    def load_config(self, config_path: str = "global_config.json") -> None:
        """Load configuration from JSON file.
        
        Args:
            config_path: Path to the configuration file
        """
        try:
            with open(config_path, "rt") as f:
                self._config = json.load(f)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            self._config = {}
    
    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            section: Configuration section name
            key: Optional key within the section
            default: Default value if section/key not found
            
        Returns:
            Configuration value or default
        """
        if section not in self._config:
            return default
            
        if key is None:
            return self._config[section]
            
        return self._config[section].get(key, default)
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration.
        
        Returns:
            Logging configuration dictionary
        """
        return self.get("logging", default={})
    
    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI configuration.
        
        Returns:
            OpenAI configuration dictionary
        """
        return self.get("openai", default={})
    
    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage configuration.
        
        Returns:
            Storage configuration dictionary
        """
        return self.get("storage", default={})
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration.
        
        Returns:
            API configuration dictionary
        """
        return self.get("api", default={})
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Get cache configuration.
        
        Returns:
            Cache configuration dictionary
        """
        return self.get("cache", default={})

# Create global config instance
config = Config() 