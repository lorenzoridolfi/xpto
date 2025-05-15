"""
Configuration settings for the multi-agent system.

This module contains default configuration settings and configuration management utilities
for the agent system. It provides a centralized location for all configurable parameters
and their default values.
"""

import os
import json
from typing import Dict, Any

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