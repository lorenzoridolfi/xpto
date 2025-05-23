"""
Configuration Module

This module provides configuration management for the multi-agent system. It includes:
- Default configuration settings
- Configuration loading and saving
- Singleton configuration manager
- Section-specific configuration access

The module supports loading configuration from JSON files with fallback to defaults,
and provides type-safe access to configuration values.
"""

import os
import json
from typing import Dict, Any, Optional, TypedDict, Union, TypeVar, Generic


# Custom Exceptions
class ConfigError(Exception):
    """Base exception for configuration-related errors."""

    pass


class ConfigLoadError(ConfigError):
    """Raised when configuration loading fails."""

    pass


class ConfigSaveError(ConfigError):
    """Raised when configuration saving fails."""

    pass


class ConfigValidationError(ConfigError):
    """Raised when configuration validation fails."""

    pass


# Type Definitions
class LoggingConfig(TypedDict):
    """Type definition for logging configuration."""

    level: str
    format: str
    file: str
    console: bool


class CacheConfig(TypedDict):
    """Type definition for cache configuration."""

    enabled: bool


class StorageConfig(TypedDict):
    """Type definition for storage configuration."""

    max_entries: int
    keep_last_n_entries: int
    purge_older_than_hours: int


class APIConfig(TypedDict):
    """Type definition for API configuration."""

    base_url: str
    timeout: int
    retry_count: int


class OpenAIConfig(TypedDict):
    """Type definition for OpenAI configuration."""

    api_key: str
    model: str
    temperature: float
    max_tokens: int


# Default configuration settings
DEFAULT_CONFIG: Dict[str, Any] = {
    "file_manifest": [],
    "max_retries": 3,
    "retry_delay": 1,
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s [%(levelname)s] %(message)s",
        "file": "agent_system.log",
        "console": True,
    },
    "cache": {"enabled": False},
}


def load_config(config_path: str = "agent_config.json") -> Dict[str, Any]:
    """Load configuration from a JSON file, falling back to defaults if file doesn't exist.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dict containing the configuration settings

    Raises:
        ConfigLoadError: If configuration file exists but cannot be loaded
    """
    # Try to find config in config/shared directory first
    shared_config_path = os.path.join("config", "shared", config_path)
    if os.path.exists(shared_config_path):
        config_path = shared_config_path

    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            raise ConfigLoadError(
                f"Failed to load configuration from {config_path}: {str(e)}"
            )
    return DEFAULT_CONFIG.copy()


def save_config(config: Dict[str, Any], config_path: str = "agent_config.json") -> None:
    """Save configuration to a JSON file.

    Args:
        config: Configuration dictionary to save
        config_path: Path where to save the configuration

    Raises:
        ConfigSaveError: If configuration cannot be saved
    """
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        raise ConfigSaveError(
            f"Failed to save configuration to {config_path}: {str(e)}"
        )


class Config:
    """Global configuration manager.

    This class implements a singleton pattern for managing global configuration.
    It provides methods for loading configuration from files and accessing
    configuration values in a type-safe manner.
    """

    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls) -> "Config":
        """Create or return the singleton instance.

        Returns:
            The singleton Config instance
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the configuration manager."""
        if not self._config:
            self.load_config()

    def load_config(self, config_path: str = "global_config.json") -> None:
        """Load configuration from JSON file.

        Args:
            config_path: Path to the configuration file

        Raises:
            ConfigLoadError: If configuration cannot be loaded
        """
        # Try to find config in config/shared directory first
        shared_config_path = os.path.join("config", "shared", config_path)
        if os.path.exists(shared_config_path):
            config_path = shared_config_path

        try:
            with open(config_path, "rt") as f:
                self._config = json.load(f)
        except Exception as e:
            error_msg = f"Error loading configuration: {str(e)}"
            print(error_msg)  # Fallback to print if logging isn't configured
            self._config = {}
            raise ConfigLoadError(error_msg)

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

    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration.

        Returns:
            Logging configuration dictionary

        Raises:
            ConfigValidationError: If logging configuration is invalid
        """
        config = self.get("logging", default={})
        try:
            return LoggingConfig(**config)
        except Exception as e:
            raise ConfigValidationError(f"Invalid logging configuration: {str(e)}")

    def get_openai_config(self) -> OpenAIConfig:
        """Get OpenAI configuration.

        Returns:
            OpenAI configuration dictionary

        Raises:
            ConfigValidationError: If OpenAI configuration is invalid
        """
        config = self.get("openai", default={})
        try:
            return OpenAIConfig(**config)
        except Exception as e:
            raise ConfigValidationError(f"Invalid OpenAI configuration: {str(e)}")

    def get_storage_config(self) -> StorageConfig:
        """Get storage configuration.

        Returns:
            Storage configuration dictionary

        Raises:
            ConfigValidationError: If storage configuration is invalid
        """
        config = self.get("storage", default={})
        try:
            return StorageConfig(**config)
        except Exception as e:
            raise ConfigValidationError(f"Invalid storage configuration: {str(e)}")

    def get_api_config(self) -> APIConfig:
        """Get API configuration.

        Returns:
            API configuration dictionary

        Raises:
            ConfigValidationError: If API configuration is invalid
        """
        config = self.get("api", default={})
        try:
            return APIConfig(**config)
        except Exception as e:
            raise ConfigValidationError(f"Invalid API configuration: {str(e)}")

    def get_cache_config(self) -> CacheConfig:
        """Get cache configuration.

        Returns:
            Cache configuration dictionary

        Raises:
            ConfigValidationError: If cache configuration is invalid
        """
        config = self.get("cache", default={})
        try:
            return CacheConfig(**config)
        except Exception as e:
            raise ConfigValidationError(f"Invalid cache configuration: {str(e)}")


# Create global config instance
config = Config()
