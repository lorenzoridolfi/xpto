# Configuration and Logging System

## Overview
The system uses a centralized configuration management system with a global configuration file (`global_config.json`) and a structured logging system. This document describes how to configure and use these systems.

## Configuration System

### Global Configuration File
The `global_config.json` file contains all configuration settings for the application. It is structured into several sections:

```json
{
    "logging": { ... },
    "openai": { ... },
    "storage": { ... },
    "api": { ... },
    "cache": { ... }
}
```

### Configuration Sections

#### 1. Logging Configuration
```json
{
    "logging": {
        "version": 1,
        "disable_existing_loggers": false,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]"
            }
        },
        "handlers": {
            "console": { ... },
            "file": { ... },
            "error_file": { ... }
        },
        "loggers": {
            "feedback_storage": { ... },
            "feedback_manager": { ... },
            "api": { ... }
        }
    }
}
```

#### 2. OpenAI Configuration
```json
{
    "openai": {
        "api_key": "your-api-key-here",
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 4096,
        "timeout": 30,
        "retry_attempts": 3,
        "retry_delay": 1
    }
}
```

#### 3. Storage Configuration
```json
{
    "storage": {
        "max_entries": 1000,
        "purge_older_than_hours": 24,
        "keep_last_n_entries": 100
    }
}
```

#### 4. API Configuration
```json
{
    "api": {
        "host": "0.0.0.0",
        "port": 8000,
        "debug": false,
        "workers": 4,
        "timeout": 60,
        "cors_origins": ["*"]
    }
}
```

#### 5. Cache Configuration
```json
{
    "cache": {
        "enabled": true,
        "max_size": 1000,
        "ttl_seconds": 3600,
        "cleanup_interval_seconds": 300
    }
}
```

## Using the Configuration System

### Accessing Configuration
The configuration system is implemented as a singleton in `src/config.py`. To use it in your code:

```python
from src.config import config

# Get entire section
openai_config = config.get_openai_config()

# Get specific value
model = config.get("openai", "model", "gpt-4")  # with default value
```

### Available Methods
- `get(section: str, key: Optional[str] = None, default: Any = None) -> Any`
- `get_logging_config() -> Dict[str, Any]`
- `get_openai_config() -> Dict[str, Any]`
- `get_storage_config() -> Dict[str, Any]`
- `get_api_config() -> Dict[str, Any]`
- `get_cache_config() -> Dict[str, Any]`

## Logging System

### Setup
The logging system is initialized at application startup:

```python
from src.logger import setup_logging

# Initialize logging
setup_logging()
```

### Using Logging in Classes
Use the `LoggerMixin` to add logging capabilities to your classes:

```python
from src.logger import LoggerMixin

class YourClass(LoggerMixin):
    def __init__(self):
        super().__init__()  # Initialize logger
        
    def some_method(self):
        self.log_info("Processing...")
        try:
            # Your code here
            self.log_debug("Operation successful")
        except Exception as e:
            self.log_error("Operation failed", error=str(e))
```

### Available Logging Methods
- `log_debug(message: str, **kwargs) -> None`
- `log_info(message: str, **kwargs) -> None`
- `log_warning(message: str, **kwargs) -> None`
- `log_error(message: str, exc_info: bool = True, **kwargs) -> None`

### Log Files
The system creates the following log files in the `logs` directory:
- `app.log`: General application logs
- `error.log`: Error-specific logs
- Module-specific logs (e.g., `storage.log`)

## Best Practices

### Configuration
1. Always provide default values when accessing configuration
2. Use type hints for configuration values
3. Document configuration options
4. Validate configuration values
5. Use environment variables for sensitive data

### Logging
1. Use appropriate log levels
2. Include relevant context in log messages
3. Use structured logging with extra fields
4. Handle exceptions properly
5. Rotate log files regularly

## Example Usage

### Configuration Example
```python
from src.config import config

# Get OpenAI configuration
openai_config = config.get_openai_config()
model = openai_config.get("model", "gpt-4")

# Get storage configuration
storage_config = config.get_storage_config()
max_entries = storage_config.get("max_entries", 1000)
```

### Logging Example
```python
from src.logger import LoggerMixin

class DataProcessor(LoggerMixin):
    def __init__(self):
        super().__init__()
        
    async def process_data(self, data: dict):
        self.log_info("Starting data processing", data_size=len(data))
        try:
            # Process data
            self.log_debug("Processing step 1", step="validation")
            # More processing
            self.log_info("Data processing completed", 
                         processed_items=len(data),
                         success_rate=0.95)
        except Exception as e:
            self.log_error("Data processing failed",
                          error=str(e),
                          data_size=len(data))
            raise
```

## Troubleshooting

### Common Issues
1. **Configuration Not Found**
   - Check if `global_config.json` exists
   - Verify file permissions
   - Check JSON syntax

2. **Logging Issues**
   - Verify log directory exists
   - Check file permissions
   - Ensure proper logger configuration

3. **Configuration Access**
   - Use correct section names
   - Provide default values
   - Check type compatibility 