# Base Agent System

The Base Agent System provides a foundation for building multi-agent systems with standardized functionality, logging, and agent management.

## Overview

Located in `src/base_agent_system.py`, this module provides shared functionality for agent-based systems, including:
- Common agent creation and configuration
- Standard logging and event tracking
- Shared utility functions
- Base agent classes

## Key Components

### Global State

```python
# Global logs and cache
ROOT_CAUSE_DATA: List[dict] = []  # Stores detailed event data for root cause analysis
FILE_LOG: List[str] = []         # Tracks file operations
ACTION_LOG: List[str] = []       # Records agent actions and decisions

# Initialize LLM cache
llm_cache = LLMCache(
    max_size=1000,
    similarity_threshold=0.85,
    expiration_hours=24
)
```

### Core Functions

#### Logging Setup
```python
def setup_logging(config: dict) -> None:
    """Configure logging based on configuration settings."""
```
- Configures system-wide logging
- Supports both console and file handlers
- Uses standardized formatting
- Returns configured logger instance

#### Event Logging
```python
def log_event(logger: logging.Logger, agent_name: str, event_type: str, inputs: List[BaseChatMessage], outputs) -> None:
    """Log an event in the system with detailed information."""
```
- Records detailed agent interactions
- Tracks input/output messages
- Maintains event history for analysis
- Supports JSON message parsing

#### Base Agent Creation
```python
def create_base_agents(config: Dict, logger: logging.Logger) -> Dict[str, Any]:
    """Create and configure the base agents using settings from config file."""
```
- Creates UserProxy agent
- Creates Supervisor agent
- Configures agent settings from config
- Returns dictionary of base agents

#### Group Chat Management
```python
def create_group_chat(agents: List[Any], config: Dict, logger: logging.Logger) -> GroupChatManager:
    """Create and configure the group chat system."""
```
- Creates GroupChat instance
- Configures GroupChatManager
- Sets up agent communication
- Manages conversation flow

### Utility Functions

#### File Operations
```python
def load_json_file(file_path: str) -> Dict:
    """Load and parse a JSON file."""

def save_json_file(data: Dict, file_path: str) -> None:
    """Save data to a JSON file with proper formatting."""
```
- Standardized JSON file handling
- Error handling and validation
- Consistent formatting

## Usage Example

```python
from base_agent_system import (
    setup_logging, log_event, create_base_agents, create_group_chat,
    load_json_file, save_json_file, FILE_LOG, ROOT_CAUSE_DATA
)

# Load configuration
config = load_json_file("config.json")

# Setup logging
logger = setup_logging(config)

# Create base agents
base_agents = create_base_agents(config, logger)

# Create specialized agents
specialized_agents = create_specialized_agents(config, logger)

# Create group chat
all_agents = [base_agents["user_proxy"]] + specialized_agents + [base_agents["supervisor"]]
group_chat_manager = create_group_chat(all_agents, config, logger)
```

## Best Practices

1. **Configuration**
   - Use standardized configuration format
   - Include all required agent settings
   - Define logging configuration
   - Specify group chat parameters

2. **Logging**
   - Use the provided logging setup
   - Log all significant events
   - Include detailed context in logs
   - Monitor system performance

3. **Agent Creation**
   - Extend base agents for specialized functionality
   - Follow consistent naming conventions
   - Implement required interfaces
   - Handle errors appropriately

4. **Group Chat**
   - Include all necessary agents
   - Set appropriate max rounds
   - Configure speaker selection
   - Monitor conversation flow

## Error Handling

The base system includes standardized error handling for:
- File operations
- JSON parsing
- Agent creation
- Group chat management
- Logging setup

## Integration

The base system is designed to be integrated with specialized agent systems:
1. Import required components
2. Extend base functionality
3. Implement specialized agents
4. Configure system settings
5. Initialize group chat 