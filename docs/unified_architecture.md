# Unified Agent System Architecture

This document describes the unified architecture for agent-based systems, implemented across `toy_example.py` and `update_manifest.py`.

## Architecture Overview

The unified architecture consists of three main components:

1. **Base Module** (`src/base_agent_system.py`)
   - Common functionality
   - Standard logging
   - Base agent creation
   - Group chat management

2. **Specialized Agents**
   - System-specific agents
   - Extended functionality
   - Custom validation
   - Domain-specific logic

3. **Configuration System**
   - Standardized JSON config
   - Agent settings
   - System parameters
   - Logging configuration

## System Components

### Base Module

The base module provides shared functionality used by both systems:

```python
from base_agent_system import (
    setup_logging, log_event, create_base_agents, create_group_chat,
    load_json_file, save_json_file, FILE_LOG, ROOT_CAUSE_DATA
)
```

### Common Agents

Both systems share these base agents:

1. **UserProxyAgent**
   - Handles user interaction
   - Manages input/output
   - Controls conversation flow

2. **SupervisorAgent**
   - Orchestrates workflow
   - Manages agent coordination
   - Ensures task completion

### Specialized Agents

#### Toy Example System
- `FileReaderAgent`: Reads and processes input files
- `WriterAgent`: Generates content
- `InformationVerifierAgent`: Validates information
- `TextQualityAgent`: Ensures content quality

#### Update Manifest System
- `FileReaderAgent`: Reads manifest files
- `ManifestUpdaterAgent`: Updates manifest content
- `LoggingConfigAgent`: Configures logging
- `ValidationAgent`: Validates configuration

## Configuration Structure

Both systems use a standardized configuration format:

```json
{
    "task_description": "string",
    "hierarchy": ["string"],
    "file_manifest": [
        {
            "filename": "string",
            "description": "string"
        }
    ],
    "agents": {
        "AgentName": {
            "name": "string",
            "description": "string",
            "system_message": "string"
        }
    },
    "llm_config": {
        "supervisor": {
            "model": "string",
            "parameters": {}
        },
        "creator": {
            "model": "string",
            "parameters": {}
        },
        "validator": {
            "model": "string",
            "parameters": {}
        }
    },
    "system": {
        "user_proxy": {
            "name": "string",
            "human_input_mode": "string",
            "max_consecutive_auto_reply": "number"
        },
        "group_chat": {
            "max_round": "number",
            "speaker_selection_method": "string",
            "allow_repeat_speaker": "boolean"
        }
    }
}
```

## Implementation Example

### Agent Creation

```python
def create_agents(config: Dict) -> Dict[str, Any]:
    """Create and configure the agents using settings from config file."""
    # Create base agents
    base_agents = create_base_agents(config, logger)
    
    # Create specialized agents
    specialized_agents = create_specialized_agents(config, logger)
    
    # Create group chat
    all_agents = [base_agents["user_proxy"]] + specialized_agents + [base_agents["supervisor"]]
    group_chat_manager = create_group_chat(all_agents, config, logger)
    
    return {
        "specialized_agents": specialized_agents,
        "user_proxy": base_agents["user_proxy"],
        "supervisor": base_agents["supervisor"],
        "group_chat_manager": group_chat_manager
    }
```

### Main Process

```python
def main():
    """Main entry point."""
    try:
        # Load configuration
        config = load_json_file("config.json")
        
        # Setup logging
        logger = setup_logging(config)
        
        # Create agents
        agents = create_agents(config)
        
        # Start processing
        logger.info("Starting process")
        
        # Initialize the process
        agents["user_proxy"].initiate_chat(
            agents["group_chat_manager"],
            message="Process initialization message"
        )
        
        logger.info("Process completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise
```

## Best Practices

1. **Code Organization**
   - Use base module for common functionality
   - Implement specialized agents in separate files
   - Follow consistent naming conventions
   - Maintain clear separation of concerns

2. **Configuration Management**
   - Use standardized JSON format
   - Include all required settings
   - Validate configuration
   - Document configuration options

3. **Error Handling**
   - Implement comprehensive error handling
   - Log all errors with context
   - Provide meaningful error messages
   - Handle edge cases gracefully

4. **Logging**
   - Use standardized logging
   - Include detailed context
   - Track all significant events
   - Monitor system performance

5. **Testing**
   - Test base functionality
   - Test specialized agents
   - Test error handling
   - Test configuration loading

## Integration Guidelines

To integrate a new system with this architecture:

1. **Setup**
   - Import base module
   - Create configuration file
   - Implement specialized agents
   - Set up logging

2. **Implementation**
   - Extend base functionality
   - Implement required interfaces
   - Handle errors appropriately
   - Follow naming conventions

3. **Configuration**
   - Create JSON config
   - Define agent settings
   - Configure logging
   - Set system parameters

4. **Testing**
   - Test base integration
   - Test specialized functionality
   - Test error handling
   - Test configuration

## Maintenance

Regular maintenance tasks:

1. **Code**
   - Update base module
   - Maintain specialized agents
   - Review error handling
   - Update documentation

2. **Configuration**
   - Review settings
   - Update parameters
   - Validate structure
   - Document changes

3. **Logging**
   - Monitor logs
   - Review error patterns
   - Update logging levels
   - Clean up old logs

4. **Performance**
   - Monitor system metrics
   - Optimize agent behavior
   - Review resource usage
   - Update caching strategy 