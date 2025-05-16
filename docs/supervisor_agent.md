# Supervisor Agent Documentation

## Overview
The Supervisor Agent is a central component that orchestrates and manages the behavior of other agents in both the `update_manifest` and `toy_example` systems. It ensures consistent operation, error handling, and coordination across all agent interactions.

## Core Responsibilities

### 1. Agent Management
- Initializes and configures all subordinate agents
- Maintains agent lifecycle
- Monitors agent health and performance
- Handles agent failures and recovery

### 2. Task Coordination
- Distributes tasks among agents
- Ensures proper task sequencing
- Manages task dependencies
- Tracks task completion status

### 3. Communication Management
- Routes messages between agents
- Maintains conversation history
- Ensures message delivery
- Handles communication failures

### 4. Error Handling
- Catches and processes agent errors
- Implements retry mechanisms
- Provides error recovery strategies
- Maintains system stability

## Implementation Details

### Base Supervisor Class
```python
class SupervisorAgent(LoggerMixin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.agents = {}
        self.conversation_history = []
        self.task_queue = asyncio.Queue()
        self.error_count = 0
        self.max_retries = config.get("max_retries", 3)
```

### Key Methods

#### 1. Agent Initialization
```python
async def initialize_agents(self):
    """Initialize all subordinate agents with proper configuration."""
    try:
        # Initialize each agent with its specific config
        for agent_name, agent_config in self.config["agents"].items():
            agent = await self._create_agent(agent_name, agent_config)
            self.agents[agent_name] = agent
            self.log_info(f"Initialized {agent_name}", config=agent_config)
    except Exception as e:
        self.log_error("Failed to initialize agents", error=str(e))
        raise
```

#### 2. Task Management
```python
async def process_task(self, task: Dict[str, Any]):
    """Process a single task through the agent pipeline."""
    try:
        # Validate task
        if not self._validate_task(task):
            raise ValueError("Invalid task format")
            
        # Route task to appropriate agent
        result = await self._route_task(task)
        
        # Handle result
        await self._handle_result(result)
        
        return result
    except Exception as e:
        self.log_error("Task processing failed", error=str(e), task=task)
        raise
```

#### 3. Error Handling
```python
async def handle_error(self, error: Exception, context: Dict[str, Any]):
    """Handle errors in agent operations."""
    self.error_count += 1
    
    if self.error_count > self.max_retries:
        self.log_error("Max retries exceeded", error=str(error))
        raise error
        
    # Implement retry logic
    await self._retry_operation(context)
```

#### 4. Message Routing
```python
async def route_message(self, message: Dict[str, Any]):
    """Route messages between agents."""
    try:
        sender = message.get("sender")
        recipient = message.get("recipient")
        content = message.get("content")
        
        if not all([sender, recipient, content]):
            raise ValueError("Invalid message format")
            
        # Route message to recipient
        await self.agents[recipient].receive_message(message)
        
        # Log message
        self.log_debug("Message routed", 
                      sender=sender,
                      recipient=recipient,
                      message_type=message.get("type"))
    except Exception as e:
        self.log_error("Message routing failed", error=str(e))
        raise
```

## Configuration

### Supervisor Configuration
```json
{
    "supervisor": {
        "max_retries": 3,
        "retry_delay": 1,
        "timeout": 30,
        "error_threshold": 5,
        "logging": {
            "level": "DEBUG",
            "format": "detailed"
        }
    }
}
```

### Agent Configuration
```json
{
    "agents": {
        "agent1": {
            "type": "specialized",
            "config": { ... }
        },
        "agent2": {
            "type": "general",
            "config": { ... }
        }
    }
}
```

## Usage in Both Systems

### Update Manifest System
The supervisor in `update_manifest.py` manages:
- FileReaderAgent
- ManifestUpdaterAgent
- LoggingConfigAgent
- ValidationAgent

### Toy Example System
The supervisor in `toy_example.py` manages:
- FileReaderAgent
- WriterAgent
- VerifierAgent
- QualityAgent

## Common Features

### 1. Task Processing
Both implementations:
- Use the same task validation logic
- Implement identical retry mechanisms
- Follow the same error handling patterns
- Maintain consistent logging

### 2. Agent Management
Both systems:
- Initialize agents with proper configuration
- Monitor agent health
- Handle agent failures
- Maintain agent state

### 3. Communication
Both implementations:
- Use the same message routing system
- Maintain conversation history
- Handle communication failures
- Implement retry logic

### 4. Error Handling
Both systems:
- Use the same error recovery strategies
- Implement identical retry mechanisms
- Maintain error counts
- Log errors consistently

## Best Practices

### 1. Agent Initialization
- Always validate agent configuration
- Initialize agents in the correct order
- Handle initialization failures
- Log initialization status

### 2. Task Management
- Validate tasks before processing
- Maintain task state
- Handle task failures
- Implement proper cleanup

### 3. Error Handling
- Use appropriate error types
- Implement retry mechanisms
- Maintain error context
- Log errors properly

### 4. Communication
- Validate messages
- Handle communication failures
- Maintain message history
- Implement proper routing

## Example Usage

### Basic Usage
```python
# Initialize supervisor
supervisor = SupervisorAgent(config)

# Start supervisor
await supervisor.start()

# Process task
result = await supervisor.process_task({
    "type": "file_processing",
    "content": "file_content",
    "metadata": { ... }
})

# Handle result
await supervisor.handle_result(result)
```

### Error Handling
```python
try:
    await supervisor.process_task(task)
except Exception as e:
    await supervisor.handle_error(e, {
        "task": task,
        "context": "processing"
    })
```

## Troubleshooting

### Common Issues
1. **Agent Initialization Failures**
   - Check agent configuration
   - Verify dependencies
   - Check resource availability

2. **Task Processing Errors**
   - Validate task format
   - Check agent availability
   - Verify task dependencies

3. **Communication Issues**
   - Check message format
   - Verify agent status
   - Check network connectivity

4. **Error Recovery**
   - Check error logs
   - Verify retry configuration
   - Check resource limits 