# Trace Mechanism Documentation

## Overview
The trace mechanism provides a comprehensive system for tracking and monitoring agent interactions, decisions, and system events in the AutoGen-based multi-agent system. It is implemented through the `SupervisorTrace` class and integrated with the `BaseSupervisor` to provide centralized tracing capabilities.

## Components

### 1. SupervisorTrace Class
The core tracing component that handles all trace-related operations.

```python
class SupervisorTrace(LoggerMixin):
    def __init__(self, config: Dict[str, Any]):
        self.traces = []  # List of all traces
        self.current_trace = None  # Active trace
        self.trace_enabled = config.get("tracing", {}).get("enabled", True)
        self.trace_level = config.get("tracing", {}).get("level", "INFO")
```

### 2. Trace Structure
Each trace contains the following information:
```python
{
    "task_id": str,  # Unique identifier for the task
    "task_type": str,  # Type of task being processed
    "start_time": str,  # ISO format timestamp
    "end_time": str,  # ISO format timestamp
    "events": List[Dict],  # System events
    "agent_interactions": List[Dict],  # Agent communication
    "decisions": List[Dict],  # Supervisor decisions
    "result": Dict,  # Task result
    "status": str  # Task status
}
```

## Trace Events

### 1. Agent Creation
```python
{
    "type": "agent_creation",
    "timestamp": str,
    "agent_name": str,
    "agent_type": str,
    "config": Dict
}
```

### 2. Agent Interactions
```python
{
    "timestamp": str,
    "sender": str,
    "recipient": str,
    "message": str
}
```

### 3. Supervisor Decisions
```python
{
    "timestamp": str,
    "type": str,
    "context": Dict
}
```

### 4. Error Events
```python
{
    "type": "error",
    "timestamp": str,
    "error_type": str,
    "error_message": str,
    "context": Dict
}
```

## Configuration

The trace mechanism is configured through the `global_config.json` file:

```json
{
    "tracing": {
        "enabled": true,
        "level": "DEBUG",
        "persist_traces": true,
        "max_traces": 1000,
        "trace_retention": "7d",
        "events": {
            "agent_creation": true,
            "agent_interaction": true,
            "task_processing": true,
            "error_handling": true,
            "decision_making": true
        },
        "metrics": {
            "enabled": true,
            "collect_interval": 60,
            "retention_period": "30d"
        }
    }
}
```

## Usage

### 1. Starting a Trace
```python
# In BaseSupervisor
async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
    self.trace.start_trace(task)
    # ... process task ...
    self.trace.end_trace(result)
```

### 2. Tracing Agent Creation
```python
# In BaseSupervisor
async def initialize_agents(self) -> None:
    agent = await self._create_agent(agent_name, agent_config)
    self.trace.trace_agent_creation(agent, agent_config)
```

### 3. Tracing Agent Interactions
```python
# In BaseSupervisor
async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
    # ... process task ...
    self.trace.trace_agent_interaction(sender, recipient, message)
```

### 4. Tracing Decisions
```python
# In BaseSupervisor
async def handle_error(self, error: Exception, context: Dict[str, Any]) -> None:
    self.trace.trace_decision("retry_operation", {
        "error": str(error),
        "retry_count": self.error_count
    })
```

### 5. Tracing Errors
```python
# In BaseSupervisor
async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # ... process task ...
    except Exception as e:
        self.trace.trace_error(e, {"task": task})
```

## Accessing Traces

### 1. Get Specific Trace
```python
trace = supervisor.get_trace(task_id)
```

### 2. Get All Traces
```python
all_traces = supervisor.get_all_traces()
```

### 3. Get Trace Metrics
```python
metrics = supervisor.get_trace_metrics()
```

## Trace Metrics

The trace system provides the following metrics:
- Total number of traces
- Number of active traces
- Total number of events
- Total number of agent interactions
- Total number of decisions
- Error count

## Best Practices

1. **Trace Level Selection**
   - Use DEBUG level for development
   - Use INFO level for production
   - Configure appropriate log handlers

2. **Trace Retention**
   - Set appropriate max_traces limit
   - Configure trace_retention period
   - Regularly clear old traces

3. **Error Handling**
   - Always trace errors with context
   - Include relevant task information
   - Maintain error count metrics

4. **Performance Considerations**
   - Disable tracing in production if not needed
   - Use appropriate trace levels
   - Monitor trace storage usage

## Integration with AutoGen

The trace mechanism integrates with AutoGen's native capabilities:
- Uses AutoGen's Agent and GroupChat classes
- Captures agent interactions in group chats
- Maintains conversation history
- Tracks agent states

## Example Trace Output

```python
{
    "task_id": "task_123",
    "task_type": "file_processing",
    "start_time": "2024-03-21T10:00:00",
    "events": [
        {
            "type": "agent_creation",
            "timestamp": "2024-03-21T10:00:00",
            "agent_name": "file_reader",
            "agent_type": "FileReaderAgent"
        }
    ],
    "agent_interactions": [
        {
            "timestamp": "2024-03-21T10:00:01",
            "sender": "file_reader",
            "recipient": "writer",
            "message": "Processing file content..."
        }
    ],
    "decisions": [
        {
            "timestamp": "2024-03-21T10:00:02",
            "type": "task_start",
            "context": {"task": {...}}
        }
    ],
    "end_time": "2024-03-21T10:00:03",
    "result": {"status": "success"},
    "status": "completed"
}
``` 