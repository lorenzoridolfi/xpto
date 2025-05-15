# LLM Usage Tracing

## Overview

The agent tracer (`src/agent_tracer.py`) provides comprehensive tracing of LLM interactions.

## Key Features

1. **Token Usage Tracking**
   - Prompt tokens
   - Completion tokens
   - Total tokens
   - Model information

2. **Cache Integration**
   - Cache hit/miss tracking
   - Cache savings calculation
   - Cache key management

3. **Performance Monitoring**
   - Response times
   - Token usage patterns
   - Cost tracking
   - Error monitoring

## Usage Example

```python
from src.agent_tracer import AgentTracer, TokenUsage

# Initialize tracer
tracer = AgentTracer(config={
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "agent_trace.log",
        "console": True
    }
})

# Record token usage
token_usage = TokenUsage(
    prompt_tokens=100,
    completion_tokens=50,
    total_tokens=150,
    model="gpt-4",
    cache_hit=True,
    cache_key="abc123",
    cache_savings={"prompt": 100, "completion": 50}
)

# Record event
tracer.on_messages_invoke(
    agent_name="assistant",
    messages=messages,
    token_usage=token_usage,
    cache_hit=True
)
```

## Configuration

```python
tracer_config = {
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "agent_trace.log",
        "console": True
    },
    "token_tracking": True,
    "cache_tracking": True,
    "performance_tracking": True
}
``` 