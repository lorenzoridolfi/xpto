# Agent Tracing System

This document describes the agent tracing and monitoring system implemented in `src/agent_tracer.py`.

## Overview

The Agent Tracing System provides comprehensive monitoring and analytics capabilities for agent interactions, including:
- Detailed event logging
- Token usage tracking
- Performance monitoring
- Cache effectiveness analysis
- Comprehensive metrics collection

## Key Components

### Token Usage Tracking

```python
@dataclass
class TokenUsage:
    """Represents token usage statistics for an LLM call."""
```

Features:
- Prompt token counting
- Completion token tracking
- Total token calculation
- Cache hit/miss tracking
- Savings calculation

### Event Tracking

```python
@dataclass
class AgentEvent:
    """Represents an event in the agent's execution."""
```

Features:
- Timestamp recording
- Agent identification
- Event type tracking
- Input/output logging
- Metadata storage
- Token usage statistics

### Agent Tracer

```python
class AgentTracer:
    """Tracks agent events and messages with comprehensive monitoring capabilities."""
```

Features:
- Event recording
- Token usage tracking
- Cache statistics
- Performance metrics
- Analytics generation

## Usage Example

```python
# Initialize tracer
tracer = AgentTracer({
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "agent_trace.log",
        "console": True
    }
})

# Record message processing start
tracer.on_messages_invoke(
    agent_name="MyAgent",
    messages=[{"role": "user", "content": "Hello"}],
    token_usage=TokenUsage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        model="gpt-4"
    ),
    cache_hit=False
)

# Record message processing completion
tracer.on_messages_complete(
    agent_name="MyAgent",
    outputs=[{"role": "assistant", "content": "Hi there!"}],
    token_usage=TokenUsage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        model="gpt-4"
    ),
    cache_hit=False
)

# Get cache statistics
stats = tracer.get_cache_statistics()
print(f"Cache savings: {stats['savings_percentage']}%")

# Save trace to file
tracer.save_trace("agent_trace.json")
```

## Configuration Structure

```json
{
    "logging": {
        "level": "string",
        "format": "string",
        "file": "string",
        "console": "boolean"
    }
}
```

## Best Practices

1. **Event Tracking**
   - Record all significant events
   - Include detailed metadata
   - Track timing information
   - Log error conditions

2. **Token Usage**
   - Monitor all LLM calls
   - Track cache effectiveness
   - Calculate savings
   - Optimize token usage

3. **Performance Monitoring**
   - Track processing times
   - Monitor cache hits
   - Analyze bottlenecks
   - Optimize performance

4. **Logging**
   - Use appropriate log levels
   - Include context information
   - Format messages clearly
   - Rotate log files

## Integration

To integrate the Agent Tracing System:

1. **Setup**
   - Import AgentTracer
   - Configure logging
   - Initialize tracer
   - Set up monitoring

2. **Implementation**
   - Record events
   - Track token usage
   - Monitor performance
   - Generate reports

3. **Configuration**
   - Set log levels
   - Configure output
   - Define formats
   - Enable features

4. **Testing**
   - Test event recording
   - Verify token tracking
   - Check cache stats
   - Validate reports

## Maintenance

Regular maintenance tasks:

1. **Logging**
   - Review log levels
   - Rotate log files
   - Clean up old logs
   - Monitor disk usage

2. **Performance**
   - Review metrics
   - Optimize tracking
   - Update thresholds
   - Clean up data

3. **Cache**
   - Monitor effectiveness
   - Update strategies
   - Clean up old entries
   - Optimize savings

4. **Reports**
   - Generate summaries
   - Analyze trends
   - Update formats
   - Archive old data 