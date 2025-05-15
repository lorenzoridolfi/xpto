# Agent Tracing and Root Cause Analysis

This document describes the tracing system, root cause analysis, and analytics for agents.

## Overview

The system consists of three main modules:

1. `agent_tracer.py`: Tracks agent events and messages
2. `root_cause_analyzer.py`: Analyzes traces to identify issues
3. `tool_analytics.py`: Analyzes tool usage by agents

## Modules

### 1. AgentTracer

The `AgentTracer` is responsible for tracking all agent interactions.

#### Data Structures

```python
@dataclass
class TokenUsage:
    """LLM token usage statistics."""
    prompt_tokens: int      # Tokens used in prompt
    completion_tokens: int  # Tokens used in response
    total_tokens: int       # Total tokens
    model: str             # Model used (e.g., gpt-4)

@dataclass
class AgentEvent:
    """Agent event."""
    timestamp: str                    # Event date/time
    agent_name: str                   # Agent name
    event_type: str                   # Type (invoke/complete)
    inputs: List[Dict[str, str]]      # Input messages
    outputs: List[Dict[str, Any]]     # Output messages
    metadata: Optional[Dict[str, Any]] # Extra metadata
    token_usage: Optional[TokenUsage]  # Token usage
```

#### Features

1. **Event Tracing**
   ```python
   # Start processing
   tracer.on_messages_invoke("WriterAgent", messages, token_usage)
   
   # End processing
   tracer.on_messages_complete("WriterAgent", outputs, token_usage)
   ```

2. **Token Tracing**
   - Records prompt and completion tokens
   - Identifies model used
   - Calculates total tokens

3. **Time Tracing**
   - Start time
   - Processing time
   - End time

4. **Persistence**
   - Saves traces to JSON
   - Maintains event history
   - Enables later analysis

### 2. RootCauseAnalyzer

The `RootCauseAnalyzer` analyzes traces to identify issues and generate recommendations.

#### Data Structures

```python
@dataclass
class RootCauseAnalysis:
    """Root cause analysis result."""
    summary: str                    # Analysis summary
    issues: List[Dict[str, Any]]    # Issues found
    recommendations: List[str]      # Recommendations
    metadata: Optional[Dict[str, Any]] # Extra metadata
```

#### Analysis Types

1. **Performance**
   - Processing time
   - Token usage
   - Communication bottlenecks

2. **Communication**
   - Message patterns
   - Communication volume
   - Flow efficiency

3. **Errors**
   - Exceptions and errors
   - Validation failures
   - Integration issues

4. **User Feedback**
   - Reported issues
   - Improvement suggestions
   - User experience

#### Features

1. **Event Analysis**
   ```python
   # Complete analysis
   analysis = analyzer.analyze(tracer)
   
   # Analysis with feedback
   analysis = analyzer.analyze(tracer, user_feedback="System slow")
   ```

2. **Recommendation Generation**
   - Based on found issues
   - Considering user feedback
   - Prioritized by severity

3. **Persistence**
   - Saves analyses to JSON
   - Maintains issue history
   - Enables improvement tracking

### 3. ToolAnalytics

The `ToolAnalytics` analyzes tool usage by agents.

#### Data Structures

```python
@dataclass
class ToolUsage:
    """Tool usage statistics."""
    tool_name: str                  # Tool name
    call_count: int                 # Number of calls
    success_rate: float             # Success rate
    avg_duration: float             # Average duration
    error_count: int                # Error count
    last_used: str                  # Last used
```

#### Analysis Types

1. **Tool Usage**
   - Usage frequency
   - Success rate
   - Execution time

2. **Usage Patterns**
   - Tool sequence
   - Common combinations
   - Dependencies

3. **Issues**
   - Frequent errors
   - Timeouts
   - Integration failures

#### Features

1. **Usage Analysis**
   ```python
   # Tool analysis
   usage = analytics.analyze_tool("search_tool")
   
   # Agent analysis
   usage = analytics.analyze_agent("WriterAgent")
   ```

2. **Recommendations**
   - Usage optimization
   - Tool replacement
   - Integration improvements

3. **Persistence**
   - Saves analytics to JSON
   - Maintains usage history
   - Enables trend analysis

## Integration

### Configuration

```json
{
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "agent_tracer.log",
    "console": true
  },
  "analysis": {
    "performance": {
      "response_time_threshold": 5.0,
      "error_rate_threshold": 0.1
    },
    "token_tracking": {
      "enabled": true
    }
  }
}
```

### Usage in Projects

1. **Initialization**
   ```python
   # Load configuration
   with open("config.json", "r") as f:
       config = json.load(f)
   
   # Initialize modules
   tracer = AgentTracer(config)
   analyzer = RootCauseAnalyzer(config)
   analytics = ToolAnalytics(config)
   ```

2. **Tracing**
   ```python
   # During execution
   tracer.on_messages_invoke(agent_name, messages, token_usage)
   tracer.on_messages_complete(agent_name, outputs, token_usage)
   ```

3. **Analysis**
   ```python
   # Root cause analysis
   analysis = analyzer.analyze(tracer)
   
   # Tool analysis
   tool_analysis = analytics.analyze_tool("search_tool")
   ```

4. **Persistence**
   ```python
   # Save traces
   tracer.save_trace("trace.json")
   
   # Save analyses
   analyzer.save_analysis(analysis, "analysis.json")
   analytics.save_analysis(tool_analysis, "tool_analysis.json")
   ```

## Best Practices

1. **Tracing**
   - Trace all important events
   - Include relevant metadata
   - Keep traces organized

2. **Analysis**
   - Analyze traces regularly
   - Consider user feedback
   - Implement recommendations

3. **Tools**
   - Monitor tool usage
   - Optimize usage patterns
   - Maintain issue history

4. **Configuration**
   - Adjust thresholds as needed
   - Configure logging appropriately
   - Keep configurations updated

## Examples

### Complete Trace
```python
# Initialization
tracer = AgentTracer(config)
analyzer = RootCauseAnalyzer(config)
analytics = ToolAnalytics(config)

# During execution
tracer.on_messages_invoke("WriterAgent", messages, token_usage)
writer_output = writer_agent.process_text()
tracer.on_messages_complete("WriterAgent", writer_output, token_usage)

# Analysis
analysis = analyzer.analyze(tracer)
tool_analysis = analytics.analyze_tool("search_tool")

# Save results
tracer.save_trace("trace.json")
analyzer.save_analysis(analysis, "analysis.json")
analytics.save_analysis(tool_analysis, "tool_analysis.json")
```

### Error Analysis
```python
try:
    # Normal processing
    tracer.on_messages_invoke("WriterAgent", messages)
    writer_output = writer_agent.process_text()
    tracer.on_messages_complete("WriterAgent", writer_output)
except Exception as e:
    # Error trace
    tracer.on_messages_invoke("Error", [{"source": "system", "content": str(e)}])
    tracer.save_trace("error_trace.json")
    
    # Error analysis
    analysis = analyzer.analyze(tracer, user_feedback=f"Error: {str(e)}")
    analyzer.save_analysis(analysis, "error_analysis.json")
    
    # Tool analysis
    tool_analysis = analytics.analyze_tool("search_tool")
    analytics.save_analysis(tool_analysis, "error_tool_analysis.json")
```

# Agent Tracing and Autogen Integration

## Overview

The `AgentTracer` is an observability component that integrates with Autogen agents to provide:
- Event tracking
- Token usage metrics
- Cache statistics
- Detailed logs
- Performance analysis

## Architecture

```
Autogen Agents (AssistantAgent, UserProxyAgent)
        ↓
AgentTracer (Observer)
        ↓
Logs, Metrics, Traces
```

### Components

1. **Autogen Agents**
   - `AssistantAgent`: Main message processing agent
   - `UserProxyAgent`: User interface
   - `GroupChat`: Agent coordination

2. **AgentTracer**
   - Agent observer
   - Metric collector
   - Log generator
   - Statistics calculator

3. **Logging System**
   - Log files
   - Console output
   - Custom formatting

## Integration

### 1. Initialization

```python
from autogen import AssistantAgent, UserProxyAgent
from agent_tracer import AgentTracer, TokenUsage

# Tracer configuration
config = {
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "agent_trace.log",
        "console": true
    }
}

# Create tracer
tracer = AgentTracer(config)

# Create agents
assistant = AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config={"config_list": [{"model": "gpt-4"}]}
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE"
)
```

### 2. Event Tracing

```python
# Before processing messages
tracer.on_messages_invoke(
    agent_name="assistant",
    messages=[{"role": "user", "content": "Hello"}],
    token_usage=TokenUsage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        model="gpt-4"
    ),
    cache_hit=False
)

# After processing messages
tracer.on_messages_complete(
    agent_name="assistant",
    outputs=[{"role": "assistant", "content": "Hi there!"}],
    token_usage=TokenUsage(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        model="gpt-4"
    ),
    cache_hit=True
)
```

### 3. Cache Integration

```python
from llm_cache import LLMCache

# Initialize cache
cache = LLMCache(
    max_size=1000,
    similarity_threshold=0.85,
    expiration_hours=24
)

# Check cache before LLM call
cache_key = generate_cache_key(messages)
cached_response = cache.get(cache_key)

if cached_response:
    # Cache hit
    tracer.on_messages_complete(
        agent_name="assistant",
        outputs=cached_response,
        token_usage=TokenUsage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            model="gpt-4"
        ),
        cache_hit=True,
        cache_key=cache_key
    )
else:
    # Cache miss
    response = assistant.generate_response(messages)
    cache.set(cache_key, response)
    
    tracer.on_messages_complete(
        agent_name="assistant",
        outputs=response,
        token_usage=TokenUsage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            model="gpt-4"
        ),
        cache_hit=False
    )
```

### 4. Performance Analysis

```python
# Get cache statistics
cache_stats = tracer.get_cache_statistics()
print(f"Cache hit rate: {cache_stats['savings_percentage']}%")
print(f"Total tokens saved: {cache_stats['total_savings']}")

# Save complete trace
tracer.save_trace("agent_trace.json")
```

## Complete Example

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat
from agent_tracer import AgentTracer, TokenUsage
from llm_cache import LLMCache

def setup_agents_with_tracing():
    # Configuration
    config = {
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "agent_trace.log",
            "console": true
        }
    }
    
    # Initialize components
    tracer = AgentTracer(config)
    cache = LLMCache(max_size=1000)
    
    # Create agents
    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant.",
        llm_config={"config_list": [{"model": "gpt-4"}]}
    )
    
    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="TERMINATE"
    )
    
    # Create group chat
    groupchat = GroupChat(
        agents=[user_proxy, assistant],
        messages=[],
        max_round=10
    )
    
    # Tracing wrapper function
    def traced_chat(agent_name, messages):
        # Start trace
        tracer.on_messages_invoke(agent_name, messages)
        
        # Check cache
        cache_key = generate_cache_key(messages)
        cached_response = cache.get(cache_key)
        
        if cached_response:
            # Cache hit
            tracer.on_messages_complete(
                agent_name,
                cached_response,
                token_usage=TokenUsage(
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    model="gpt-4"
                ),
                cache_hit=True,
                cache_key=cache_key
            )
            return cached_response
        
        # Cache miss - normal processing
        response = assistant.generate_response(messages)
        cache.set(cache_key, response)
        
        # End trace
        tracer.on_messages_complete(
            agent_name,
            response,
            token_usage=TokenUsage(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                model="gpt-4"
            ),
            cache_hit=False
        )
        
        return response
    
    return {
        "tracer": tracer,
        "cache": cache,
        "assistant": assistant,
        "user_proxy": user_proxy,
        "groupchat": groupchat,
        "traced_chat": traced_chat
    }

# Usage
def main():
    # Setup
    components = setup_agents_with_tracing()
    
    # Start chat
    components["user_proxy"].initiate_chat(
        components["groupchat"],
        message="Hello, how can you help me?"
    )
    
    # Final analysis
    cache_stats = components["tracer"].get_cache_statistics()
    print(f"Cache hit rate: {cache_stats['savings_percentage']}%")
    print(f"Total tokens saved: {cache_stats['total_savings']}")
    
    # Save trace
    components["tracer"].save_trace("chat_trace.json")
```

## Best Practices

1. **Configuration**
   - Configure logging appropriately
   - Adjust log levels as needed
   - Use consistent log formats

2. **Tracing**
   - Trace all important events
   - Include relevant metadata
   - Keep traces organized

3. **Cache**
   - Use consistent cache keys
   - Monitor hit rates
   - Adjust thresholds as needed

4. **Performance**
   - Monitor token usage
   - Track response times
   - Analyze usage patterns

5. **Maintenance**
   - Clean old traces
   - Rotate logs
   - Keep statistics updated

## Troubleshooting

1. **Logs not appearing**
   - Check logging configuration
   - Confirm log levels
   - Check file permissions

2. **Cache not working**
   - Check cache keys
   - Confirm thresholds
   - Monitor hit rates

3. **Poor performance**
   - Analyze traces
   - Check token usage
   - Optimize configurations

4. **Integration errors**
   - Check call order
   - Confirm data types
   - Validate configurations 