# Supervisor Agent

## Overview

The Supervisor Agent is a specialized agent responsible for orchestrating the workflow between different agents in the system. It ensures proper coordination, task distribution, and process management through a combination of direct supervision and group chat management.

## Unified Implementation

The supervisor agent is implemented as a combination of two components:

1. **Supervisor Agent** (Base Component)
```python
supervisor = AssistantAgent(
    name=supervisor_config["name"],
    system_message=supervisor_config["system_message"],
    llm_config=config["llm_config"]["supervisor"]
)
```

2. **Group Chat Manager** (Coordination Component)
```python
group_chat_manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=config["llm_config"]["supervisor"]
)
```

### Key Features

1. **Workflow Management**
   - Task distribution and coordination
   - Process monitoring and control
   - Progress tracking
   - Error handling and recovery

2. **Agent Coordination**
   - Inter-agent communication management
   - Message routing and aggregation
   - State management
   - Conflict resolution

3. **Quality Control**
   - Process validation
   - Output verification
   - Performance monitoring
   - Error detection and handling

4. **System Optimization**
   - Resource allocation
   - Performance tracking
   - Cache management
   - API efficiency monitoring

## Configuration

The supervisor agent can be configured through the system's configuration file:

```json
{
    "agents": {
        "supervisor": {
            "name": "supervisor",
            "description": "Coordinates the workflow between agents",
            "system_message": "You are a supervisor agent responsible for coordinating the workflow between different agents. Your role includes task distribution, process monitoring, and ensuring quality control.",
            "llm_config": {
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 4096
            }
        }
    },
    "group_chat": {
        "max_round": 10,
        "speaker_selection_method": "round_robin",
        "allow_repeat_speaker": false
    }
}
```

## Usage Example

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

def create_supervisor_system(config):
    # Create base agents
    creator = AssistantAgent(
        name=config["agents"]["creator"]["name"],
        system_message=config["agents"]["creator"]["system_message"],
        llm_config=config["llm_config"]["creator"]
    )
    
    critic = AssistantAgent(
        name=config["agents"]["critic"]["name"],
        system_message=config["agents"]["critic"]["system_message"],
        llm_config=config["llm_config"]["critic"]
    )
    
    # Create supervisor
    supervisor = AssistantAgent(
        name=config["agents"]["supervisor"]["name"],
        system_message=config["agents"]["supervisor"]["system_message"],
        llm_config=config["llm_config"]["supervisor"]
    )
    
    # Create user proxy
    user_proxy = UserProxyAgent(
        name=config["system"]["user_proxy"]["name"],
        human_input_mode=config["system"]["user_proxy"]["human_input_mode"]
    )
    
    # Create group chat
    groupchat = GroupChat(
        agents=[user_proxy, creator, critic, supervisor],
        messages=[],
        max_round=config["group_chat"]["max_round"]
    )
    
    # Create group chat manager
    group_chat_manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=config["llm_config"]["supervisor"]
    )
    
    return {
        "supervisor": supervisor,
        "group_chat_manager": group_chat_manager,
        "groupchat": groupchat
    }
```

## Best Practices

1. **Workflow Design**
   - Define clear agent roles and responsibilities
   - Establish communication protocols
   - Set process boundaries
   - Implement error handling

2. **Performance Optimization**
   - Monitor agent interactions
   - Track resource usage
   - Optimize message flow
   - Manage cache effectively

3. **Error Handling**
   - Implement retry logic
   - Log error events
   - Provide recovery mechanisms
   - Maintain system stability

4. **Monitoring**
   - Track agent performance
   - Monitor system metrics
   - Log important events
   - Generate analytics reports

## Integration with Analytics

The supervisor agent integrates with the analytics system to provide comprehensive monitoring and optimization:

```python
from tool_analytics import ToolAnalytics

# Initialize analytics
analytics = ToolAnalytics()

# Record supervisor events
analytics.record_usage(ToolUsage(
    timestamp=datetime.now(),
    tool_name="supervisor",
    parameters={"action": "coordinate", "agents": ["creator", "critic"]},
    duration=0.5,
    success=True
))

# Get performance metrics
metrics = analytics.get_tool_metrics("supervisor")
print(f"Success rate: {metrics.success_rate}")
print(f"Average duration: {metrics.avg_duration}")
``` 