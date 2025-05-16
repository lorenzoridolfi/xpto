"""
Supervisor Agent Module

This module implements a supervisor agent that manages and coordinates other agents
using AutoGen's GroupChat functionality. It provides:
- Agent lifecycle management
- Task processing and coordination
- Error handling and retry mechanisms
- System metrics and monitoring

The supervisor agent serves as the central coordinator for multi-agent systems,
ensuring proper communication and task execution between agents.
"""

from typing import Dict, Any, List, Optional, Union, TypedDict, TypeVar, Generic
import asyncio
from datetime import datetime
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from .logger import LoggerMixin
from .base_supervisor import TaskDict, TaskResult, AgentStatus, SystemMetrics

# Custom Exceptions
class SupervisorAgentError(Exception):
    """Base exception for supervisor agent-related errors."""
    pass

class ConfigurationError(SupervisorAgentError):
    """Raised when supervisor agent configuration is invalid."""
    pass

class AgentError(SupervisorAgentError):
    """Raised when agent operations fail."""
    pass

class TaskError(SupervisorAgentError):
    """Raised when task processing fails."""
    pass

# Type Definitions
class AgentConfig(TypedDict):
    name: str
    system_message: str
    llm_config: Dict[str, Any]

T = TypeVar('T')

class SupervisorAgent(LoggerMixin):
    """Base supervisor agent class that manages and coordinates other agents using AutoGen's GroupChat.
    
    This class implements the core supervisor agent functionality for managing and
    coordinating multiple agents in a group chat environment. It handles agent
    lifecycle, task processing, error handling, and system monitoring.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the supervisor agent.
        
        Args:
            config: Configuration dictionary containing agent settings
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            super().__init__()
            self.config = config
            self.agents: Dict[str, Union[AssistantAgent, UserProxyAgent]] = {}
            self.group_chat: Optional[GroupChat] = None
            self.group_chat_manager: Optional[GroupChatManager] = None
            self.error_count: int = 0
            self.max_retries: int = config.get("max_retries", 3)
            self.retry_delay: int = config.get("retry_delay", 1)
            self.timeout: int = config.get("timeout", 30)
            self.error_threshold: int = config.get("error_threshold", 5)
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize supervisor agent: {str(e)}")
        
    async def initialize_agents(self) -> None:
        """Initialize all subordinate agents with proper configuration.
        
        Raises:
            AgentError: If agent initialization fails
        """
        try:
            # Create base agents
            for agent_name, agent_config in self.config["agents"].items():
                agent = await self._create_agent(agent_name, agent_config)
                self.agents[agent_name] = agent
                self.log_info(f"Initialized {agent_name}", config=agent_config)
            
            # Create user proxy agent
            user_proxy = UserProxyAgent(
                name=self.config["system"]["user_proxy"]["name"],
                human_input_mode=self.config["system"]["user_proxy"]["human_input_mode"]
            )
            self.agents["user_proxy"] = user_proxy
            
            # Create group chat
            self.group_chat = GroupChat(
                agents=list(self.agents.values()),
                messages=[],
                max_round=self.config["group_chat"]["max_round"]
            )
            
            # Create group chat manager
            self.group_chat_manager = GroupChatManager(
                groupchat=self.group_chat,
                llm_config=self.config["llm_config"]["supervisor"]
            )
            
            self.log_info("Group chat initialized", 
                         agent_count=len(self.agents),
                         max_rounds=self.config["group_chat"]["max_round"])
                         
        except Exception as e:
            self.log_error("Failed to initialize agents", error=str(e))
            raise AgentError(f"Agent initialization failed: {str(e)}")
            
    async def _create_agent(self, name: str, config: Dict[str, Any]) -> AssistantAgent:
        """Create an agent instance based on its type and configuration.
        
        Args:
            name: Name of the agent
            config: Agent configuration
            
        Returns:
            Initialized AssistantAgent instance
            
        Raises:
            AgentError: If agent creation fails
        """
        try:
            return AssistantAgent(
                name=name,
                system_message=config["system_message"],
                llm_config=self.config["llm_config"][name]
            )
        except Exception as e:
            raise AgentError(f"Failed to create agent {name}: {str(e)}")
            
    async def process_task(self, task: TaskDict) -> TaskResult:
        """Process a single task through the agent pipeline.
        
        Args:
            task: Task dictionary containing task details
            
        Returns:
            Task result dictionary
            
        Raises:
            TaskError: If task processing fails
        """
        try:
            if not self._validate_task(task):
                raise TaskError("Invalid task format")
                
            # Convert task to chat message
            message = self._task_to_message(task)
            
            # Process through group chat
            result = await self.group_chat_manager.run(message)
            
            # Convert result back to task format
            task_result = self._message_to_task(result)
            await self._handle_result(task_result)
            
            return task_result
        except Exception as e:
            self.log_error("Task processing failed", error=str(e), task=task)
            raise TaskError(f"Task processing failed: {str(e)}")
            
    def _validate_task(self, task: TaskDict) -> bool:
        """Validate task format and content.
        
        Args:
            task: Task dictionary to validate
            
        Returns:
            True if task is valid, False otherwise
        """
        required_fields = ["type", "content"]
        return all(field in task for field in required_fields)
        
    def _task_to_message(self, task: TaskDict) -> str:
        """Convert task dictionary to chat message.
        
        Args:
            task: Task dictionary to convert
            
        Returns:
            Formatted chat message
        """
        return f"Task Type: {task['type']}\nContent: {task['content']}\nMetadata: {task.get('metadata', {})}"
        
    def _message_to_task(self, message: str) -> TaskResult:
        """Convert chat message to task result.
        
        Args:
            message: Chat message to convert
            
        Returns:
            Task result dictionary
        """
        return {
            "status": "success",
            "content": message,
            "timestamp": datetime.now().isoformat(),
            "error": None
        }
        
    async def _handle_result(self, result: TaskResult) -> None:
        """Handle task result and update system state.
        
        Args:
            result: Task result dictionary
            
        Raises:
            TaskError: If result handling fails
        """
        try:
            if result.get("status") == "error":
                await self.handle_error(
                    Exception(result.get("error", "Unknown error")),
                    {"result": result}
                )
        except Exception as e:
            raise TaskError(f"Failed to handle result: {str(e)}")
            
    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Handle errors in agent operations.
        
        Args:
            error: Exception that occurred
            context: Additional context about the error
            
        Raises:
            AgentError: If max retries exceeded
        """
        self.error_count += 1
        
        if self.error_count > self.max_retries:
            self.log_error("Max retries exceeded", error=str(error))
            raise AgentError(f"Max retries exceeded: {str(error)}")
            
        self.log_warning(
            "Operation failed, retrying",
            error=str(error),
            retry_count=self.error_count,
            context=context
        )
        
        await asyncio.sleep(self.retry_delay)
        await self._retry_operation(context)
        
    async def _retry_operation(self, context: Dict[str, Any]) -> None:
        """Retry failed operation.
        
        Args:
            context: Context of the failed operation
            
        Raises:
            TaskError: If retry operation fails
        """
        try:
            task = context.get("task")
            if task:
                await self.process_task(task)
        except Exception as e:
            raise TaskError(f"Retry operation failed: {str(e)}")
            
    async def start(self) -> None:
        """Start the supervisor agent and initialize all components.
        
        Raises:
            SupervisorAgentError: If supervisor startup fails
        """
        try:
            await self.initialize_agents()
            self.log_info("Supervisor agent started")
        except Exception as e:
            raise SupervisorAgentError(f"Failed to start supervisor agent: {str(e)}")
        
    async def stop(self) -> None:
        """Stop the supervisor agent and cleanup resources.
        
        Raises:
            SupervisorAgentError: If supervisor shutdown fails
        """
        try:
            self.log_info("Supervisor agent stopped")
        except Exception as e:
            raise SupervisorAgentError(f"Failed to stop supervisor agent: {str(e)}")
        
    def get_agent_status(self) -> Dict[str, AgentStatus]:
        """Get status of all managed agents.
        
        Returns:
            Dictionary containing agent status information
        """
        return {
            name: {
                "status": "active" if agent else "inactive",
                "error_count": getattr(agent, "error_count", 0),
                "last_active": getattr(agent, "last_active", None)
            }
            for name, agent in self.agents.items()
        }
        
    def get_system_metrics(self) -> SystemMetrics:
        """Get system-wide metrics.
        
        Returns:
            Dictionary containing system metrics
        """
        active_agents = sum(1 for status in self.get_agent_status().values() 
                          if status["status"] == "active")
        
        return {
            "agent_count": len(self.agents),
            "active_agents": active_agents,
            "error_count": self.error_count,
            "total_tasks": 0,  # TODO: Implement task tracking
            "successful_tasks": 0,  # TODO: Implement task tracking
            "failed_tasks": 0,  # TODO: Implement task tracking
            "average_processing_time": 0.0  # TODO: Implement processing time calculation
        } 