from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from .logger import LoggerMixin

class SupervisorAgent(LoggerMixin):
    """Base supervisor agent class that manages and coordinates other agents using AutoGen's GroupChat."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the supervisor agent.
        
        Args:
            config: Configuration dictionary containing agent settings
        """
        super().__init__()
        self.config = config
        self.agents = {}
        self.group_chat = None
        self.group_chat_manager = None
        self.error_count = 0
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay", 1)
        self.timeout = config.get("timeout", 30)
        self.error_threshold = config.get("error_threshold", 5)
        
    async def initialize_agents(self) -> None:
        """Initialize all subordinate agents with proper configuration."""
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
            raise
            
    async def _create_agent(self, name: str, config: Dict[str, Any]) -> AssistantAgent:
        """Create an agent instance based on its type and configuration.
        
        Args:
            name: Name of the agent
            config: Agent configuration
            
        Returns:
            Initialized AssistantAgent instance
        """
        return AssistantAgent(
            name=name,
            system_message=config["system_message"],
            llm_config=self.config["llm_config"][name]
        )
            
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single task through the agent pipeline.
        
        Args:
            task: Task dictionary containing task details
            
        Returns:
            Task result dictionary
            
        Raises:
            ValueError: If task format is invalid
            Exception: If task processing fails
        """
        try:
            if not self._validate_task(task):
                raise ValueError("Invalid task format")
                
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
            raise
            
    def _validate_task(self, task: Dict[str, Any]) -> bool:
        """Validate task format and content.
        
        Args:
            task: Task dictionary to validate
            
        Returns:
            True if task is valid, False otherwise
        """
        required_fields = ["type", "content"]
        return all(field in task for field in required_fields)
        
    def _task_to_message(self, task: Dict[str, Any]) -> str:
        """Convert task dictionary to chat message.
        
        Args:
            task: Task dictionary to convert
            
        Returns:
            Formatted chat message
        """
        return f"Task Type: {task['type']}\nContent: {task['content']}\nMetadata: {task.get('metadata', {})}"
        
    def _message_to_task(self, message: str) -> Dict[str, Any]:
        """Convert chat message to task result.
        
        Args:
            message: Chat message to convert
            
        Returns:
            Task result dictionary
        """
        return {
            "status": "success",
            "content": message,
            "timestamp": datetime.now().isoformat()
        }
        
    async def _handle_result(self, result: Dict[str, Any]) -> None:
        """Handle task result and update system state.
        
        Args:
            result: Task result dictionary
        """
        if result.get("status") == "error":
            await self.handle_error(
                Exception(result.get("error", "Unknown error")),
                {"result": result}
            )
            
    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Handle errors in agent operations.
        
        Args:
            error: Exception that occurred
            context: Additional context about the error
            
        Raises:
            Exception: If max retries exceeded
        """
        self.error_count += 1
        
        if self.error_count > self.max_retries:
            self.log_error("Max retries exceeded", error=str(error))
            raise error
            
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
        """
        task = context.get("task")
        if task:
            await self.process_task(task)
            
    async def start(self) -> None:
        """Start the supervisor agent and initialize all components."""
        await self.initialize_agents()
        self.log_info("Supervisor agent started")
        
    async def stop(self) -> None:
        """Stop the supervisor agent and cleanup resources."""
        self.log_info("Supervisor agent stopped")
        
    def get_agent_status(self) -> Dict[str, Any]:
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
        
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics.
        
        Returns:
            Dictionary containing system metrics
        """
        return {
            "total_agents": len(self.agents),
            "active_agents": sum(1 for a in self.agents.values() if a),
            "error_count": self.error_count,
            "conversation_history_length": len(self.group_chat.messages) if self.group_chat else 0
        } 