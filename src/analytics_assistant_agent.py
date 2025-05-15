"""
Analytics Assistant Agent

This module implements an analytics assistant agent that extends the base AssistantAgent
with analytics capabilities and tool usage tracking.
"""

import logging
from typing import Dict, List, Any, Optional
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage, BaseChatMessage
from autogen_agentchat.base import Response
from src.tool_analytics import ToolAnalytics, ToolUsageMetrics
from src.llm_cache import LLMCache

logger = logging.getLogger(__name__)

class AnalyticsAssistantAgent(AssistantAgent):
    """Assistant agent with analytics capabilities."""
    
    def __init__(
        self,
        name: str,
        description: str,
        llm_config: Dict[str, Any],
        tool_analytics: Optional[ToolAnalytics] = None,
        llm_cache: Optional[LLMCache] = None
    ):
        # Initialize parent class with required arguments
        super().__init__(name=name, llm_config=llm_config)
        self.description = description
        self.tool_analytics = tool_analytics or ToolAnalytics()
        self.llm_cache = llm_cache or LLMCache()
        
    async def run(self, task: str) -> str:
        """Run a task with analytics tracking."""
        logger.debug(f"Running task: {task}")
        
        # Check cache first
        cached_response = self.llm_cache.get(task)
        if cached_response:
            logger.debug("Using cached response")
            return cached_response
            
        # Execute task
        response = await super().run(task)
        
        # Cache response
        self.llm_cache.add(task, response)
        
        return response
        
    async def on_messages(self, messages: List[BaseChatMessage], cancellation_token) -> Response:
        """Process messages with analytics tracking."""
        logger.debug(f"Processing messages: {messages}")
        
        # Track tool usage
        self.tool_analytics.record_usage(self.name, "message_processing")
        
        # Process messages
        response = await super().on_messages(messages, cancellation_token)
        
        return response
        
    async def on_messages_stream(self, messages: List[BaseChatMessage], cancellation_token):
        """Process message stream with analytics tracking."""
        logger.debug(f"Processing message stream: {messages}")
        
        # Track tool usage
        self.tool_analytics.record_usage(self.name, "message_stream_processing")
        
        # Process message stream
        async for response in super().on_messages_stream(messages, cancellation_token):
            yield response 