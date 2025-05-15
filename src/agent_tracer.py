"""
Agent Tracing and Monitoring System

This module provides comprehensive tracing and monitoring capabilities for agent interactions,
including event tracking, token usage monitoring, and performance analytics.

The system enables:
- Detailed event logging of agent interactions
- Token usage tracking and optimization
- Performance monitoring and analysis
- Cache hit/miss tracking
- Comprehensive metrics collection
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import json
import logging
import time
from datetime import datetime

@dataclass
class TokenUsage:
    """
    Represents token usage statistics for an LLM call.
    
    Attributes:
        prompt_tokens (int): Number of tokens used in the prompt
        completion_tokens (int): Number of tokens used in the completion
        total_tokens (int): Total number of tokens used
        model (str): Name of the model used
        cache_hit (bool): Whether this was a cache hit
        cache_key (Optional[str]): Cache key if this was a cache hit
        cache_savings (Optional[Dict[str, int]]): Token savings from cache if applicable
    """
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    cache_hit: bool = False
    cache_key: Optional[str] = None
    cache_savings: Optional[Dict[str, int]] = None

@dataclass
class AgentEvent:
    """
    Represents an event in the agent's execution.
    
    Attributes:
        timestamp (str): ISO format timestamp of the event
        agent_name (str): Name of the agent generating the event
        event_type (str): Type of event (e.g., 'invoke', 'complete')
        inputs (List[Dict[str, str]]): Input messages for the event
        outputs (List[Dict[str, Any]]): Output messages from the event
        metadata (Optional[Dict[str, Any]]): Additional event metadata
        token_usage (Optional[TokenUsage]): Token usage statistics if applicable
    """
    timestamp: str
    agent_name: str
    event_type: str
    inputs: List[Dict[str, str]]
    outputs: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None
    token_usage: Optional[TokenUsage] = None

class AgentTracer:
    """
    Tracks agent events and messages with comprehensive monitoring capabilities.
    
    This class provides functionality for:
    - Recording agent interactions and events
    - Tracking token usage and costs
    - Monitoring cache effectiveness
    - Collecting performance metrics
    - Generating analytics reports
    
    Attributes:
        events (List[AgentEvent]): List of recorded events
        logger (logging.Logger): Logger instance for event logging
        cache_stats (Dict[str, int]): Statistics about cache usage
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AgentTracer with configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing:
                - logging: Logging configuration with:
                    - level: Logging level (e.g., 'INFO', 'DEBUG')
                    - format: Log message format
                    - file: Log file path
                    - console: Whether to log to console
        """
        self.events: List[AgentEvent] = []
        self.logger = logging.getLogger("AgentTracer")
        self._configure_logging(config.get("logging", {}))
        self.cache_stats = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "cached_prompt_tokens": 0,
            "cached_completion_tokens": 0,
            "total_savings": 0
        }

    def _configure_logging(self, logging_config: Dict[str, Any]) -> None:
        """
        Configure logging based on the configuration.
        
        Args:
            logging_config (Dict[str, Any]): Logging configuration dictionary
        """
        level = getattr(logging, logging_config.get("level", "INFO"))
        self.logger.setLevel(level)
        
        formatter = logging.Formatter(logging_config.get("format", 
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        
        if logging_config.get("file"):
            file_handler = logging.FileHandler(logging_config["file"])
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        if logging_config.get("console", True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def on_messages_invoke(self, 
                          agent_name: str, 
                          messages: List[Dict[str, str]], 
                          token_usage: Optional[TokenUsage] = None,
                          cache_hit: bool = False) -> None:
        """
        Record when an agent starts processing messages.
        
        Args:
            agent_name (str): Name of the agent
            messages (List[Dict[str, str]]): List of messages being processed
            token_usage (Optional[TokenUsage]): Token usage statistics if available
            cache_hit (bool): Whether this was a cache hit
        """
        event = AgentEvent(
            timestamp=datetime.utcnow().isoformat(),
            agent_name=agent_name,
            event_type="invoke",
            inputs=messages,
            outputs=[],
            metadata={
                "start_time": time.time(),
                "cache_hit": cache_hit
            },
            token_usage=token_usage
        )
        self.events.append(event)
        self.logger.info(f"Agent {agent_name} started processing {len(messages)} messages")

    def on_messages_complete(self,
                           agent_name: str,
                           outputs: List[Dict[str, Any]],
                           token_usage: Optional[TokenUsage] = None,
                           cache_hit: bool = False,
                           cache_key: Optional[str] = None) -> None:
        """
        Record when an agent completes processing messages.
        
        Args:
            agent_name (str): Name of the agent
            outputs (List[Dict[str, Any]]): Output messages from processing
            token_usage (Optional[TokenUsage]): Token usage statistics if available
            cache_hit (bool): Whether this was a cache hit
            cache_key (Optional[str]): Cache key if this was a cache hit
        """
        event = AgentEvent(
            timestamp=datetime.utcnow().isoformat(),
            agent_name=agent_name,
            event_type="complete",
            inputs=[],
            outputs=outputs,
            metadata={
                "end_time": time.time(),
                "cache_hit": cache_hit,
                "cache_key": cache_key
            },
            token_usage=token_usage
        )
        self.events.append(event)
        
        if token_usage:
            self._update_cache_stats(token_usage, cache_hit)
            
        cache_status = "CACHE HIT" if cache_hit else "CACHE MISS"
        self.logger.info(
            f"Agent {agent_name} completed processing with {cache_status}\n"
            f"Token usage:\n"
            f"  Prompt tokens: {token_usage.prompt_tokens if token_usage else 'N/A'}\n"
            f"  Completion tokens: {token_usage.completion_tokens if token_usage else 'N/A'}\n"
            f"  Total tokens: {token_usage.total_tokens if token_usage else 'N/A'}\n"
            f"  Cache savings: {self.cache_stats['total_savings']} tokens"
        )

    def _update_cache_stats(self, token_usage: TokenUsage, cache_hit: bool) -> None:
        """
        Update cache statistics based on token usage.
        
        Args:
            token_usage (TokenUsage): Token usage statistics
            cache_hit (bool): Whether this was a cache hit
        """
        self.cache_stats["total_prompt_tokens"] += token_usage.prompt_tokens
        self.cache_stats["total_completion_tokens"] += token_usage.completion_tokens
        
        if cache_hit:
            self.cache_stats["cached_prompt_tokens"] += token_usage.prompt_tokens
            self.cache_stats["cached_completion_tokens"] += token_usage.completion_tokens
            self.cache_stats["total_savings"] += token_usage.total_tokens

    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get detailed cache statistics.
        
        Returns:
            Dict[str, Any]: Cache statistics including:
                - total_prompt_tokens: Total prompt tokens used
                - total_completion_tokens: Total completion tokens used
                - cached_prompt_tokens: Prompt tokens served from cache
                - cached_completion_tokens: Completion tokens served from cache
                - total_savings: Total tokens saved by cache
                - savings_percentage: Percentage of tokens saved
        """
        total_tokens = self.cache_stats["total_prompt_tokens"] + self.cache_stats["total_completion_tokens"]
        savings_percentage = (
            (self.cache_stats["total_savings"] / total_tokens * 100)
            if total_tokens > 0 else 0
        )
        
        return {
            **self.cache_stats,
            "savings_percentage": round(savings_percentage, 2)
        }

    def save_trace(self, filepath: str) -> None:
        """
        Save the complete trace to a JSON file.
        
        Args:
            filepath (str): Path where to save the trace file
        """
        trace_data = {
            "events": [
                {
                    "timestamp": event.timestamp,
                    "agent_name": event.agent_name,
                    "event_type": event.event_type,
                    "inputs": event.inputs,
                    "outputs": event.outputs,
                    "metadata": event.metadata,
                    "token_usage": {
                        "prompt_tokens": event.token_usage.prompt_tokens,
                        "completion_tokens": event.token_usage.completion_tokens,
                        "total_tokens": event.token_usage.total_tokens,
                        "model": event.token_usage.model,
                        "cache_hit": event.token_usage.cache_hit,
                        "cache_key": event.token_usage.cache_key,
                        "cache_savings": event.token_usage.cache_savings
                    } if event.token_usage else None
                }
                for event in self.events
            ],
            "cache_statistics": self.get_cache_statistics()
        }
        
        with open(filepath, "w") as f:
            json.dump(trace_data, f, indent=2)
        
        self.logger.info(f"Trace saved to {filepath}")

    def get_trace(self) -> List[Dict[str, Any]]:
        """
        Get the complete trace as a list of event dictionaries.
        
        Returns:
            List[Dict[str, Any]]: List of event dictionaries containing:
                - timestamp: Event timestamp
                - agent_name: Name of the agent
                - event_type: Type of event
                - inputs: Input messages
                - outputs: Output messages
                - metadata: Event metadata
                - token_usage: Token usage statistics
        """
        return [
            {
                "timestamp": event.timestamp,
                "agent_name": event.agent_name,
                "event_type": event.event_type,
                "inputs": event.inputs,
                "outputs": event.outputs,
                "metadata": event.metadata,
                "token_usage": {
                    "prompt_tokens": event.token_usage.prompt_tokens,
                    "completion_tokens": event.token_usage.completion_tokens,
                    "total_tokens": event.token_usage.total_tokens,
                    "model": event.token_usage.model,
                    "cache_hit": event.token_usage.cache_hit,
                    "cache_key": event.token_usage.cache_key,
                    "cache_savings": event.token_usage.cache_savings
                } if event.token_usage else None
            }
            for event in self.events
        ] 