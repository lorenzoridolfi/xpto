#!/usr/bin/env python3

import json
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class TokenUsage:
    """Represents token usage statistics for an LLM call."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    cache_hit: bool = False
    cache_key: Optional[str] = None
    cache_savings: Optional[Dict[str, int]] = (
        None  # Track savings for both prompt and completion
    )


@dataclass
class AgentEvent:
    """Represents an event in the agent's execution."""

    timestamp: str
    agent_name: str
    event_type: str
    inputs: List[Dict[str, str]]
    outputs: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None
    token_usage: Optional[TokenUsage] = None


class AgentTracer:
    """Tracks agent events and messages."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AgentTracer with configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary containing:
                - logging: Logging configuration
        """
        self.config = config
        self.logger = logging.getLogger("agent_tracer")
        self._setup_logging()
        self.events: List[AgentEvent] = []
        self.cache_stats = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "cached_prompt_tokens": 0,
            "cached_completion_tokens": 0,
            "total_savings": 0,
        }

    def _setup_logging(self) -> None:
        """Configure logging based on the configuration."""
        log_config = self.config.get("logging", {})
        self.logger.setLevel(getattr(logging, log_config.get("level", "INFO")))

        # Add file handler if specified
        if log_config.get("file"):
            file_handler = logging.FileHandler(log_config["file"])
            file_handler.setFormatter(logging.Formatter(log_config.get("format")))
            self.logger.addHandler(file_handler)

        # Add console handler if enabled
        if log_config.get("console", True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(log_config.get("format")))
            self.logger.addHandler(console_handler)

    def on_messages_invoke(
        self,
        agent_name: str,
        messages: List[Dict[str, str]],
        token_usage: Optional[TokenUsage] = None,
        cache_hit: bool = False,
        cache_key: Optional[str] = None,
    ) -> None:
        """
        Record when an agent starts processing messages.

        Args:
            agent_name (str): Name of the agent
            messages (List[Dict[str, str]]): List of messages
            token_usage (Optional[TokenUsage]): Token usage statistics if available
            cache_hit (bool): Whether this was a cache hit
            cache_key (Optional[str]): Cache key used for lookup
        """
        if token_usage:
            token_usage.cache_hit = cache_hit
            token_usage.cache_key = cache_key

            # Update cache statistics
            if cache_hit:
                self.cache_stats["cached_prompt_tokens"] += token_usage.prompt_tokens
                self.cache_stats[
                    "cached_completion_tokens"
                ] += token_usage.completion_tokens
                self.cache_stats["total_savings"] += token_usage.total_tokens

            self.cache_stats["total_prompt_tokens"] += token_usage.prompt_tokens
            self.cache_stats["total_completion_tokens"] += token_usage.completion_tokens

        event = AgentEvent(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            agent_name=agent_name,
            event_type="invoke",
            inputs=messages,
            outputs=[],
            metadata={
                "start_time": time.time(),
                "cache_stats": self.cache_stats.copy() if token_usage else None,
            },
            token_usage=token_usage,
        )

        self.events.append(event)
        self.logger.info(
            f"Agent {agent_name} started processing {len(messages)} messages"
        )

        if token_usage:
            cache_status = "CACHE HIT" if cache_hit else "CACHE MISS"
            self.logger.info(
                f"Token usage ({cache_status}):\n"
                f"  Prompt tokens: {token_usage.prompt_tokens}\n"
                f"  Completion tokens: {token_usage.completion_tokens}\n"
                f"  Total tokens: {token_usage.total_tokens}\n"
                f"  Cache savings: {self.cache_stats['total_savings']} tokens"
            )

    def on_messages_complete(
        self,
        agent_name: str,
        outputs: List[Dict[str, Any]],
        token_usage: Optional[TokenUsage] = None,
        cache_hit: bool = False,
        cache_key: Optional[str] = None,
    ) -> None:
        """
        Record when an agent completes processing messages.

        Args:
            agent_name (str): Name of the agent
            outputs (List[Dict[str, Any]]): List of outputs
            token_usage (Optional[TokenUsage]): Token usage statistics if available
            cache_hit (bool): Whether this was a cache hit
            cache_key (Optional[str]): Cache key used for lookup
        """
        # Find the corresponding invoke event
        invoke_event = next(
            (
                e
                for e in reversed(self.events)
                if e.agent_name == agent_name and e.event_type == "invoke"
            ),
            None,
        )

        if invoke_event:
            processing_time = time.time() - invoke_event.metadata["start_time"]

            if token_usage:
                token_usage.cache_hit = cache_hit
                token_usage.cache_key = cache_key

                # Calculate cache savings for this completion
                if cache_hit:
                    self.cache_stats[
                        "cached_prompt_tokens"
                    ] += token_usage.prompt_tokens
                    self.cache_stats[
                        "cached_completion_tokens"
                    ] += token_usage.completion_tokens
                    self.cache_stats["total_savings"] += token_usage.total_tokens

                self.cache_stats["total_prompt_tokens"] += token_usage.prompt_tokens
                self.cache_stats[
                    "total_completion_tokens"
                ] += token_usage.completion_tokens

            event = AgentEvent(
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                agent_name=agent_name,
                event_type="complete",
                inputs=invoke_event.inputs,
                outputs=outputs,
                metadata={
                    "processing_time": processing_time,
                    "start_time": invoke_event.metadata["start_time"],
                    "end_time": time.time(),
                    "cache_stats": self.cache_stats.copy() if token_usage else None,
                },
                token_usage=token_usage,
            )

            self.events.append(event)
            self.logger.info(
                f"Agent {agent_name} completed processing in {processing_time:.2f}s"
            )

            if token_usage:
                cache_status = "CACHE HIT" if cache_hit else "CACHE MISS"
                self.logger.info(
                    f"Token usage ({cache_status}):\n"
                    f"  Prompt tokens: {token_usage.prompt_tokens}\n"
                    f"  Completion tokens: {token_usage.completion_tokens}\n"
                    f"  Total tokens: {token_usage.total_tokens}\n"
                    f"  Cache savings: {self.cache_stats['total_savings']} tokens"
                )

    def get_trace(self) -> List[Dict[str, Any]]:
        """
        Get the complete trace of events.

        Returns:
            List[Dict[str, Any]]: List of events
        """
        return [asdict(event) for event in self.events]

    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get current cache statistics.

        Returns:
            Dict[str, Any]: Cache statistics including:
                - total_prompt_tokens: Total prompt tokens used
                - total_completion_tokens: Total completion tokens used
                - cached_prompt_tokens: Prompt tokens served from cache
                - cached_completion_tokens: Completion tokens served from cache
                - total_savings: Total tokens saved by cache
                - savings_percentage: Percentage of tokens saved
        """
        total_tokens = (
            self.cache_stats["total_prompt_tokens"]
            + self.cache_stats["total_completion_tokens"]
        )
        savings_percentage = (
            (self.cache_stats["total_savings"] / total_tokens * 100)
            if total_tokens > 0
            else 0
        )

        return {**self.cache_stats, "savings_percentage": round(savings_percentage, 2)}

    def save_trace(self, file_path: str) -> None:
        """
        Save the trace to a file.

        Args:
            file_path (str): Path to save the trace
        """
        try:
            trace_data = {
                "events": self.get_trace(),
                "cache_statistics": self.get_cache_statistics(),
            }
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(trace_data, f, indent=2)
            self.logger.info(f"Trace saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving trace to {file_path}: {e}")
            raise

    def clear_trace(self) -> None:
        """Clear the current trace."""
        self.events = []
        self.logger.info("Trace cleared")
