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
    
    def on_messages_invoke(self, agent_name: str, messages: List[Dict[str, str]], 
                          token_usage: Optional[TokenUsage] = None) -> None:
        """
        Record when an agent starts processing messages.
        
        Args:
            agent_name (str): Name of the agent
            messages (List[Dict[str, str]]): List of messages
            token_usage (Optional[TokenUsage]): Token usage statistics if available
        """
        event = AgentEvent(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            agent_name=agent_name,
            event_type="invoke",
            inputs=messages,
            outputs=[],
            metadata={"start_time": time.time()},
            token_usage=token_usage
        )
        
        self.events.append(event)
        self.logger.info(f"Agent {agent_name} started processing {len(messages)} messages")
        
        if token_usage:
            self.logger.info(f"Token usage: {token_usage.prompt_tokens} prompt, "
                           f"{token_usage.completion_tokens} completion, "
                           f"{token_usage.total_tokens} total")
    
    def on_messages_complete(self, agent_name: str, outputs: List[Dict[str, Any]],
                           token_usage: Optional[TokenUsage] = None) -> None:
        """
        Record when an agent completes processing messages.
        
        Args:
            agent_name (str): Name of the agent
            outputs (List[Dict[str, Any]]): List of outputs
            token_usage (Optional[TokenUsage]): Token usage statistics if available
        """
        # Find the corresponding invoke event
        invoke_event = next((e for e in reversed(self.events) 
                           if e.agent_name == agent_name and e.event_type == "invoke"), None)
        
        if invoke_event:
            processing_time = time.time() - invoke_event.metadata["start_time"]
            
            event = AgentEvent(
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                agent_name=agent_name,
                event_type="complete",
                inputs=invoke_event.inputs,
                outputs=outputs,
                metadata={
                    "processing_time": processing_time,
                    "start_time": invoke_event.metadata["start_time"],
                    "end_time": time.time()
                },
                token_usage=token_usage
            )
            
            self.events.append(event)
            self.logger.info(f"Agent {agent_name} completed processing in {processing_time:.2f}s")
            
            if token_usage:
                self.logger.info(f"Token usage: {token_usage.prompt_tokens} prompt, "
                               f"{token_usage.completion_tokens} completion, "
                               f"{token_usage.total_tokens} total")
    
    def get_trace(self) -> List[Dict[str, Any]]:
        """
        Get the complete trace of events.
        
        Returns:
            List[Dict[str, Any]]: List of events
        """
        return [asdict(event) for event in self.events]
    
    def save_trace(self, file_path: str) -> None:
        """
        Save the trace to a file.
        
        Args:
            file_path (str): Path to save the trace
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.get_trace(), f, indent=2)
            self.logger.info(f"Trace saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving trace to {file_path}: {e}")
            raise
    
    def clear_trace(self) -> None:
        """Clear the current trace."""
        self.events = []
        self.logger.info("Trace cleared") 