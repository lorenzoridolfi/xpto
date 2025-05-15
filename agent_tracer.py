#!/usr/bin/env python3

import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseChatMessage

@dataclass
class AgentEvent:
    """Represents a single event in the agent's execution."""
    timestamp: str
    agent_name: str
    event_type: str
    inputs: List[Dict[str, str]]
    outputs: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None

class AgentTracer:
    """Handles tracing of agent events and messages for root cause analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AgentTracer with configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing:
                - logging: Logging configuration
                - cache: Cache configuration
                - analytics: Analytics configuration
        """
        self.config = config
        self.events: List[AgentEvent] = []
        self.logger = logging.getLogger("agent_tracer")
        
        # Setup logging
        self._setup_logging()
        
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
    
    def on_messages_invoke(self, agent_name: str, inputs: List[BaseChatMessage]) -> None:
        """
        Called when an agent starts processing messages.
        
        Args:
            agent_name (str): Name of the agent
            inputs (List[BaseChatMessage]): Input messages
        """
        self.logger.debug(f"Agent {agent_name} starting to process messages")
        # Store initial state for later use in on_messages_complete
        self._current_event = {
            "agent_name": agent_name,
            "inputs": inputs,
            "start_time": datetime.datetime.utcnow()
        }
    
    def on_messages_complete(self, agent_name: str, outputs: Union[Response, List[Response], List[BaseChatMessage]]) -> None:
        """
        Called when an agent finishes processing messages.
        
        Args:
            agent_name (str): Name of the agent
            outputs: Output from the agent (can be Response, list of Responses, or list of BaseChatMessage)
        """
        if not hasattr(self, '_current_event'):
            self.logger.warning(f"No matching on_messages_invoke found for {agent_name}")
            return
            
        # Create event entry
        event = AgentEvent(
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            agent_name=agent_name,
            event_type="message_complete",
            inputs=[{"source": m.source, "content": m.content} for m in self._current_event["inputs"]],
            outputs=self._process_outputs(outputs),
            metadata={
                "processing_time": (datetime.datetime.utcnow() - self._current_event["start_time"]).total_seconds()
            }
        )
        
        # Store event
        self.events.append(event)
        
        # Log event
        self.logger.debug(f"Agent {agent_name} completed processing: {json.dumps(asdict(event), indent=2)}")
        
        # Cleanup
        delattr(self, '_current_event')
    
    def _process_outputs(self, outputs: Union[Response, List[Response], List[BaseChatMessage]]) -> List[Dict[str, Any]]:
        """
        Process agent outputs into a standardized format.
        
        Args:
            outputs: Output from the agent
            
        Returns:
            List[Dict[str, Any]]: Processed outputs in standardized format
        """
        processed_outputs = []
        
        if isinstance(outputs, Response):
            cm = outputs.chat_message
            try:
                content = json.loads(cm.content)
                processed_outputs.append({
                    "source": cm.source,
                    "content": content,
                    "type": "json"
                })
            except json.JSONDecodeError:
                processed_outputs.append({
                    "source": cm.source,
                    "content": cm.content,
                    "type": "text"
                })
                
        elif isinstance(outputs, list):
            for output in outputs:
                if isinstance(output, Response):
                    cm = output.chat_message
                    try:
                        content = json.loads(cm.content)
                        processed_outputs.append({
                            "source": cm.source,
                            "content": content,
                            "type": "json"
                        })
                    except json.JSONDecodeError:
                        processed_outputs.append({
                            "source": cm.source,
                            "content": cm.content,
                            "type": "text"
                        })
                elif isinstance(output, BaseChatMessage):
                    try:
                        content = json.loads(output.content)
                        processed_outputs.append({
                            "source": output.source,
                            "content": content,
                            "type": "json"
                        })
                    except json.JSONDecodeError:
                        processed_outputs.append({
                            "source": output.source,
                            "content": output.content,
                            "type": "text"
                        })
        
        return processed_outputs
    
    def get_trace(self) -> List[Dict[str, Any]]:
        """
        Get the complete trace of agent events.
        
        Returns:
            List[Dict[str, Any]]: List of agent events
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