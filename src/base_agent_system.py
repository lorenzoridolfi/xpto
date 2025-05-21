"""
Base Agent System

This module provides shared functionality for agent-based systems, including:
- Common agent creation and configuration
- Standard logging and event tracking
- Shared utility functions
- Base agent classes

The module implements a robust foundation for agent-based systems with:
- Comprehensive error handling
- Type-safe interfaces
- Structured logging
- Event tracking for analysis
"""

import json
import logging
import datetime
from typing import Dict, List, Any, Optional, Union, TypedDict, Tuple
from dataclasses import dataclass
from pathlib import Path

from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.messages import TextMessage, BaseChatMessage
from autogen_agentchat.base import Response
from .llm_cache import LLMCache
from .tool_analytics import ToolAnalytics

# Custom Exceptions
class SystemError(Exception):
    """Base exception for system-related errors."""
    pass

class ConfigError(SystemError):
    """Raised when configuration operations fail."""
    pass

class FileOperationError(SystemError):
    """Raised when file operations fail."""
    pass

class AgentError(SystemError):
    """Raised when agent operations fail."""
    pass

# Type Definitions
class ConfigDict(TypedDict):
    name: str
    logging: Dict[str, Any]
    system: Dict[str, Any]
    agents: Dict[str, Any]
    llm_config: Dict[str, Any]

class EventEntry(TypedDict):
    timestamp: str
    agent: str
    event: str
    inputs: List[Dict[str, str]]
    outputs: List[Dict[str, Any]]

# Global logs and cache
ROOT_CAUSE_DATA: List[EventEntry] = []  # Stores detailed event data for root cause analysis
FILE_LOG: List[str] = []         # Tracks file operations
ACTION_LOG: List[str] = []       # Records agent actions and decisions

# Initialize LLM cache
llm_cache = LLMCache(
    max_size=1000,
    expiration_hours=24
)

def setup_logging(config: ConfigDict) -> logging.Logger:
    """
    Configure logging based on configuration settings.

    Args:
        config (ConfigDict): Configuration dictionary containing logging settings

    Returns:
        logging.Logger: Configured logger instance

    Raises:
        ConfigError: If logging configuration fails
    """
    try:
        log_config = config["logging"]
        
        # Set up logger
        logger = logging.getLogger(config.get("name", "agent_system"))
        logger.setLevel(getattr(logging, log_config["level"]))
        
        # Create formatter
        formatter = logging.Formatter(log_config["format"])
        
        # Console handler
        if log_config.get("console", True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(log_config["file"])
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Don't propagate to root logger
        logger.propagate = False
        
        return logger
    except Exception as e:
        raise ConfigError(f"Failed to configure logging: {str(e)}")

def log_event(logger: logging.Logger, agent_name: str, event_type: str, inputs: List[BaseChatMessage], outputs: Any) -> None:
    """
    Log an event in the system with detailed information.

    Args:
        logger (logging.Logger): Logger instance to use
        agent_name (str): Name of the agent generating the event
        event_type (str): Type of event (e.g., 'on_messages_invoke', 'on_messages_complete')
        inputs (List[BaseChatMessage]): Input messages to the agent
        outputs (Any): Output from the agent (can be Response, list of Responses, or other types)

    Raises:
        SystemError: If logging fails
    """
    try:
        entry: EventEntry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "agent": agent_name,
            "event": event_type,
            "inputs": [{"source": m.source, "content": m.content} for m in inputs],
            "outputs": []
        }
        
        if isinstance(outputs, Response):
            cm = outputs.chat_message
            try:
                content = json.loads(cm.content)
                entry["outputs"] = [{"source": cm.source, "content": content}]
            except json.JSONDecodeError:
                entry["outputs"] = [{"source": cm.source, "content": cm.content}]
        elif isinstance(outputs, list) and all(isinstance(o, Response) for o in outputs):
            entry["outputs"] = []
            for o in outputs:
                try:
                    content = json.loads(o.chat_message.content)
                    entry["outputs"].append({"source": o.chat_message.source, "content": content})
                except json.JSONDecodeError:
                    entry["outputs"].append({"source": o.chat_message.source, "content": o.chat_message.content})
        else:
            entry["outputs"] = outputs
        
        logger.debug(f"Event: {json.dumps(entry, indent=2)}")
        ROOT_CAUSE_DATA.append(entry)
    except Exception as e:
        raise SystemError(f"Failed to log event: {str(e)}")

def create_base_agents(config: ConfigDict, logger: logging.Logger) -> Dict[str, Union[UserProxyAgent, AssistantAgent]]:
    """
    Create and configure the base agents using settings from config file.

    Args:
        config (ConfigDict): Configuration dictionary containing agent settings
        logger (logging.Logger): Logger instance to use

    Returns:
        Dict[str, Union[UserProxyAgent, AssistantAgent]]: Dictionary containing created agents

    Raises:
        ConfigError: If agent creation fails
    """
    try:
        logger.debug("Creating base agents...")
        
        # Create User Proxy
        user_proxy_config = config["system"]["user_proxy"]
        user_proxy = UserProxyAgent(
            name=user_proxy_config["name"],
            human_input_mode=user_proxy_config["human_input_mode"],
            max_consecutive_auto_reply=user_proxy_config["max_consecutive_auto_reply"]
        )
        logger.debug("User Proxy Agent created")
        
        # Create Supervisor Agent
        supervisor_config = config["agents"]["SupervisorAgent"]
        supervisor = AssistantAgent(
            name=supervisor_config["name"],
            system_message=supervisor_config["system_message"],
            llm_config=config["llm_config"]["supervisor"]
        )
        logger.debug(f"Supervisor Agent created: {supervisor_config['description']}")
        
        return {
            "user_proxy": user_proxy,
            "supervisor": supervisor
        }
    except Exception as e:
        raise ConfigError(f"Failed to create base agents: {str(e)}")

def create_group_chat(agents: List[Any], config: ConfigDict, logger: logging.Logger) -> GroupChatManager:
    """
    Create and configure the group chat system.

    Args:
        agents (List[Any]): List of agents to include in the group chat
        config (ConfigDict): Configuration dictionary containing group chat settings
        logger (logging.Logger): Logger instance to use

    Returns:
        GroupChatManager: Configured group chat manager

    Raises:
        ConfigError: If group chat creation fails
    """
    try:
        logger.debug("Creating group chat...")
        
        # Create Group Chat
        groupchat = GroupChat(
            agents=agents,
            messages=[],
            max_round=config["system"]["group_chat"]["max_round"]
        )
        logger.debug("Group Chat created")
        
        # Create Group Chat Manager
        group_chat_manager = GroupChatManager(
            groupchat=groupchat,
            llm_config=config["llm_config"]["supervisor"]
        )
        logger.debug("Group Chat Manager created")
        
        return group_chat_manager
    except Exception as e:
        raise ConfigError(f"Failed to create group chat: {str(e)}")

def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load and parse a JSON file.

    Args:
        file_path (str): Path to the JSON file to load

    Returns:
        Dict[str, Any]: Parsed JSON data

    Raises:
        FileOperationError: If file loading fails
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise FileOperationError(f"Error loading JSON file {file_path}: {str(e)}")

def save_json_file(data: Dict[str, Any], file_path: str) -> None:
    """
    Save data to a JSON file with proper formatting.

    Args:
        data (Dict[str, Any]): Data to save
        file_path (str): Path where to save the JSON file

    Raises:
        FileOperationError: If file saving fails
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise FileOperationError(f"Error saving JSON file {file_path}: {str(e)}") 