"""
Base Agent System

This module provides shared functionality for agent-based systems, including:
- Common agent creation and configuration
- Standard logging and event tracking
- Shared utility functions
- Base agent classes
"""

import json
import logging
import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.messages import TextMessage, BaseChatMessage
from autogen_agentchat.base import Response
from llm_cache import LLMCache
from tool_analytics import ToolAnalytics

# Global logs and cache
ROOT_CAUSE_DATA: List[dict] = []  # Stores detailed event data for root cause analysis
FILE_LOG: List[str] = []         # Tracks file operations
ACTION_LOG: List[str] = []       # Records agent actions and decisions

# Initialize LLM cache
llm_cache = LLMCache(
    max_size=1000,
    similarity_threshold=0.85,
    expiration_hours=24
)

def setup_logging(config: dict) -> None:
    """Configure logging based on configuration settings."""
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

def log_event(logger: logging.Logger, agent_name: str, event_type: str, inputs: List[BaseChatMessage], outputs) -> None:
    """Log an event in the system with detailed information."""
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "agent": agent_name,
        "event": event_type,
        "inputs": [{"source": m.source, "content": m.content} for m in inputs],
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

def create_base_agents(config: Dict, logger: logging.Logger) -> Dict[str, Any]:
    """Create and configure the base agents using settings from config file."""
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

def create_group_chat(agents: List[Any], config: Dict, logger: logging.Logger) -> GroupChatManager:
    """Create and configure the group chat system."""
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

def load_json_file(file_path: str) -> Dict:
    """Load and parse a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Error loading JSON file {file_path}: {str(e)}")

def save_json_file(data: Dict, file_path: str) -> None:
    """Save data to a JSON file with proper formatting."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise ValueError(f"Error saving JSON file {file_path}: {str(e)}") 