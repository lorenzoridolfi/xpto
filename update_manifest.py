#!/usr/bin/env python3

"""
Update Manifest Program

This program demonstrates a basic implementation of the enhanced agent architecture,
focusing on manifest file management and state handling.

Architecture Components Used:
1. AdaptiveAgent
   - Implements manifest analysis
   - Handles update coordination
   - Manages manifest validation

2. StateManager
   - Implements state persistence
   - Manages state validation
   - Handles state synchronization

3. CollaborativeWorkflow
   - Orchestrates update processes
   - Manages agent coordination
   - Tracks update progress

Key Features:
- Manifest analysis and updates
- State management and validation
- Workflow orchestration

Example Usage:
    async def main():
        # Initialize workflow
        workflow = create_workflow()
        await workflow.initialize()
        
        # Process manifest update
        update_requirements = {
            "type": "dependency_update",
            "target_dependencies": ["package1", "package2"],
            "version_constraints": {
                "package1": ">=2.0.0",
                "package2": ">=1.5.0"
            }
        }
        
        result = await process_manifest_update(
            manifest_path="path/to/manifest.json",
            update_requirements=update_requirements
        )
"""

import json
import logging
from typing import Dict, List, Any, Optional, Set, Union, TypedDict
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from autogen import AssistantAgent, UserProxyAgent
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.messages import TextMessage, BaseChatMessage
from autogen_agentchat.base import Response

from src.base_agent_system import (
    setup_logging, log_event, create_base_agents, create_group_chat,
    load_json_file, save_json_file, FILE_LOG, ROOT_CAUSE_DATA
)
from src.llm_cache import LLMCache
from src.tool_analytics import ToolAnalytics
from src.analytics_assistant_agent import AnalyticsAssistantAgent
from enhanced_agent import AdaptiveAgent
from collaborative_workflow import CollaborativeWorkflow
from state_manager import StateManager

# Custom Exceptions
class ManifestError(Exception):
    """Base exception for manifest-related errors."""
    pass

class ManifestValidationError(ManifestError):
    """Raised when manifest validation fails."""
    pass

class ConfigError(Exception):
    """Base exception for configuration-related errors."""
    pass

class FileOperationError(Exception):
    """Raised when file operations fail."""
    pass

# Type Definitions
class UpdateRequirements(TypedDict):
    type: str
    target_dependencies: List[str]
    version_constraints: Dict[str, str]

class ConfigDict(TypedDict):
    cache_config: Dict[str, Union[int, float]]
    logging: Dict[str, Any]

# Global logger instance
logger = logging.getLogger("update_manifest")

def load_config() -> ConfigDict:
    """
    Load configuration from the new config directory structure.
    
    Returns:
        ConfigDict: Configuration dictionary
        
    Raises:
        ConfigError: If configuration files cannot be loaded or are invalid
    """
    config: Dict[str, Any] = {}
    
    # Load shared configurations
    shared_configs = [
        'config/shared/global_settings.json',
        'config/shared/base_config.json',
        'config/shared/agent_settings.json',
        'config/shared/logging_settings.json'
    ]
    
    for config_file in shared_configs:
        try:
            with open(config_file) as f:
                config.update(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ConfigError(f"Failed to load config file {config_file}: {str(e)}")
    
    # Load program-specific configuration
    try:
        with open('config/update_manifest/program_config.json') as f:
            config.update(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ConfigError(f"Failed to load program config: {str(e)}")
    
    return config

def load_manifest_schema() -> Dict[str, Any]:
    """
    Load manifest validation schema from the new config directory structure.
    
    Returns:
        Dict[str, Any]: Manifest validation schema
        
    Raises:
        FileOperationError: If schema file cannot be loaded or is invalid
    """
    try:
        with open('config/update_manifest/manifest_validation_schema.json') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise FileOperationError(f"Failed to load manifest schema: {str(e)}")

# Initialize LLM cache
config = load_config()
llm_cache = LLMCache(
    max_size=config["cache_config"]["max_size"],
    similarity_threshold=config["cache_config"]["similarity_threshold"],
    expiration_hours=config["cache_config"]["expiration_hours"]
)

class FileReaderAgent(BaseChatAgent):
    """
    Agent responsible for reading and processing manifest files.
    
    This agent maintains a list of files it has read and can process multiple files
    in a single request. It handles file reading errors gracefully and logs all
    file operations.
    
    Attributes:
        manifest (Set[str]): Set of filenames the agent can read
        files_read (List[str]): List of files that have been read
        file_log (List[str]): List to store file operation logs
    """
    
    def __init__(self, name: str, description: str, manifest: List[Dict[str, str]], file_log: List[str]):
        """
        Initialize the FileReaderAgent.

        Args:
            name (str): Name of the agent
            description (str): Description of the agent's role
            manifest (List[Dict[str, str]]): List of files the agent can read
            file_log (List[str]): List to store file operation logs
            
        Raises:
            ValueError: If name or description is empty
        """
        if not name or not description:
            raise ValueError("Name and description must not be empty")
            
        super().__init__(name, description=description)
        self._name = name
        self._description = description
        self.manifest: Set[str] = {f["filename"] for f in manifest}
        self.files_read: List[str] = []
        self.file_log: List[str] = file_log

    @property
    def name(self) -> str:
        """Return the agent's name."""
        return self._name

    @property
    def description(self) -> str:
        """Return the agent's description."""
        return self._description

    @property
    def produced_message_types(self) -> tuple:
        """Return the types of messages this agent can produce."""
        return (TextMessage,)

    @property
    def consumed_message_types(self) -> tuple:
        """Return the types of messages this agent can consume."""
        return (TextMessage,)

    async def run(self, task: str) -> str:
        """
        Run a task with logging.

        Args:
            task (str): The task to run, expected to be a comma-separated list of filenames

        Returns:
            str: The combined content of requested files or "NO_FILE" if no valid files requested
            
        Raises:
            FileOperationError: If file reading fails
        """
        log_event(self.name, "run_invoke", [], [])
        
        # Process the task as a file request
        requested = [fn.strip() for fn in task.split(",") if fn.strip()]
        valid = [fn for fn in requested if fn in self.manifest and fn not in self.files_read]
        
        if not valid:
            result = "NO_FILE"
        else:
            combined = []
            for fname in valid:
                try:
                    text = open(fname, encoding="utf-8").read()
                except Exception as e:
                    raise FileOperationError(f"Error reading {fname}: {str(e)}")
                self.files_read.append(fname)
                self.file_log.append(f"{self.name}: read {fname}")
                combined.append(f"--- {fname} ---\n{text}")
            result = "\n\n".join(combined)
        
        log_event(self.name, "run_complete", [], result)
        return result

    async def on_messages(self, messages: List[BaseChatMessage], cancellation_token) -> Response:
        """
        Process file reading requests.

        Args:
            messages (List[BaseChatMessage]): List of messages containing file requests
            cancellation_token: Token for cancellation support

        Returns:
            Response: Response containing the file contents or error message
            
        Raises:
            FileOperationError: If file reading fails
        """
        try:
            result = await self.run(messages[0].content)
            return Response(chat_message=TextMessage(content=result, source=self.name))
        except FileOperationError as e:
            logger.error(f"File operation error: {str(e)}")
            return Response(chat_message=TextMessage(content=f"Error: {str(e)}", source=self.name))
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return Response(chat_message=TextMessage(content=f"Unexpected error: {str(e)}", source=self.name))

class ManifestUpdaterAgent(AnalyticsAssistantAgent):
    """
    Agent responsible for updating manifest files based on requirements.
    
    This agent handles the process of updating manifest files according to specified
    requirements, including dependency updates and version constraints.
    
    Attributes:
        llm_config (Dict[str, Any]): Configuration for the LLM model
        analytics (ToolAnalytics): Analytics tracking for tool usage
    """
    
    def __init__(self, name: str, description: str, llm_config: Dict[str, Any]):
        """
        Initialize the ManifestUpdaterAgent.

        Args:
            name (str): Name of the agent
            description (str): Description of the agent's role
            llm_config (Dict[str, Any]): Configuration for the LLM model
            
        Raises:
            ValueError: If name, description, or llm_config is invalid
        """
        if not name or not description:
            raise ValueError("Name and description must not be empty")
        if not llm_config:
            raise ValueError("LLM configuration must be provided")
            
        super().__init__(name, description=description)
        self.llm_config = llm_config
        self.analytics = ToolAnalytics()

    async def update_manifest(self, content: str) -> Dict[str, Any]:
        """
        Update manifest content based on requirements.

        Args:
            content (str): Current manifest content

        Returns:
            Dict[str, Any]: Updated manifest content
            
        Raises:
            ManifestError: If manifest update fails
        """
        try:
            # Process manifest update logic here
            # This is a placeholder for the actual implementation
            return {"status": "success", "content": content}
        except Exception as e:
            raise ManifestError(f"Failed to update manifest: {str(e)}")

    async def on_messages(self, messages: List[BaseChatMessage], cancellation_token) -> Response:
        """
        Process manifest update requests.

        Args:
            messages (List[BaseChatMessage]): List of messages containing update requests
            cancellation_token: Token for cancellation support

        Returns:
            Response: Response containing the update result
            
        Raises:
            ManifestError: If manifest update fails
        """
        try:
            result = await self.update_manifest(messages[0].content)
            return Response(chat_message=TextMessage(content=json.dumps(result), source=self.name))
        except ManifestError as e:
            logger.error(f"Manifest update error: {str(e)}")
            return Response(chat_message=TextMessage(content=f"Error: {str(e)}", source=self.name))
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return Response(chat_message=TextMessage(content=f"Unexpected error: {str(e)}", source=self.name))

class LoggingConfigAgent(AnalyticsAssistantAgent):
    """
    Agent responsible for configuring logging settings.
    
    This agent handles the configuration of logging parameters and ensures
    proper logging setup throughout the system.
    
    Attributes:
        llm_config (Dict[str, Any]): Configuration for the LLM model
        analytics (ToolAnalytics): Analytics tracking for tool usage
    """
    
    def __init__(self, name: str, description: str, llm_config: Dict[str, Any]):
        """
        Initialize the LoggingConfigAgent.

        Args:
            name (str): Name of the agent
            description (str): Description of the agent's role
            llm_config (Dict[str, Any]): Configuration for the LLM model
            
        Raises:
            ValueError: If name, description, or llm_config is invalid
        """
        if not name or not description:
            raise ValueError("Name and description must not be empty")
        if not llm_config:
            raise ValueError("LLM configuration must be provided")
            
        super().__init__(name, description=description)
        self.llm_config = llm_config
        self.analytics = ToolAnalytics()

    async def configure_logging(self, content: str) -> Dict[str, Any]:
        """
        Configure logging settings based on provided content.

        Args:
            content (str): Logging configuration content

        Returns:
            Dict[str, Any]: Updated logging configuration
            
        Raises:
            ConfigError: If logging configuration fails
        """
        try:
            # Process logging configuration logic here
            # This is a placeholder for the actual implementation
            return {"status": "success", "config": content}
        except Exception as e:
            raise ConfigError(f"Failed to configure logging: {str(e)}")

    async def on_messages(self, messages: List[BaseChatMessage], cancellation_token) -> Response:
        """
        Process logging configuration requests.

        Args:
            messages (List[BaseChatMessage]): List of messages containing configuration requests
            cancellation_token: Token for cancellation support

        Returns:
            Response: Response containing the configuration result
            
        Raises:
            ConfigError: If logging configuration fails
        """
        try:
            result = await self.configure_logging(messages[0].content)
            return Response(chat_message=TextMessage(content=json.dumps(result), source=self.name))
        except ConfigError as e:
            logger.error(f"Logging configuration error: {str(e)}")
            return Response(chat_message=TextMessage(content=f"Error: {str(e)}", source=self.name))
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return Response(chat_message=TextMessage(content=f"Unexpected error: {str(e)}", source=self.name))

class ValidationAgent(AnalyticsAssistantAgent):
    """
    Agent responsible for validating configurations and manifests.
    
    This agent performs validation checks on various system components,
    including manifest files and configuration settings.
    
    Attributes:
        llm_config (Dict[str, Any]): Configuration for the LLM model
        analytics (ToolAnalytics): Analytics tracking for tool usage
    """
    
    def __init__(self, name: str, description: str, llm_config: Dict[str, Any]):
        """
        Initialize the ValidationAgent.

        Args:
            name (str): Name of the agent
            description (str): Description of the agent's role
            llm_config (Dict[str, Any]): Configuration for the LLM model
            
        Raises:
            ValueError: If name, description, or llm_config is invalid
        """
        if not name or not description:
            raise ValueError("Name and description must not be empty")
        if not llm_config:
            raise ValueError("LLM configuration must be provided")
            
        super().__init__(name, description=description)
        self.llm_config = llm_config
        self.analytics = ToolAnalytics()

    async def validate_configuration(self, content: str) -> Dict[str, Any]:
        """
        Validate configuration content.

        Args:
            content (str): Configuration content to validate

        Returns:
            Dict[str, Any]: Validation results
            
        Raises:
            ManifestValidationError: If validation fails
        """
        try:
            # Process validation logic here
            # This is a placeholder for the actual implementation
            return {"status": "success", "validation": "passed"}
        except Exception as e:
            raise ManifestValidationError(f"Validation failed: {str(e)}")

    async def on_messages(self, messages: List[BaseChatMessage], cancellation_token) -> Response:
        """
        Process validation requests.

        Args:
            messages (List[BaseChatMessage]): List of messages containing validation requests
            cancellation_token: Token for cancellation support

        Returns:
            Response: Response containing the validation result
            
        Raises:
            ManifestValidationError: If validation fails
        """
        try:
            result = await self.validate_configuration(messages[0].content)
            return Response(chat_message=TextMessage(content=json.dumps(result), source=self.name))
        except ManifestValidationError as e:
            logger.error(f"Validation error: {str(e)}")
            return Response(chat_message=TextMessage(content=f"Error: {str(e)}", source=self.name))
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return Response(chat_message=TextMessage(content=f"Unexpected error: {str(e)}", source=self.name))

def create_agents() -> Dict[str, Any]:
    """
    Create and initialize all required agents.
    
    Returns:
        Dict[str, Any]: Dictionary containing initialized agents
        
    Raises:
        ConfigError: If agent creation fails
    """
    try:
        config = load_config()
        agents = {
            "file_reader": FileReaderAgent(
                name="FileReader",
                description="Reads and processes manifest files",
                manifest=config.get("manifest_files", []),
                file_log=FILE_LOG
            ),
            "manifest_updater": ManifestUpdaterAgent(
                name="ManifestUpdater",
                description="Updates manifest files",
                llm_config=config.get("llm_config", {})
            ),
            "logging_config": LoggingConfigAgent(
                name="LoggingConfig",
                description="Configures logging settings",
                llm_config=config.get("llm_config", {})
            ),
            "validator": ValidationAgent(
                name="Validator",
                description="Validates configurations and manifests",
                llm_config=config.get("llm_config", {})
            )
        }
        return agents
    except Exception as e:
        raise ConfigError(f"Failed to create agents: {str(e)}")

def create_state_manager() -> StateManager:
    """
    Create and initialize the state manager.
    
    Returns:
        StateManager: Initialized state manager
        
    Raises:
        ConfigError: If state manager creation fails
    """
    try:
        config = load_config()
        return StateManager(
            state_file=config.get("state_file", "state.json"),
            backup_dir=config.get("backup_dir", "backups")
        )
    except Exception as e:
        raise ConfigError(f"Failed to create state manager: {str(e)}")

def create_workflow(agents: Dict[str, Any]) -> CollaborativeWorkflow:
    """
    Create and initialize the collaborative workflow.
    
    Args:
        agents (Dict[str, Any]): Dictionary of initialized agents
        
    Returns:
        CollaborativeWorkflow: Initialized workflow
        
    Raises:
        ConfigError: If workflow creation fails
    """
    try:
        config = load_config()
        return CollaborativeWorkflow(
            agents=agents,
            config=config.get("workflow_config", {}),
            analytics=ToolAnalytics()
        )
    except Exception as e:
        raise ConfigError(f"Failed to create workflow: {str(e)}")

async def process_manifest_update(
    workflow: CollaborativeWorkflow,
    manifest_path: str,
    update_requirements: UpdateRequirements,
    state_manager: StateManager
) -> Dict[str, Any]:
    """
    Process a manifest update request.
    
    Args:
        workflow (CollaborativeWorkflow): Initialized workflow
        manifest_path (str): Path to the manifest file
        update_requirements (UpdateRequirements): Update requirements
        state_manager (StateManager): State manager instance
        
    Returns:
        Dict[str, Any]: Update results
        
    Raises:
        ManifestError: If manifest update fails
    """
    try:
        # Process manifest update logic here
        # This is a placeholder for the actual implementation
        return {"status": "success", "message": "Manifest updated successfully"}
    except Exception as e:
        raise ManifestError(f"Failed to process manifest update: {str(e)}")

async def main() -> None:
    """
    Main entry point for the manifest update program.
    
    Raises:
        Exception: If program execution fails
    """
    try:
        # Initialize components
        agents = create_agents()
        state_manager = create_state_manager()
        workflow = create_workflow(agents)
        
        # Initialize workflow
        await workflow.initialize()
        
        # Process manifest update
        update_requirements: UpdateRequirements = {
            "type": "dependency_update",
            "target_dependencies": ["package1", "package2"],
            "version_constraints": {
                "package1": ">=2.0.0",
                "package2": ">=1.5.0"
            }
        }
        
        result = await process_manifest_update(
            workflow=workflow,
            manifest_path="path/to/manifest.json",
            update_requirements=update_requirements,
            state_manager=state_manager
        )
        
        logger.info(f"Manifest update completed: {result}")
    except Exception as e:
        logger.error(f"Program execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
