#!/usr/bin/env python3

"""
Manifest Update System

This module implements a multi-agent system for updating manifest files with proper
logging configuration and system structure. The system uses multiple specialized agents
to read, analyze, and update manifest files.

Key Components:
- FileReaderAgent: Reads and processes manifest files
- ManifestUpdaterAgent: Updates manifest content based on analysis
- LoggingConfigAgent: Configures logging settings
- ValidationAgent: Validates manifest and logging configuration
- CoordinatorAgent: Orchestrates the workflow

The system operates in a coordinated manner:
1. FileReader reads the manifest files
2. ManifestUpdater analyzes and updates the manifest
3. LoggingConfig sets up logging configuration
4. Validator ensures everything is correct
5. Coordinator manages the overall process

Each agent has specific responsibilities and communicates through a structured message system.
The system includes comprehensive logging and error handling throughout the process.
"""

import json
import logging
from typing import Dict, List, Any
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent
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

# Global logger instance
logger = logging.getLogger("update_manifest")

# Load configuration
config = load_json_file("update_manifest.json")

# Initialize LLM cache
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
        manifest (set): Set of filenames the agent can read
        files_read (list): List of files that have been read
        file_log (list): List to store file operation logs
    """
    
    def __init__(self, name: str, description: str, manifest: List[dict], file_log: List[str]):
        """
        Initialize the FileReaderAgent.

        Args:
            name (str): Name of the agent
            description (str): Description of the agent's role
            manifest (List[dict]): List of files the agent can read
            file_log (List[str]): List to store file operation logs
        """
        super().__init__(name, description=description)
        self._name = name
        self._description = description
        self.manifest = {f["filename"] for f in manifest}
        self.files_read = []
        self.file_log = file_log

    @property
    def name(self):
        """Return the agent's name."""
        return self._name

    @property
    def description(self):
        """Return the agent's description."""
        return self._description

    @property
    def produced_message_types(self):
        """Return the types of messages this agent can produce."""
        return (TextMessage,)

    @property
    def consumed_message_types(self):
        """Return the types of messages this agent can consume."""
        return (TextMessage,)

    async def run(self, task: str) -> str:
        """
        Run a task with logging.

        Args:
            task (str): The task to run, expected to be a comma-separated list of filenames

        Returns:
            str: The combined content of requested files or "NO_FILE" if no valid files requested
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
                    text = f"<error reading {fname}: {e}>"
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
            Response: Content of requested files or "NO_FILE" if no valid files requested
        """
        log_event(self.name, "on_messages_invoke", messages, [])
        instr = messages[-1].content.strip()
        
        # Handle explicit "NO_FILE" request
        if instr.upper() == "NO_FILE":
            self.file_log.append(f"{self.name}: no more files")
            resp = Response(chat_message=TextMessage(content="NO_FILE", source=self.name))
            log_event(self.name, "on_messages_complete", messages, resp)
            return resp

        # Process file requests
        requested = [fn.strip() for fn in instr.split(",") if fn.strip()]
        valid = [fn for fn in requested if fn in self.manifest and fn not in self.files_read]
        
        if not valid:
            resp = Response(chat_message=TextMessage(content="NO_FILE", source=self.name))
            log_event(self.name, "on_messages_complete", messages, resp)
            return resp

        # Read and combine file contents
        combined = []
        for fname in valid:
            try:
                text = open(fname, encoding="utf-8").read()
            except Exception as e:
                text = f"<error reading {fname}: {e}>"
            self.files_read.append(fname)
            self.file_log.append(f"{self.name}: read {fname}")
            combined.append(f"--- {fname} ---\n{text}")

        payload = "\n\n".join(combined)
        resp = Response(chat_message=TextMessage(content=payload, source=self.name))
        log_event(self.name, "on_messages_complete", messages, resp)
        return resp

    async def on_messages_stream(self, messages: List[BaseChatMessage], cancellation_token):
        """
        Process messages in a streaming fashion.

        Args:
            messages (List[BaseChatMessage]): List of messages to process
            cancellation_token: Token for cancellation support

        Yields:
            Response chunks as they are generated
        """
        response = await self.on_messages(messages, cancellation_token)
        yield response

class ManifestUpdaterAgent(AnalyticsAssistantAgent):
    """
    Agent responsible for updating manifest files.
    
    This agent analyzes manifest content and generates updates based on the analysis.
    It ensures that the manifest structure is maintained and all required fields are present.
    
    Attributes:
        name (str): Name of the agent
        description (str): Description of the agent's role
        llm_config (Dict): Configuration for the LLM model
    """
    
    def __init__(self, name: str, description: str, llm_config: Dict):
        """
        Initialize the ManifestUpdaterAgent.

        Args:
            name (str): Name of the agent
            description (str): Description of the agent's role
            llm_config (Dict): Configuration for the LLM model
        """
        super().__init__(
            name=name,
            description=description,
            llm_config=llm_config
        )

    async def update_manifest(self, content: str) -> Dict[str, Any]:
        """
        Update the manifest based on the provided content.

        Args:
            content (str): The content to analyze and update

        Returns:
            Dict[str, Any]: Update results including status and any issues found
        """
        prompt = f"""
        Please update the manifest based on the following content:
        
        Content:
        {content}
        
        Please provide the updated manifest in JSON format.
        """
        
        response = await self.run(task=prompt)
        return json.loads(response)

    async def on_messages(self, messages: List[BaseChatMessage], cancellation_token) -> Response:
        """
        Process manifest update requests.

        Args:
            messages (List[BaseChatMessage]): List of messages containing content to update
            cancellation_token: Token for cancellation support

        Returns:
            Response: Update results or termination message
        """
        log_event(self.name, "on_messages_invoke", messages, [])
        content = messages[-1].content
        update_results = await self.update_manifest(content)
        
        if update_results["status"] == "SUCCESS":
            resp = Response(chat_message=TextMessage(content="TERMINATE", source=self.name))
        else:
            feedback = f"Update failed. Issues found:\n\n" + "\n".join(update_results["issues"])
            resp = Response(chat_message=TextMessage(content=feedback, source=self.name))
        
        log_event(self.name, "on_messages_complete", messages, resp)
        return resp

class LoggingConfigAgent(AnalyticsAssistantAgent):
    """
    Agent responsible for configuring logging settings.
    
    This agent analyzes the system requirements and generates appropriate logging
    configuration. It ensures that logging is properly set up for all components.
    
    Attributes:
        name (str): Name of the agent
        description (str): Description of the agent's role
        llm_config (Dict): Configuration for the LLM model
    """
    
    def __init__(self, name: str, description: str, llm_config: Dict):
        """
        Initialize the LoggingConfigAgent.

        Args:
            name (str): Name of the agent
            description (str): Description of the agent's role
            llm_config (Dict): Configuration for the LLM model
        """
        super().__init__(
            name=name,
            description=description,
            llm_config=llm_config
        )

    async def configure_logging(self, content: str) -> Dict[str, Any]:
        """
        Configure logging based on the provided content.

        Args:
            content (str): The content to analyze for logging configuration

        Returns:
            Dict[str, Any]: Configuration results including status and any issues found
        """
        prompt = f"""
        Please configure logging based on the following content:
        
        Content:
        {content}
        
        Please provide the logging configuration in JSON format.
        """
        
        response = await self.run(task=prompt)
        return json.loads(response)

    async def on_messages(self, messages: List[BaseChatMessage], cancellation_token) -> Response:
        """
        Process logging configuration requests.

        Args:
            messages (List[BaseChatMessage]): List of messages containing content to configure
            cancellation_token: Token for cancellation support

        Returns:
            Response: Configuration results or termination message
        """
        log_event(self.name, "on_messages_invoke", messages, [])
        content = messages[-1].content
        config_results = await self.configure_logging(content)
        
        if config_results["status"] == "SUCCESS":
            resp = Response(chat_message=TextMessage(content="TERMINATE", source=self.name))
        else:
            feedback = f"Configuration failed. Issues found:\n\n" + "\n".join(config_results["issues"])
            resp = Response(chat_message=TextMessage(content=feedback, source=self.name))
        
        log_event(self.name, "on_messages_complete", messages, resp)
        return resp

class ValidationAgent(AnalyticsAssistantAgent):
    """
    Agent responsible for validating manifest and logging configuration.
    
    This agent performs comprehensive validation of both manifest and logging
    configuration to ensure they meet all requirements and are properly structured.
    
    Attributes:
        name (str): Name of the agent
        description (str): Description of the agent's role
        llm_config (Dict): Configuration for the LLM model
    """
    
    def __init__(self, name: str, description: str, llm_config: Dict):
        """
        Initialize the ValidationAgent.

        Args:
            name (str): Name of the agent
            description (str): Description of the agent's role
            llm_config (Dict): Configuration for the LLM model
        """
        super().__init__(
            name=name,
            description=description,
            llm_config=llm_config
        )

    async def validate_configuration(self, content: str) -> Dict[str, Any]:
        """
        Validate the configuration based on the provided content.

        Args:
            content (str): The content to validate

        Returns:
            Dict[str, Any]: Validation results including status and any issues found
        """
        prompt = f"""
        Please validate the following configuration:
        
        Content:
        {content}
        
        Please provide validation results in JSON format.
        """
        
        response = await self.run(task=prompt)
        return json.loads(response)

    async def on_messages(self, messages: List[BaseChatMessage], cancellation_token) -> Response:
        """
        Process validation requests.

        Args:
            messages (List[BaseChatMessage]): List of messages containing content to validate
            cancellation_token: Token for cancellation support

        Returns:
            Response: Validation results or termination message
        """
        log_event(self.name, "on_messages_invoke", messages, [])
        content = messages[-1].content
        validation_results = await self.validate_configuration(content)
        
        if validation_results["status"] == "PASS":
            resp = Response(chat_message=TextMessage(content="TERMINATE", source=self.name))
        else:
            feedback = f"Validation failed. Issues found:\n\n" + "\n".join(validation_results["issues"])
            resp = Response(chat_message=TextMessage(content=feedback, source=self.name))
        
        log_event(self.name, "on_messages_complete", messages, resp)
        return resp

def create_agents(config: Dict) -> Dict[str, Any]:
    """
    Create and configure the agents using settings from config file.
    
    Args:
        config (Dict): Configuration dictionary containing agent settings
        
    Returns:
        Dict[str, Any]: Dictionary containing all created agents
    """
    logger.debug("Creating agents...")
    
    # Create base agents (user proxy and supervisor)
    base_agents = create_base_agents(config, logger)
    
    # Create File Reader Agent
    file_reader_config = config["agents"]["FileReaderAgent"]
    file_reader = FileReaderAgent(
        name=file_reader_config["name"],
        description=file_reader_config["description"],
        manifest=config["file_manifest"],
        file_log=FILE_LOG
    )
    logger.debug(f"File Reader Agent created: {file_reader_config['description']}")
    
    # Create Manifest Updater Agent
    updater_config = config["agents"]["ManifestUpdaterAgent"]
    updater = ManifestUpdaterAgent(
        name=updater_config["name"],
        description=updater_config["description"],
        llm_config=config["llm_config"]["creator"]
    )
    logger.debug(f"Manifest Updater Agent created: {updater_config['description']}")
    
    # Create Logging Config Agent
    logging_config = config["agents"]["LoggingConfigAgent"]
    logging_agent = LoggingConfigAgent(
        name=logging_config["name"],
        description=logging_config["description"],
        llm_config=config["llm_config"]["creator"]
    )
    logger.debug(f"Logging Config Agent created: {logging_config['description']}")
    
    # Create Validation Agent
    validator_config = config["agents"]["ValidationAgent"]
    validator = ValidationAgent(
        name=validator_config["name"],
        description=validator_config["description"],
        llm_config=config["llm_config"]["validator"]
    )
    logger.debug(f"Validation Agent created: {validator_config['description']}")
    
    # Create Group Chat Manager
    all_agents = [
        base_agents["user_proxy"],
        file_reader,
        updater,
        logging_agent,
        validator,
        base_agents["supervisor"]
    ]
    group_chat_manager = create_group_chat(all_agents, config, logger)
    
    return {
        "file_reader": file_reader,
        "updater": updater,
        "logging": logging_agent,
        "validator": validator,
        "user_proxy": base_agents["user_proxy"],
        "supervisor": base_agents["supervisor"],
        "group_chat_manager": group_chat_manager
    }

def main():
    """
    Main entry point for the manifest update system.
    
    This function:
    1. Loads the configuration
    2. Sets up logging
    3. Creates all necessary agents
    4. Initializes the manifest update process
    5. Handles any errors that occur during execution
    """
    try:
        # Load configuration
        config = load_json_file("update_manifest.json")
        
        # Setup logging
        logger = setup_logging(config)
        
        # Create agents
        agents = create_agents(config)
        
        # Start processing
        logger.info("Starting manifest update process")
        
        # Initialize the process
        agents["user_proxy"].initiate_chat(
            agents["group_chat_manager"],
            message="""Let's update the manifest files with proper logging configuration.
            The Supervisor should coordinate the process:
            1. FileReader should read the manifest files
            2. ManifestUpdater should update the manifest
            3. LoggingConfig should configure logging
            4. Validator should ensure everything is correct
            5. Supervisor should ensure the process completes successfully"""
        )
        
        logger.info("Manifest update completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()
