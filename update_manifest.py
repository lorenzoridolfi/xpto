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
from typing import Dict, List, Any
from pathlib import Path

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

# Global logger instance
logger = logging.getLogger("update_manifest")

def load_config():
    """Load configuration from the new config directory structure."""
    config = {}
    
    # Load shared configurations
    shared_configs = [
        'config/shared/global_settings.json',
        'config/shared/base_config.json',
        'config/shared/agent_settings.json',
        'config/shared/logging_settings.json'
    ]
    
    for config_file in shared_configs:
        with open(config_file) as f:
            config.update(json.load(f))
    
    # Load program-specific configuration
    with open('config/update_manifest/program_config.json') as f:
        config.update(json.load(f))
    
    return config

def load_manifest_schema():
    """Load manifest validation schema from the new config directory structure."""
    with open('config/update_manifest/manifest_validation_schema.json') as f:
        return json.load(f)

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

def create_agents():
    """
    Create and configure the adaptive agents for manifest management.
    
    Returns:
        tuple: (adaptive_assistant, adaptive_user_proxy)
            - adaptive_assistant: AdaptiveAgent for manifest analysis and updates
            - adaptive_user_proxy: AdaptiveAgent for update coordination
    """
    # Create base agents with detailed system messages
    assistant = AssistantAgent(
        name="manifest_assistant",
        llm_config={"config_list": [{"model": "gpt-4"}]},
        system_message="""You are an expert in manifest file management and updates.
        Your responsibilities include:
        1. Analyzing manifest files
        2. Identifying required updates
        3. Generating update recommendations
        4. Validating changes
        5. Ensuring compatibility"""
    )

    user_proxy = UserProxyAgent(
        name="manifest_user_proxy",
        human_input_mode="TERMINATE",
        system_message="""You are a manifest update coordinator.
        Your responsibilities include:
        1. Managing update requests
        2. Coordinating with other agents
        3. Validating update proposals
        4. Tracking update history
        5. Ensuring update quality"""
    )

    # Create adaptive agents
    adaptive_assistant = AdaptiveAgent(
        base_agent=assistant,
        role="manifest_analyzer",
        capabilities=[
            "manifest_analysis",
            "update_generation",
            "compatibility_checking",
            "validation"
        ],
        description="""An adaptive agent specialized in manifest analysis and updates:
        1. Analyzes manifest files
        2. Generates update recommendations
        3. Checks compatibility
        4. Validates changes
        5. Tracks update history"""
    )

    adaptive_user_proxy = AdaptiveAgent(
        base_agent=user_proxy,
        role="update_coordinator",
        capabilities=[
            "update_coordination",
            "feedback_collection",
            "history_tracking",
            "quality_assurance"
        ],
        description="""An adaptive agent for update coordination:
        1. Coordinates update processes
        2. Collects user feedback
        3. Tracks update history
        4. Ensures update quality
        5. Manages update workflow"""
    )
    
    return adaptive_assistant, adaptive_user_proxy

def create_state_manager():
    """
    Create and configure the state management system.
    
    Returns:
        tuple: (state_manager, state_synchronizer, state_validator)
            - state_manager: Main state manager instance
            - state_synchronizer: State synchronization handler
            - state_validator: State validation handler
    """
    # Initialize state manager with advanced configuration
    state_manager = StateManager(
        storage_backend="file",
        cache_size=2000,
        validation_rules={
            "manifest_format": "json",
            "required_fields": ["name", "version", "dependencies"],
            "version_format": "semver"
        }
    )

    # Initialize state synchronizer
    state_synchronizer = StateManager.StateSynchronizer(state_manager)

    # Initialize state validator
    state_validator = StateManager.StateValidator(
        validation_rules={
            "manifest_integrity": True,
            "version_consistency": True,
            "dependency_validity": True
        }
    )
    
    return state_manager, state_synchronizer, state_validator

def create_workflow(agents):
    """
    Create and configure the collaborative workflow.
    
    Args:
        agents (tuple): (adaptive_assistant, adaptive_user_proxy)
            The adaptive agents to include in the workflow
    
    Returns:
        CollaborativeWorkflow: Configured workflow instance
    """
    adaptive_assistant, adaptive_user_proxy = agents
    
    workflow = CollaborativeWorkflow(
        agents=[adaptive_assistant, adaptive_user_proxy],
        sequential=True,
        consensus_required=True,
        consensus_threshold=0.8,
        timeout=300
    )
    
    workflow.description = """A manifest update workflow that:
    1. Analyzes manifest files
    2. Generates update recommendations
    3. Validates proposed changes
    4. Coordinates update execution
    5. Tracks update history
    6. Ensures update quality
    7. Maintains state consistency
    8. Manages update dependencies"""
    
    return workflow

async def process_manifest_update(workflow, manifest_path, update_requirements, state_manager):
    """
    Process a manifest update.
    
    Args:
        workflow (CollaborativeWorkflow): The workflow to use
        manifest_path (str): Path to the manifest file
        update_requirements (dict): Update requirements
        state_manager (StateManager): State manager instance
    
    Returns:
        dict: Result of the update process
    """
    # Initialize workflow
    await workflow.initialize()
    
    # Load and validate manifest
    manifest_state = await workflow.agents[0].analyze_manifest(
        manifest_path=manifest_path,
        validation_rules=state_manager.validation_rules
    )
    
    # Generate update recommendations
    recommendations = await workflow.agents[0].generate_recommendations(
        manifest_state=manifest_state,
        requirements=update_requirements
    )
    
    # Coordinate update process
    update_result = await workflow.agents[1].coordinate_update(
        recommendations=recommendations,
        context={
            "manifest_path": manifest_path,
            "requirements": update_requirements
        }
    )
    
    # Save final state
    state_manager.save_state(
        entity_id="workflow",
        state={
            "manifest_path": manifest_path,
            "update_requirements": update_requirements,
            "recommendations": recommendations,
            "update_result": update_result
        }
    )
    
    return update_result

async def main():
    """
    Main entry point for the update manifest program.
    Demonstrates the basic usage of the enhanced agent architecture.
    """
    try:
        # Load configuration
        config = load_config()
        
        # Load manifest schema
        manifest_schema = load_manifest_schema()
        
        # Create components
        agents = create_agents()
        state_manager, state_synchronizer, state_validator = create_state_manager()
        workflow = create_workflow(agents)
        
        # Initialize workflow
        await workflow.initialize()
        
        # Define update requirements
        update_requirements = {
            "type": "dependency_update",
            "target_dependencies": ["package1", "package2"],
            "version_constraints": {
                "package1": ">=2.0.0",
                "package2": ">=1.5.0"
            },
            "compatibility_checks": True,
            "validation_requirements": {
                "format": "json",
                "semver": True,
                "dependency_tree": True
            }
        }
        
        # Process manifest update
        manifest_path = "path/to/manifest.json"
        update_result = await process_manifest_update(
            workflow=workflow,
            manifest_path=manifest_path,
            update_requirements=update_requirements,
            state_manager=state_manager
        )
        
        # Generate update report
        report = await workflow.agents[0].generate_update_report(
            update_result=update_result,
            include_recommendations=True
        )
        
        print(f"Update Report:\n{report}")
    except Exception as e:
        print(f"Error processing manifest update: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
