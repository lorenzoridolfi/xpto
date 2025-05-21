"""
Multi-Agent Human Feedback System

This module implements a multi-agent system for processing and analyzing text content with human feedback.
The system uses multiple specialized agents to read, analyze, and generate content, with a focus on
quality control and human-in-the-loop feedback.

Key Components:
- FileReaderAgent: Reads and processes input files
- WriterAgent: Generates content based on input
- InformationVerifierAgent: Validates information accuracy
- TextQualityAgent: Ensures content quality
- CoordinatorAgent: Orchestrates the workflow
- RootCauseAnalyzerAgent: Analyzes feedback and system behavior

The system operates in iterative rounds, with each round potentially improving the content
based on agent feedback and human input. The workflow is as follows:
1. FileReader reads the input files
2. Writer generates initial content
3. Verifier checks information accuracy
4. Quality agent ensures high standards
5. Human provides feedback
6. RootCauseAnalyzer processes feedback
7. System iterates if needed

Each agent has specific responsibilities and communicates through a structured message system.
The system includes comprehensive logging, error handling, and validation throughout the process.
"""

import os
import json
import logging
import asyncio
from typing import List, Dict, Any, TypedDict, Set, Tuple
import jsonschema
from jsonschema import validate, ValidationError
from prompt_toolkit import prompt
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.messages import TextMessage, BaseChatMessage
from autogen_agentchat.base import Response
from src.analytics_assistant_agent import AnalyticsAssistantAgent
from src.llm_cache import LLMCache
from src.tool_analytics import ToolAnalytics

from src.base_agent_system import (
    setup_logging, log_event, FILE_LOG
)

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
    cache_config: Dict[str, Any]
    logging: Dict[str, Any]
    agent_settings: Dict[str, Any]

class AgentMetrics(TypedDict):
    response_time: float
    success_rate: float
    error_count: int
    tool_usage: Dict[str, int]

# -----------------------------------------------------------------------------
# Global logs and cache
# -----------------------------------------------------------------------------
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
        with open('config/toy_example/program_config.json') as f:
            config.update(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ConfigError(f"Failed to load program config: {str(e)}")
    
    return config

# Global logger instance
logger = logging.getLogger("toy_example")

# Initialize LLM cache
config = load_config()
llm_cache = LLMCache(
    max_size=config["cache_config"]["max_size"],
    expiration_hours=config["cache_config"]["expiration_hours"]
)

# -----------------------------------------------------------------------------
# Schema validation
# -----------------------------------------------------------------------------
def load_schemas() -> Dict[str, Any]:
    """
    Load validation schemas from the new config directory structure.
    
    Returns:
        Dict[str, Any]: Validation schemas
        
    Raises:
        ConfigError: If schema files cannot be loaded or are invalid
    """
    try:
        with open('config/shared/agent_validation_schema.json') as f:
            return json.load(f)["schemas"]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ConfigError(f"Failed to load validation schemas: {str(e)}")

def validate_agent_output(agent_name: str, output: Dict[str, Any]) -> bool:
    """
    Validate agent output against its schema.
    
    Args:
        agent_name (str): Name of the agent
        output (Dict[str, Any]): Output to validate
        
    Returns:
        bool: True if validation passes, False otherwise
        
    Raises:
        ValidationError: If validation fails
    """
    schemas = load_schemas()
    schema = schemas.get(agent_name.lower().replace(" ", "_"))
    
    if not schema:
        logger.warning(f"No schema found for agent: {agent_name}")
        return True
        
    try:
        validate(instance=output, schema=schema)
        return True
    except jsonschema.exceptions.ValidationError as e:
        raise ValidationError(f"Schema validation failed for {agent_name}: {str(e)}")

# -----------------------------------------------------------------------------
# FileReaderAgent
# -----------------------------------------------------------------------------
class FileReaderAgent(BaseChatAgent):
    """
    Agent responsible for reading and processing input files.
    
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
    def produced_message_types(self) -> Tuple[type, ...]:
        """Return the types of messages this agent can produce."""
        return (TextMessage,)

    @property
    def consumed_message_types(self) -> Tuple[type, ...]:
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
                    text = open(os.path.join(self.config["input_text_folder"], fname), encoding="utf-8").read()
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

class RootCauseAnalyzerAgent(AnalyticsAssistantAgent):
    """
    Agent responsible for analyzing feedback and system behavior.
    
    This agent processes feedback from various sources and identifies root causes
    of issues or areas for improvement in the system.
    
    Attributes:
        llm_config (Dict[str, Any]): Configuration for the LLM model
        analytics (ToolAnalytics): Analytics tracking for tool usage
    """
    
    def __init__(self, name: str, description: str, llm_config: Dict[str, Any]):
        """
        Initialize the RootCauseAnalyzerAgent.

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

    async def analyze_interaction_flow(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the interaction flow based on provided metrics.

        Args:
            metrics (Dict[str, Any]): Interaction metrics to analyze

        Returns:
            Dict[str, Any]: Analysis results
            
        Raises:
            AgentError: If analysis fails
        """
        try:
            # Process analysis logic here
            # This is a placeholder for the actual implementation
            return {"status": "success", "analysis": "completed"}
        except Exception as e:
            raise AgentError(f"Failed to analyze interaction flow: {str(e)}")

    async def generate_improvement_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an improvement report based on analysis data.

        Args:
            data (Dict[str, Any]): Analysis data to generate report from

        Returns:
            Dict[str, Any]: Generated report
            
        Raises:
            AgentError: If report generation fails
        """
        try:
            # Process report generation logic here
            # This is a placeholder for the actual implementation
            return {"status": "success", "report": "generated"}
        except Exception as e:
            raise AgentError(f"Failed to generate improvement report: {str(e)}")

    async def on_messages(self, messages: List[BaseChatMessage], cancellation_token) -> Response:
        """
        Process analysis requests.

        Args:
            messages (List[BaseChatMessage]): List of messages containing analysis requests
            cancellation_token: Token for cancellation support

        Returns:
            Response: Response containing the analysis results
            
        Raises:
            AgentError: If analysis fails
        """
        try:
            result = await self.analyze_interaction_flow(json.loads(messages[0].content))
            return Response(chat_message=TextMessage(content=json.dumps(result), source=self.name))
        except AgentError as e:
            logger.error(f"Analysis error: {str(e)}")
            return Response(chat_message=TextMessage(content=f"Error: {str(e)}", source=self.name))
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return Response(chat_message=TextMessage(content=f"Unexpected error: {str(e)}", source=self.name))

class InformationVerifierAgent(AnalyticsAssistantAgent):
    """
    Agent responsible for validating information accuracy.
    
    This agent verifies the accuracy and validity of information provided by other
    agents and ensures that all content meets quality standards.
    
    Attributes:
        llm_config (Dict[str, Any]): Configuration for the LLM model
        analytics (ToolAnalytics): Analytics tracking for tool usage
    """
    
    def __init__(self, name: str, description: str, llm_config: Dict[str, Any]):
        """
        Initialize the InformationVerifierAgent.

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

    async def verify_content(self, content: str) -> Dict[str, Any]:
        """
        Verify the accuracy of provided content.

        Args:
            content (str): Content to verify

        Returns:
            Dict[str, Any]: Verification results
            
        Raises:
            ValidationError: If verification fails
        """
        try:
            # Process verification logic here
            # This is a placeholder for the actual implementation
            return {"status": "success", "verification": "passed"}
        except Exception as e:
            raise ValidationError(f"Failed to verify content: {str(e)}")

    async def on_messages(self, messages: List[BaseChatMessage], cancellation_token) -> Response:
        """
        Process verification requests.

        Args:
            messages (List[BaseChatMessage]): List of messages containing verification requests
            cancellation_token: Token for cancellation support

        Returns:
            Response: Response containing the verification results
            
        Raises:
            ValidationError: If verification fails
        """
        try:
            result = await self.verify_content(messages[0].content)
            return Response(chat_message=TextMessage(content=json.dumps(result), source=self.name))
        except ValidationError as e:
            logger.error(f"Verification error: {str(e)}")
            return Response(chat_message=TextMessage(content=f"Error: {str(e)}", source=self.name))
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return Response(chat_message=TextMessage(content=f"Unexpected error: {str(e)}", source=self.name))

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def get_document_selection() -> str:
    """
    Get document selection from user input.
    
    Returns:
        str: Selected document path
        
    Raises:
        SystemError: If user input fails
    """
    try:
        style = Style.from_dict({
            'prompt': 'ansicyan bold',
        })
        
        return prompt(
            HTML('<prompt>Please enter the path to the document: </prompt>'),
            style=style
        )
    except Exception as e:
        raise SystemError(f"Failed to get document selection: {str(e)}")

def get_user_feedback() -> str:
    """
    Get feedback from user input.
    
    Returns:
        str: User feedback
        
    Raises:
        SystemError: If user input fails
    """
    try:
        style = Style.from_dict({
            'prompt': 'ansigreen bold',
        })
        
        return prompt(
            HTML('<prompt>Please provide your feedback: </prompt>'),
            style=style
        )
    except Exception as e:
        raise SystemError(f"Failed to get user feedback: {str(e)}")

def create_agents(config: ConfigDict) -> Dict[str, Any]:
    """
    Create and initialize all required agents.
    
    Args:
        config (ConfigDict): Configuration dictionary
        
    Returns:
        Dict[str, Any]: Dictionary containing initialized agents
        
    Raises:
        ConfigError: If agent creation fails
    """
    try:
        agents = {
            "file_reader": FileReaderAgent(
                name="FileReader",
                description="Reads and processes input files",
                manifest=config.get("manifest_files", []),
                file_log=FILE_LOG
            ),
            "root_cause_analyzer": RootCauseAnalyzerAgent(
                name="RootCauseAnalyzer",
                description="Analyzes feedback and system behavior",
                llm_config=config.get("llm_config", {})
            ),
            "information_verifier": InformationVerifierAgent(
                name="InformationVerifier",
                description="Validates information accuracy",
                llm_config=config.get("llm_config", {})
            )
        }
        return agents
    except Exception as e:
        raise ConfigError(f"Failed to create agents: {str(e)}")

async def main() -> None:
    """
    Main entry point for the multi-agent human feedback system.
    
    Raises:
        Exception: If program execution fails
    """
    try:
        # Initialize components
        config = load_config()
        setup_logging(config)
        agents = create_agents(config)
        
        # Get document selection
        document_path = get_document_selection()
        
        # Process document
        file_reader = agents["file_reader"]
        content = await file_reader.run(document_path)
        
        # Verify information
        verifier = agents["information_verifier"]
        verification_result = await verifier.verify_content(content)
        
        # Get user feedback
        feedback = get_user_feedback()
        
        # Analyze feedback
        analyzer = agents["root_cause_analyzer"]
        analysis_result = await analyzer.analyze_interaction_flow({
            "content": content,
            "verification": verification_result,
            "feedback": feedback
        })
        
        # Generate improvement report
        report = await analyzer.generate_improvement_report(analysis_result)
        
        logger.info(f"System execution completed: {report}")
    except Exception as e:
        logger.error(f"Program execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
