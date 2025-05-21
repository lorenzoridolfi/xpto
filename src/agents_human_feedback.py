"""
Multi-Agent System for Human Feedback Analysis

This module implements a multi-agent system that processes files, generates summaries,
and performs root cause analysis based on user feedback. The system uses a combination
of specialized agents to handle different aspects of the workflow, from file reading
to analysis generation.

Key Components:
- Configuration Management: Versioned configuration system with backup support
- File Operations: Asynchronous file reading with error handling and retries
- Event Logging: Comprehensive logging of agent interactions and system events
- Root Cause Analysis: Structured analysis of issues and recommendations
"""

import os
import json
import logging
import asyncio
import datetime
from typing import List, Dict, Union, Any
from dataclasses import dataclass
from enum import Enum
import shutil
import aiofiles

from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseChatMessage
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from .analytics_assistant_agent import AnalyticsAssistantAgent
from .tool_analytics import ToolAnalytics
from .llm_cache import LLMCache
from .config import DEFAULT_CONFIG

# -----------------------------------------------------------------------------
# Configuration Management
# -----------------------------------------------------------------------------

class ConfigVersion:
    """
    Represents a versioned configuration with timestamp.
    
    Attributes:
        version (str): Semantic version number (e.g., '1.0.0')
        timestamp (str): ISO format timestamp of when the version was created
    """
    def __init__(self, version: str, timestamp: str):
        self.version = version
        self.timestamp = timestamp

    @classmethod
    def from_dict(cls, data: dict) -> 'ConfigVersion':
        """Create a ConfigVersion instance from a dictionary."""
        return cls(data['version'], data['timestamp'])

    def to_dict(self) -> dict:
        """Convert the ConfigVersion instance to a dictionary."""
        return {
            'version': self.version,
            'timestamp': self.timestamp
        }

def load_config_with_versioning(config_path: str, default_config: dict = None) -> tuple[dict, ConfigVersion]:
    """
    Load configuration with versioning support and handle updates.
    
    Args:
        config_path: Path to the configuration file
        default_config: Default configuration to use if file doesn't exist
        
    Returns:
        Tuple of (configuration dictionary, ConfigVersion instance)
    """
    if default_config is None:
        default_config = DEFAULT_CONFIG
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            version = ConfigVersion.from_dict(config_data.get('_version', {'version': '1.0.0', 'timestamp': datetime.datetime.now().isoformat()}))
            config = {k: v for k, v in config_data.items() if not k.startswith('_')}
            
            # Check if config needs update
            if config != default_config:
                print(f"\nCurrent config version: {version.version}")
                print("Configuration has been modified. Would you like to:")
                print("1. Keep the current configuration")
                print("2. Update to the default configuration")
                print("3. Create a new version with your changes")
                
                choice = input("Enter your choice (1/2/3): ").strip()
                
                if choice == "2":
                    config = default_config
                    version = ConfigVersion("1.0.0", datetime.datetime.now().isoformat())
                elif choice == "3":
                    new_version = input("Enter new version number (e.g., 1.0.1): ").strip()
                    version = ConfigVersion(new_version, datetime.datetime.now().isoformat())
                
                # Save the updated config
                save_config_with_versioning(config_path, config, version)
    else:
        config = default_config
        version = ConfigVersion("1.0.0", datetime.datetime.now().isoformat())
        save_config_with_versioning(config_path, config, version)
    
    return config, version

def save_config_with_versioning(config_path: str, config: dict, version: ConfigVersion):
    """
    Save configuration with versioning information and create backup.
    
    Args:
        config_path: Path where to save the configuration
        config: Configuration dictionary to save
        version: ConfigVersion instance with version information
    """
    config_data = config.copy()
    config_data['_version'] = version.to_dict()
    
    # Create backup of existing config if it exists
    if os.path.exists(config_path):
        backup_path = f"{config_path}.bak"
        shutil.copy2(config_path, backup_path)
    
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)

# -----------------------------------------------------------------------------
# Error Handling
# -----------------------------------------------------------------------------

@dataclass
class FileReadError:
    """
    Represents an error that occurred during file reading.
    
    Attributes:
        file_path: Path to the file that caused the error
        error_message: Description of the error
        error_type: Type of the error (e.g., 'FileNotFoundError')
        timestamp: When the error occurred
    """
    file_path: str
    error_message: str
    error_type: str
    timestamp: str

class ErrorSeverity(Enum):
    """Severity levels for agent errors."""
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class AgentError:
    """
    Represents an error that occurred during agent operation.
    
    Attributes:
        agent_name: Name of the agent that encountered the error
        error_message: Description of the error
        severity: ErrorSeverity level
        timestamp: When the error occurred
        context: Additional context about the error
    """
    agent_name: str
    error_message: str
    severity: ErrorSeverity
    timestamp: str
    context: Dict[str, Any]

# -----------------------------------------------------------------------------
# File Operations
# -----------------------------------------------------------------------------

class FileReaderAgent:
    """
    Handles asynchronous file reading operations with retry logic and error handling.
    
    Attributes:
        config: Configuration dictionary
        file_manifest: List of files to process
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
    """
    def __init__(self, config: dict):
        self.config = config
        self.file_manifest = config.get('file_manifest', [])
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1)

    async def read_file(self, fname: str) -> Union[str, FileReadError]:
        """
        Asynchronously read a file with retry logic and error handling.
        
        Args:
            fname: Path to the file to read
            
        Returns:
            File contents as string or FileReadError if reading failed
        """
        for attempt in range(self.max_retries):
            try:
                async with aiofiles.open(fname, encoding='utf-8') as f:
                    content = await f.read()
                    return content
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return FileReadError(
                        file_path=fname,
                        error_message=str(e),
                        error_type=type(e).__name__,
                        timestamp=datetime.datetime.now().isoformat()
                    )
                await asyncio.sleep(self.retry_delay)

    async def process_files(self) -> List[Union[str, FileReadError]]:
        """
        Process all files in the manifest asynchronously.
        
        Returns:
            List of file contents or FileReadError instances
        """
        tasks = [self.read_file(fname) for fname in self.file_manifest]
        return await asyncio.gather(*tasks)

# -----------------------------------------------------------------------------
# Root Cause Analysis
# -----------------------------------------------------------------------------

@dataclass
class RootCauseInput:
    """
    Input data for root cause analysis.
    
    Attributes:
        config: Current configuration
        user_feedback: User's feedback about the system
        action_log: Log of actions taken by agents
        event_log: Log of system events
    """
    config: Dict[str, Any]
    user_feedback: str
    action_log: List[Dict[str, Any]]
    event_log: List[Dict[str, Any]]

class RootCauseAnalyzerAgent(AnalyticsAssistantAgent):
    """
    Analyzes system behavior and user feedback to identify root causes and recommendations.
    
    This agent combines configuration data, user feedback, and system logs to provide
    insights into system behavior and potential improvements.
    """
    def __init__(self, config: dict):
        super().__init__(
            name="root_cause_analyzer",
            model_client=OpenAIChatCompletionClient(),
            system_message=self._create_system_message(),
            tools=None,
            reflect_on_tool_use=True,
            cache=LLMCache() if config.get("cache", {}).get("enabled", False) else None
        )
        self.config = config

    def _create_system_message(self) -> str:
        """Create the system message defining the agent's behavior and output format."""
        return """You are RootCauseAnalyzerAgent. Analyze the provided information and output a JSON response with the following structure:
        {
            "root_causes": [
                {
                    "description": "string",
                    "severity": "high|medium|low",
                    "evidence": ["string"],
                    "affected_components": ["string"]
                }
            ],
            "recommendations": [
                {
                    "description": "string",
                    "priority": "high|medium|low",
                    "implementation_steps": ["string"]
                }
            ],
            "confidence_score": float
        }"""

    async def analyze_interaction_flow(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the entire interaction flow between agents.
        
        Args:
            metrics: Dictionary containing metrics from all agents
            
        Returns:
            Dictionary containing interaction analysis results
        """
        log_event(self.name, "analyze_interaction_flow_start", metrics, [])
        
        # Prepare analysis prompt
        prompt = f"""
        Analyze the following agent interaction metrics and provide insights:
        
        Metrics:
        {json.dumps(metrics, indent=2)}
        
        Please provide a detailed analysis in JSON format with the following structure:
        {{
            "interaction_patterns": [
                {{
                    "pattern": "string",
                    "frequency": "number",
                    "impact": "high|medium|low",
                    "suggestion": "string"
                }}
            ],
            "communication_efficiency": {{
                "score": "number",
                "bottlenecks": ["string"],
                "improvements": ["string"]
            }},
            "workflow_optimization": {{
                "current_flow": ["string"],
                "suggested_flow": ["string"],
                "expected_improvements": ["string"]
            }},
            "agent_collaboration": {{
                "strengths": ["string"],
                "weaknesses": ["string"],
                "improvement_areas": ["string"]
            }}
        }}
        """
        
        # Get analysis response
        response = await self.run(task=prompt)
        
        log_event(self.name, "analyze_interaction_flow_complete", metrics, response)
        return response

    async def generate_improvement_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive system improvement report.
        
        Args:
            data: Dictionary containing system metrics and analysis data
            
        Returns:
            Dictionary containing improvement recommendations
        """
        log_event(self.name, "generate_improvement_report_start", data, [])
        
        # Prepare report prompt
        prompt = f"""
        Generate a comprehensive system improvement report based on the following data:
        
        Data:
        {json.dumps(data, indent=2)}
        
        Please provide a detailed report in JSON format with the following structure:
        {{
            "data_analysis": {{
                "input_quality": {{
                    "score": "number",
                    "issues": ["string"],
                    "recommendations": ["string"]
                }},
                "processing_efficiency": {{
                    "score": "number",
                    "bottlenecks": ["string"],
                    "optimizations": ["string"]
                }},
                "transformation_accuracy": {{
                    "score": "number",
                    "errors": ["string"],
                    "improvements": ["string"]
                }},
                "validation_results": {{
                    "score": "number",
                    "issues": ["string"],
                    "enhancements": ["string"]
                }}
            }},
            "agent_performance": {{
                "response_accuracy": {{
                    "overall_score": "number",
                    "agent_scores": {{
                        "agent_name": "number"
                    }},
                    "improvements": ["string"]
                }},
                "processing_speed": {{
                    "overall_score": "number",
                    "agent_scores": {{
                        "agent_name": "number"
                    }},
                    "optimizations": ["string"]
                }},
                "error_rates": {{
                    "overall_score": "number",
                    "agent_scores": {{
                        "agent_name": "number"
                    }},
                    "error_patterns": ["string"],
                    "prevention_strategies": ["string"]
                }}
            }},
            "system_optimization": {{
                "resource_utilization": {{
                    "score": "number",
                    "issues": ["string"],
                    "recommendations": ["string"]
                }},
                "processing_bottlenecks": {{
                    "identified": ["string"],
                    "solutions": ["string"]
                }},
                "cache_effectiveness": {{
                    "score": "number",
                    "issues": ["string"],
                    "optimizations": ["string"]
                }},
                "api_efficiency": {{
                    "score": "number",
                    "issues": ["string"],
                    "improvements": ["string"]
                }},
                "memory_usage": {{
                    "score": "number",
                    "issues": ["string"],
                    "optimizations": ["string"]
                }}
            }},
            "recommendations": [
                {{
                    "area": "string",
                    "priority": "high|medium|low",
                    "description": "string",
                    "implementation_steps": ["string"],
                    "expected_impact": "string"
                }}
            ],
            "summary": "string"
        }}
        """
        
        # Get report response
        response = await self.run(task=prompt)
        
        log_event(self.name, "generate_improvement_report_complete", data, response)
        return response

    async def analyze(self, input_data: RootCauseInput) -> Dict[str, Any]:
        """
        Analyze the input data and return structured results.
        
        Args:
            input_data: RootCauseInput instance containing analysis data
            
        Returns:
            Dictionary containing root causes and recommendations
        """
        # Combine all analysis methods
        interaction_analysis = await self.analyze_interaction_flow(input_data.action_log)
        
        report_data = {
            "config": input_data.config,
            "user_feedback": input_data.user_feedback,
            "action_log": input_data.action_log,
            "event_log": input_data.event_log,
            "interaction_analysis": interaction_analysis
        }
        
        improvement_report = await self.generate_improvement_report(report_data)
        
        return {
            "root_causes": improvement_report.get("data_analysis", {}),
            "recommendations": improvement_report.get("recommendations", []),
            "confidence_score": self._calculate_confidence_score(improvement_report)
        }

    def _calculate_confidence_score(self, report: Dict[str, Any]) -> float:
        """Calculate confidence score based on report data."""
        scores = []
        
        # Extract scores from different sections
        for section in ["input_quality", "processing_efficiency", "transformation_accuracy", "validation_results"]:
            if section in report.get("data_analysis", {}):
                scores.append(report["data_analysis"][section].get("score", 0))
        
        # Calculate average score
        return sum(scores) / len(scores) if scores else 0.0

# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------

# Global logs for system-wide tracking
ROOT_CAUSE_DATA: List[dict] = []  # Stores root cause analysis data
FILE_LOG: List[str] = []          # Tracks file operations
ACTION_LOG: List[str] = []        # Records agent actions

# Configure logging
logger = logging.getLogger("MultiAgentSystem")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
ch = logging.StreamHandler()
ch.setFormatter(fmt)
fh = logging.FileHandler("agent_system.log", mode="w")
fh.setFormatter(fmt)
logger.addHandler(ch)
logger.addHandler(fh)

# -----------------------------------------------------------------------------
# Event Logging
# -----------------------------------------------------------------------------

def log_event(agent_name: str, event_type: str, inputs: List[BaseChatMessage], outputs):
    """
    Log an agent event with its inputs and outputs.
    
    Args:
        agent_name: Name of the agent generating the event
        event_type: Type of event (e.g., 'on_messages_invoke')
        inputs: List of input messages
        outputs: Output messages or responses
    """
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "agent": agent_name,
        "event": event_type,
        "inputs": [{"source": m.source, "content": m.content} for m in inputs],
    }
    if isinstance(outputs, Response):
        cm = outputs.chat_message
        entry["outputs"] = [{"source": cm.source, "content": cm.content}]
    elif isinstance(outputs, list) and all(isinstance(o, Response) for o in outputs):
        entry["outputs"] = [{"source": o.chat_message.source, "content": o.chat_message.content} for o in outputs]
    elif isinstance(outputs, list) and all(isinstance(o, BaseChatMessage) for o in outputs):
        entry["outputs"] = [{"source": o.source, "content": o.content} for o in outputs]
    else:
        entry["outputs"] = outputs
    ROOT_CAUSE_DATA.append(entry)

# -----------------------------------------------------------------------------
# Agent Classes
# -----------------------------------------------------------------------------

class LoggingAssistantAgent(AssistantAgent):
    """
    Assistant agent with built-in event logging.
    
    This agent extends the base AssistantAgent to automatically log all
    message processing events.
    """
    async def on_messages(self, messages: List[BaseChatMessage], cancellation_token) -> Response:
        """Process messages with logging."""
        log_event(self.name, "on_messages_invoke", messages, [])
        response = await super().on_messages(messages, cancellation_token)
        log_event(self.name, "on_messages_complete", messages, response)
        return response

    async def on_messages_stream(self, messages: List[BaseChatMessage], cancellation_token):
        """Process message stream with logging."""
        log_event(self.name, "on_messages_stream_start", messages, [])
        async for chunk in super().on_messages_stream(messages, cancellation_token):
            log_event(self.name, "on_messages_stream_chunk", messages, chunk)
            yield chunk
        log_event(self.name, "on_messages_stream_end", messages, [])

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

async def main():
    """
    Main execution function that orchestrates the multi-agent system.
    
    This function:
    1. Loads and validates configuration
    2. Initializes agents
    3. Processes files
    4. Handles errors
    5. Performs root cause analysis if needed
    """
    # Load configuration with versioning
    config, version = load_config_with_versioning('agent_config.json')
    
    # Initialize agents
    file_reader = FileReaderAgent(config)
    root_cause_analyzer = RootCauseAnalyzerAgent(config)
    
    # Process files
    results = await file_reader.process_files()
    
    # Handle results
    errors = [r for r in results if isinstance(r, FileReadError)]
    if errors:
        print(f"Encountered {len(errors)} errors during file processing:")
        for error in errors:
            print(f"- {error.file_path}: {error.error_message}")
    
    # Continue with root cause analysis if needed
    if errors:
        input_data = RootCauseInput(
            config=config,
            user_feedback="File reading errors occurred",
            action_log=[{"type": "file_read", "result": "error", "details": e.__dict__} for e in errors],
            event_log=[]
        )
        analysis = root_cause_analyzer.analyze(input_data)
        print("\nRoot Cause Analysis:")
        print(json.dumps(analysis, indent=2))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Fatal error: {e}")
