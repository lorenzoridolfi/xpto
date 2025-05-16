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
import datetime
from typing import List, Dict, Any, Optional
import jsonschema
from jsonschema import validate, ValidationError
import sys
from prompt_toolkit import prompt
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
import psutil
import time
from pathlib import Path

from autogen import AssistantAgent, UserProxyAgent
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.messages import TextMessage, BaseChatMessage
from autogen_agentchat.base import Response
from autogen_ext.models.openai import OpenAIChatCompletionClient
from src.json_validator_tool import get_tool_for_agent
from src.tool_analytics import ToolAnalytics, ToolUsageMetrics
from src.analytics_assistant_agent import AnalyticsAssistantAgent
from src.llm_cache import LLMCache
import autogen
from autogen import GroupChat, GroupChatManager

from src.base_agent_system import (
    setup_logging, log_event, create_base_agents, create_group_chat,
    load_json_file, save_json_file, FILE_LOG, ROOT_CAUSE_DATA
)

# -----------------------------------------------------------------------------
# Global logs and cache
# -----------------------------------------------------------------------------
# Load configuration
config = load_json_file("toy_example.json")

# Global logger instance
logger = logging.getLogger("toy_example")

# Initialize LLM cache
llm_cache = LLMCache(
    max_size=config["cache_config"]["max_size"],
    similarity_threshold=config["cache_config"]["similarity_threshold"],
    expiration_hours=config["cache_config"]["expiration_hours"]
)

# -----------------------------------------------------------------------------
# Schema validation
# -----------------------------------------------------------------------------
def load_schemas() -> Dict[str, Any]:
    """
    Load JSON schemas for agent output validation.
    
    Returns:
        Dict[str, Any]: Dictionary containing all agent schemas
    """
    try:
        with open("agent_schemas.json", "r", encoding="utf-8") as f:
            return json.load(f)["schemas"]
    except Exception as e:
        logger.error(f"Error loading schemas: {e}")
        return {}

def validate_agent_output(agent_name: str, output: Dict[str, Any]) -> bool:
    """
    Validate agent output against its schema.
    
    Args:
        agent_name (str): Name of the agent
        output (Dict[str, Any]): Output to validate
        
    Returns:
        bool: True if validation passes, False otherwise
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
        logger.error(f"Schema validation failed for {agent_name}: {e}")
        return False

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
def setup_logging(config: dict) -> None:
    """
    Configure logging based on configuration settings.

    Args:
        config (dict): Configuration dictionary containing logging settings
    """
    log_config = config["logging"]
    
    # Set up logger
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
    
    # Don't propagate to root logger to avoid duplicate logs
    logger.propagate = False

# -----------------------------------------------------------------------------
# Event logger
# -----------------------------------------------------------------------------
def log_event(agent_name: str, event_type: str, inputs: List[BaseChatMessage], outputs) -> None:
    """
    Log an event in the system with detailed information about inputs and outputs.

    Args:
        agent_name (str): Name of the agent generating the event
        event_type (str): Type of event (e.g., 'on_messages_invoke', 'on_messages_complete')
        inputs (List[BaseChatMessage]): Input messages to the agent
        outputs: Output from the agent (can be Response, list of Responses, or other types)
    """
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "agent": agent_name,
        "event": event_type,
        "inputs": [{"source": m.source, "content": m.content} for m in inputs],
    }
    
    if isinstance(outputs, Response):
        cm = outputs.chat_message
        try:
            # Try to parse JSON content
            content = json.loads(cm.content)
            # Validate against schema
            if not validate_agent_output(agent_name, content):
                logger.warning(f"Invalid output format from {agent_name}")
            entry["outputs"] = [{"source": cm.source, "content": content}]
        except json.JSONDecodeError:
            # If not JSON, store as is
            entry["outputs"] = [{"source": cm.source, "content": cm.content}]
    elif isinstance(outputs, list) and all(isinstance(o, Response) for o in outputs):
        entry["outputs"] = []
        for o in outputs:
            try:
                content = json.loads(o.chat_message.content)
                if not validate_agent_output(agent_name, content):
                    logger.warning(f"Invalid output format from {agent_name}")
                entry["outputs"].append({"source": o.chat_message.source, "content": content})
            except json.JSONDecodeError:
                entry["outputs"].append({"source": o.chat_message.source, "content": o.chat_message.content})
    elif isinstance(outputs, list) and all(isinstance(o, BaseChatMessage) for o in outputs):
        entry["outputs"] = []
        for o in outputs:
            try:
                content = json.loads(o.content)
                if not validate_agent_output(agent_name, content):
                    logger.warning(f"Invalid output format from {agent_name}")
                entry["outputs"].append({"source": o.source, "content": content})
            except json.JSONDecodeError:
                entry["outputs"].append({"source": o.source, "content": o.content})
    else:
        entry["outputs"] = outputs
    
    # Log the event
    logger.debug(f"Event: {json.dumps(entry, indent=2)}")
    ROOT_CAUSE_DATA.append(entry)

# -----------------------------------------------------------------------------
# Agents
# -----------------------------------------------------------------------------
class FileReaderAgent(BaseChatAgent):
    """
    Agent responsible for reading and processing input files.
    
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
        self.manifest = {f["filename"] for f in manifest}
        self.files_read = []
        self.file_log = file_log

    @property
    def produced_message_types(self):
        """Return the types of messages this agent can produce."""
        return (TextMessage,)

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

    @property
    def consumed_message_types(self):
        """Return the types of messages this agent can consume."""
        return (TextMessage,)

class RootCauseAnalyzerAgent(AnalyticsAssistantAgent):
    """
    Agent responsible for analyzing system behavior and user feedback.
    
    This agent combines configuration data, user feedback, and system logs to provide
    insights into system behavior and potential improvements.
    
    Attributes:
        name (str): Name of the agent
        description (str): Description of the agent's role
        llm_config (Dict): Configuration for the LLM model
    """

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

class InformationVerifierAgent(AnalyticsAssistantAgent):
    """
    Agent responsible for validating information accuracy and source compliance.
    
    This agent verifies that all information in the generated text is:
    1. Present in the source files
    2. Not contradicted by the source material
    3. Properly supported by the sources
    4. Free from hallucinations or additions not present in the sources
    
    Attributes:
        name (str): Name of the agent
        description (str): Description of the agent's role
        llm_config (Dict): Configuration for the LLM model
        source_files (list): List of source files to verify against
        source_content (dict): Dictionary mapping filenames to their content
    """

    def __init__(self, name: str, description: str, llm_config: Dict):
        """
        Initialize the InformationVerifierAgent.

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
        self.source_files = []
        self.source_content = {}

    async def verify_content(self, content: str) -> Dict[str, Any]:
        """
        Verify the content against source files.

        Args:
            content (str): The content to verify

        Returns:
            Dict[str, Any]: Verification results including factual accuracy, source compliance,
                           logical consistency, and unsupported claims
        """
        # Prepare verification prompt
        prompt = f"""
        Please verify the following content against these source files:
        
        Source Files:
        {json.dumps(self.source_files, indent=2)}
        
        Content to Verify:
        {content}
        
        Please provide a detailed verification report in JSON format with the following structure:
        {{
            "verification_status": "PASS" | "FAIL" | "NEEDS_IMPROVEMENT",
            "verification_results": {{
                "factual_accuracy": {{
                    "status": "PASS" | "FAIL" | "NEEDS_IMPROVEMENT",
                    "issues": [
                        {{
                            "type": "string",
                            "description": "string",
                            "location": "string",
                            "suggestion": "string"
                        }}
                    ]
                }},
                "source_compliance": {{
                    "status": "PASS" | "FAIL" | "NEEDS_IMPROVEMENT",
                    "issues": [
                        {{
                            "type": "string",
                            "description": "string",
                            "location": "string",
                            "suggestion": "string"
                        }}
                    ],
                    "sources_used": [
                        {{
                            "filename": "string",
                            "content_references": [
                                {{
                                    "text": "string",
                                    "location": "string"
                                }}
                            ]
                        }}
                    ]
                }},
                "logical_consistency": {{
                    "status": "PASS" | "FAIL" | "NEEDS_IMPROVEMENT",
                    "issues": [
                        {{
                            "type": "string",
                            "description": "string",
                            "location": "string",
                            "suggestion": "string"
                        }}
                    ]
                }},
                "unsupported_claims": [
                    {{
                        "claim": "string",
                        "location": "string",
                        "suggestion": "string"
                    }}
                ]
            }},
            "summary": "string",
            "termination_reason": "string"
        }}
        
        For each claim in the content:
        1. Check if it's present in any source file
        2. Verify it's not contradicted by any source
        3. Ensure it's properly supported by the sources
        4. Look for any additions not present in the sources
        
        If you find any issues, provide specific feedback. If the content is accurate and fully supported by the sources, set verification_status to "PASS" and termination_reason to "TERMINATE".
        """

        # Get verification response
        response = await self.run(task=prompt)
        
        try:
            # Parse and validate the response
            verification_results = json.loads(response)
            validate(instance=verification_results, schema=load_schemas()["schemas"]["information_verifier"])
            return verification_results
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Error parsing verification results: {e}")
            return {
                "verification_status": "FAIL",
                "verification_results": {
                    "factual_accuracy": {"status": "FAIL", "issues": []},
                    "source_compliance": {"status": "FAIL", "issues": [], "sources_used": []},
                    "logical_consistency": {"status": "FAIL", "issues": []},
                    "unsupported_claims": []
                },
                "summary": f"Error in verification process: {str(e)}",
                "termination_reason": "Verification failed due to error"
            }

    async def on_messages(self, messages: List[BaseChatMessage], cancellation_token) -> Response:
        """
        Process verification requests.

        Args:
            messages (List[BaseChatMessage]): List of messages containing content to verify
            cancellation_token: Token for cancellation support

        Returns:
            Response: Verification results or termination message
        """
        log_event(self.name, "on_messages_invoke", messages, [])
        
        # Get content to verify from the last message
        content = messages[-1].content
        
        # Perform verification
        verification_results = await self.verify_content(content)
        
        # Check if verification passed
        if verification_results["verification_status"] == "PASS":
            resp = Response(chat_message=TextMessage(content="TERMINATE", source=self.name))
        else:
            # Format issues for feedback
            issues = []
            
            # Add factual accuracy issues
            for issue in verification_results["verification_results"]["factual_accuracy"]["issues"]:
                issues.append(f"Factual Issue: {issue['description']} (Location: {issue['location']})")
            
            # Add source compliance issues
            for issue in verification_results["verification_results"]["source_compliance"]["issues"]:
                issues.append(f"Source Issue: {issue['description']} (Location: {issue['location']})")
            
            # Add logical consistency issues
            for issue in verification_results["verification_results"]["logical_consistency"]["issues"]:
                issues.append(f"Logical Issue: {issue['description']} (Location: {issue['location']})")
            
            # Add unsupported claims
            for claim in verification_results["verification_results"]["unsupported_claims"]:
                issues.append(f"Unsupported Claim: {claim['claim']} (Location: {claim['location']})")
            
            # Create feedback message
            feedback = f"Verification failed. Issues found:\n\n" + "\n".join(issues)
            resp = Response(chat_message=TextMessage(content=feedback, source=self.name))
        
        log_event(self.name, "on_messages_complete", messages, resp)
        return resp

# -----------------------------------------------------------------------------
# User Input Functions
# -----------------------------------------------------------------------------
def get_document_selection() -> str:
    """
    Prompt the user to select which document they want the agents to generate.
    Uses prompt-toolkit for a better user experience.
    
    Returns:
        str: The selected document name
    """
    style = Style.from_dict({
        'prompt': 'ansicyan bold',
    })
    
    message = HTML('<prompt>Which document would you like the agents to generate? </prompt>')
    return prompt(message, style=style)

def get_user_feedback() -> str:
    """
    Prompt the user for feedback after text generation.
    Uses prompt-toolkit for a better user experience.
    
    Returns:
        str: The user's feedback
    """
    style = Style.from_dict({
        'prompt': 'ansigreen bold',
    })
    
    message = HTML('<prompt>Please provide your feedback on the generated text: </prompt>')
    return prompt(message, style=style)

# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------
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
    
    # Create Writer Agent
    writer_config = config["agents"]["WriterAgent"]
    writer = AssistantAgent(
        name=writer_config["name"],
        system_message=writer_config["system_message"],
        llm_config=config["llm_config"]["writer"]
    )
    logger.debug(f"Writer Agent created: {writer_config['description']}")
    
    # Create Information Verifier Agent
    verifier_config = config["agents"]["InformationVerifierAgent"]
    verifier = InformationVerifierAgent(
        name=verifier_config["name"],
        description=verifier_config["description"],
        llm_config=config["llm_config"]["verifier"]
    )
    logger.debug(f"Information Verifier Agent created: {verifier_config['description']}")
    
    # Create Text Quality Agent
    quality_config = config["agents"]["TextQualityAgent"]
    quality = AssistantAgent(
        name=quality_config["name"],
        system_message=quality_config["system_message"],
        llm_config=config["llm_config"]["quality"]
    )
    logger.debug(f"Text Quality Agent created: {quality_config['description']}")
    
    # Create Group Chat Manager
    all_agents = [
        base_agents["user_proxy"],
        file_reader,
        writer,
        verifier,
        quality,
        base_agents["supervisor"]
    ]
    group_chat_manager = create_group_chat(all_agents, config, logger)
    
    return {
        "file_reader": file_reader,
        "writer": writer,
        "verifier": verifier,
        "quality": quality,
        "user_proxy": base_agents["user_proxy"],
        "supervisor": base_agents["supervisor"],
        "group_chat_manager": group_chat_manager
    }

def main():
    """
    Main entry point for the multi-agent human feedback system.
    
    This function:
    1. Loads the configuration
    2. Sets up logging
    3. Creates all necessary agents
    4. Initializes the text processing workflow
    5. Handles any errors that occur during execution
    """
    try:
        # Load configuration
        config = load_json_file("toy_example.json")
        
        # Setup logging
        setup_logging(config)
        
        # Create agents
        agents = create_agents(config)
        
        # Start processing
        logger.info("Starting text processing with human feedback")
        
        # Initialize the process
        agents["user_proxy"].initiate_chat(
            agents["group_chat_manager"],
            message="""Let's process and analyze the text content.
            The Supervisor should coordinate the process:
            1. FileReader should read the input files
            2. Writer should generate content
            3. Verifier should check accuracy
            4. Quality agent should ensure high standards
            5. Supervisor should ensure the process completes successfully"""
        )
        
        logger.info("Text processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()
