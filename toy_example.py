"""
Multi-Agent Human Feedback System

This module implements a multi-agent system for processing and analyzing text content with human feedback.
The system uses multiple specialized agents to read, analyze, and generate content, with a focus on
quality control and human-in-the-loop feedback.

The system consists of the following components:
- FileReaderAgent: Reads and processes input files
- WriterAgent: Generates content based on input
- InformationVerifierAgent: Validates information accuracy
- TextQualityAgent: Ensures content quality
- CoordinatorAgent: Orchestrates the workflow
- RootCauseAnalyzerAgent: Analyzes feedback and system behavior

The system operates in iterative rounds, with each round potentially improving the content
based on agent feedback and human input.
"""

import os
import json
import logging
import asyncio
import datetime
from typing import List, Dict, Any, Optional
import jsonschema
from jsonschema import validate
import sys
from prompt_toolkit import prompt
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
import psutil
import time

from autogen_agentchat.agents import BaseChatAgent, AssistantAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import TextMessage, BaseChatMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from json_validator_tool import get_tool_for_agent
from tool_analytics import ToolAnalytics, ToolUsageMetrics
from analytics_assistant_agent import AnalyticsAssistantAgent
from llm_cache import LLMCache
import autogen
from autogen import UserProxyAgent, GroupChat, GroupChatManager

# -----------------------------------------------------------------------------
# Global logs and cache
# -----------------------------------------------------------------------------
# Lists to store system-wide logging information
ROOT_CAUSE_DATA: List[dict] = []  # Stores detailed event data for root cause analysis
FILE_LOG: List[str] = []         # Tracks file operations
ACTION_LOG: List[str] = []       # Records agent actions and decisions

# Global logger instance
logger = logging.getLogger("toy_example")

# Initialize LLM cache
llm_cache = LLMCache(
    max_size=1000,
    similarity_threshold=0.85,
    expiration_hours=24
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
    Agent responsible for reading and processing files from the manifest.
    
    This agent maintains a list of files it has read and can process multiple files
    in a single request. It handles file reading errors gracefully and logs all
    file operations.
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
    """

    def __init__(self, name: str, model_client, system_message: str, tools: Optional[List[Dict]] = None, cache: Optional[LLMCache] = None):
        """
        Initialize the InformationVerifierAgent.

        Args:
            name (str): Name of the agent
            model_client: The model client to use
            system_message (str): System message for the agent
            tools (Optional[List[Dict]]): List of tools available to the agent
            cache (Optional[LLMCache]): Cache instance to use for response caching
        """
        super().__init__(
            name=name,
            model_client=model_client,
            system_message=system_message,
            tools=tools,
            reflect_on_tool_use=True,
            cache=cache
        )
        self.source_files = []
        self.source_content = {}

    def set_source_files(self, files: List[str], content: Dict[str, str]):
        """
        Set the source files and their content for verification.

        Args:
            files (List[str]): List of source file names
            content (Dict[str, str]): Dictionary mapping file names to their content
        """
        self.source_files = files
        self.source_content = content

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
async def main():
    """
    Main execution function.
    
    This function:
    1. Loads configuration
    2. Initializes agents
    3. Processes files
    4. Handles errors
    5. Performs root cause analysis
    6. Generates system improvement report
    """
    try:
        # Load configuration
        config = load_json_file("toy_example.json")
        
        # Initialize agents
        file_reader = FileReaderAgent(
            name="file_reader",
            description="Reads and processes text files",
            system_message=config["agents"]["file_reader"]["system_message"]
        )
        
        writer = WriterAgent(
            name="writer",
            description="Generates content based on input",
            system_message=config["agents"]["writer"]["system_message"]
        )
        
        verifier = InformationVerifierAgent(
            name="verifier",
            description="Verifies information accuracy",
            system_message=config["agents"]["verifier"]["system_message"]
        )
        
        quality_checker = TextQualityAgent(
            name="quality_checker",
            description="Checks text quality",
            system_message=config["agents"]["quality_checker"]["system_message"]
        )
        
        analyzer = RootCauseAnalyzerAgent(
            name="analyzer",
            description="Analyzes system behavior and generates reports",
            system_message=config["agents"]["analyzer"]["system_message"]
        )
        
        # Process document
        logger.info("Starting document processing")
        
        # Read document
        document = await file_reader.run(task="Read the input document")
        logger.info("Document read successfully")
        
        # Generate content
        content = await writer.run(task=f"Generate content based on: {document}")
        logger.info("Content generated successfully")
        
        # Verify information
        verification = await verifier.run(task=f"Verify information in: {content}")
        logger.info("Information verified successfully")
        
        # Check quality
        quality = await quality_checker.run(task=f"Check quality of: {content}")
        logger.info("Quality check completed successfully")
        
        # Get user feedback
        feedback = input("Please provide feedback on the generated content: ")
        logger.info("User feedback received")
        
        # Post-feedback analysis
        logger.info("Starting post-feedback analysis")
        
        # Gather system metrics
        metrics = {
            "document_metrics": {
                "size": len(document),
                "processing_time": time.time() - start_time,
                "quality_score": quality.get("score", 0)
            },
            "agent_metrics": {
                "file_reader": {
                    "response_time": file_reader.response_time,
                    "error_count": file_reader.error_count
                },
                "writer": {
                    "response_time": writer.response_time,
                    "error_count": writer.error_count
                },
                "verifier": {
                    "response_time": verifier.response_time,
                    "error_count": verifier.error_count
                },
                "quality_checker": {
                    "response_time": quality_checker.response_time,
                    "error_count": quality_checker.error_count
                }
            },
            "system_metrics": {
                "cache_stats": llm_cache.get_stats(),
                "api_calls": get_api_metrics(),
                "memory_usage": get_memory_usage()
            },
            "user_feedback": feedback
        }
        
        # Analyze interaction flow
        interaction_analysis = await analyzer.analyze_interaction_flow(metrics)
        logger.info("Interaction flow analysis completed")
        
        # Generate improvement report
        report_data = {
            "metrics": metrics,
            "interaction_analysis": interaction_analysis,
            "bottlenecks": identify_bottlenecks(metrics),
            "cache_metrics": llm_cache.get_stats(),
            "api_metrics": get_api_metrics()
        }
        
        improvement_report = await analyzer.generate_improvement_report(report_data)
        logger.info("Improvement report generated")
        
        # Save report
        with open("system_improvement_report.json", "w") as f:
            json.dump(improvement_report, f, indent=2)
        logger.info("Report saved to system_improvement_report.json")
        
        # Log results
        logger.info("Document processing completed successfully")
        logger.info(f"Quality score: {quality.get('score', 0)}")
        logger.info(f"Processing time: {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

def identify_bottlenecks(metrics: Dict[str, Any]) -> List[str]:
    """
    Identify system bottlenecks from metrics.
    
    Args:
        metrics: Dictionary containing system metrics
        
    Returns:
        List of identified bottlenecks
    """
    bottlenecks = []
    
    # Check response times
    for agent, agent_metrics in metrics["agent_metrics"].items():
        if agent_metrics["response_time"] > 5.0:  # 5 seconds threshold
            bottlenecks.append(f"Slow response time for {agent}: {agent_metrics['response_time']:.2f}s")
    
    # Check error rates
    for agent, agent_metrics in metrics["agent_metrics"].items():
        if agent_metrics["error_count"] > 0:
            bottlenecks.append(f"High error rate for {agent}: {agent_metrics['error_count']} errors")
    
    # Check cache effectiveness
    cache_stats = metrics["system_metrics"]["cache_stats"]
    if cache_stats["hit_ratio"] < 0.5:  # 50% threshold
        bottlenecks.append(f"Low cache hit ratio: {cache_stats['hit_ratio']:.2%}")
    
    # Check memory usage
    memory_usage = metrics["system_metrics"]["memory_usage"]
    if memory_usage["percent"] > 80:  # 80% threshold
        bottlenecks.append(f"High memory usage: {memory_usage['percent']}%")
    
    return bottlenecks

def get_api_metrics() -> Dict[str, Any]:
    """
    Get API call metrics.
    
    Returns:
        Dictionary containing API metrics
    """
    return {
        "total_calls": llm_cache.total_api_calls,
        "successful_calls": llm_cache.successful_api_calls,
        "failed_calls": llm_cache.failed_api_calls,
        "average_response_time": llm_cache.average_response_time
    }

def get_memory_usage() -> Dict[str, Any]:
    """
    Get current memory usage.
    
    Returns:
        Dictionary containing memory usage metrics
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        "rss": memory_info.rss,  # Resident Set Size
        "vms": memory_info.vms,  # Virtual Memory Size
        "percent": process.memory_percent()
    }

if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(main())
