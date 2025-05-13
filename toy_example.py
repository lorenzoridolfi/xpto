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
from typing import List, Dict, Any
import jsonschema
from jsonschema import validate
import sys

from autogen_agentchat.agents import BaseChatAgent, AssistantAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import TextMessage, BaseChatMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from json_validator_tool import get_tool_for_agent
from tool_analytics import ToolAnalytics, ToolUsageMetrics
from analytics_assistant_agent import AnalyticsAssistantAgent
from llm_cache import LLMCache
import autogen

# -----------------------------------------------------------------------------
# Global logs and cache
# -----------------------------------------------------------------------------
# Lists to store system-wide logging information
ROOT_CAUSE_DATA: List[dict] = []  # Stores detailed event data for root cause analysis
FILE_LOG: List[str] = []         # Tracks file operations
ACTION_LOG: List[str] = []       # Records agent actions and decisions

# Global logger instance
logger = logging.getLogger("MultiAgentSystem")

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
    global logger
    logger.setLevel(getattr(logging, config["logging"]["level"]))
    fmt = logging.Formatter(config["logging"]["format"])
    ch = logging.StreamHandler(); ch.setFormatter(fmt)
    fh = logging.FileHandler(config["logging"]["file"], mode="w"); fh.setFormatter(fmt)
    logger.addHandler(ch); logger.addHandler(fh)

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

    async def on_messages(self, messages: List[BaseChatMessage], cancellation_token) -> Response:
        """
        Analyze system behavior and user feedback.

        Args:
            messages (List[BaseChatMessage]): List of messages containing user feedback
            cancellation_token: Token for cancellation support

        Returns:
            Response: Analysis of system behavior and recommendations
        """
        log_event(self.name, "on_messages_invoke", messages, [])
        
        # Combine configuration and user feedback for analysis
        with open("agents_configuration.json", encoding="utf-8") as cfgf:
            config_content = cfgf.read()
        prompt = f"Configuration:\n{config_content}\n\nUser feedback and logs:\n{messages[-1].content}"
        
        # Try to get response from cache
        cached_response = llm_cache.get(
            messages=[{"role": "system", "content": prompt}],
            query_text=prompt
        )
        
        if cached_response:
            logger.info(f"Cache hit for {self.name}")
            response = cached_response
        else:
            response = await super().run(task=prompt)
            # Cache the response
            llm_cache.put(
                messages=[{"role": "system", "content": prompt}],
                response=response,
                metadata={
                    "agent": self.name,
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
                }
            )
        
        log_event(self.name, "on_messages_complete", messages, response)
        return response

# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------
async def main():
    """
    Main function that orchestrates the multi-agent system workflow.
    
    The workflow consists of:
    1. Getting subject input from user
    2. Loading and validating configuration
    3. Initializing agents
    4. Running iterative content generation and improvement rounds
    5. Collecting and analyzing user feedback
    """
    # Get subject from user
    print("\nPlease enter the subject of the text to be processed:")
    subject = input("> ").strip()
    while not subject:
        print("Subject cannot be empty. Please try again:")
        subject = input("> ").strip()

    # Load configuration
    with open("toy_example.json", "r") as f:
        config = json.load(f)

    # Extract configuration values
    TASK_DESCRIPTION = config["task_description"]
    HIERARCHY = config["hierarchy"]
    AGENT_CONFIGS = config["agents"]
    OUTPUT_FILES = config["output_files"]
    LOGGING_CONFIG = config["logging"]
    LLM_CONFIG = config["llm_config"]
    # Use these directly from config
    file_manifest = config["file_manifest"]
    max_rounds = config["max_rounds"]

    # Configure logging based on config
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG["level"]),
        format=LOGGING_CONFIG["format"],
        handlers=[
            logging.FileHandler(LOGGING_CONFIG["file"]),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Update agent system messages with subject
    for agent_name, agent_config in AGENT_CONFIGS.items():
        if "system_message" in agent_config:
            agent_config["system_message"] = agent_config["system_message"].replace(
                "{subject}", subject
            )

    # Write configuration files for runtime use
    with open(config["output_files"]["agent_config"], "w", encoding="utf-8") as fcfg:
        json.dump(config, fcfg, indent=2, ensure_ascii=False)
    with open(config["output_files"]["agents_configuration"], "w", encoding="utf-8") as fcfg2:
        json.dump(config, fcfg2, indent=2, ensure_ascii=False)

    # Initialize OpenAI client
    openai_client = autogen.OpenAIWrapper(
        model=LLM_CONFIG["model"],
        temperature=LLM_CONFIG["temperature"],
        max_tokens=LLM_CONFIG["max_tokens"],
        top_p=LLM_CONFIG["top_p"],
        frequency_penalty=LLM_CONFIG["frequency_penalty"],
        presence_penalty=LLM_CONFIG["presence_penalty"]
    )

    # Initialize LLM cache with parameters from config
    llm_cache = LLMCache(
        max_size=config["cache_config"]["max_size"],
        similarity_threshold=config["cache_config"]["similarity_threshold"],
        expiration_hours=config["cache_config"]["expiration_hours"],
        llm_params=LLM_CONFIG  # Pass LLM parameters to cache
    )

    # Create file manifest from config
    file_manifest = [
        {"filename": item["filename"], "description": item["description"]}
        for item in file_manifest
    ]

    # Create file log
    FILE_LOG = "file_operations.log"

    def get_tool_for_agent(agent_name: str) -> Any:
        """Get the appropriate tool for a given agent."""
        if agent_name == "WriterAgent":
            return autogen.Tool(
                name="write_content",
                description="Write content based on input",
                function=lambda x: f"Generated content: {x}"
            )
        elif agent_name == "InformationVerifierAgent":
            return autogen.Tool(
                name="verify_information",
                description="Verify information accuracy",
                function=lambda x: f"Verified content: {x}"
            )
        elif agent_name == "TextQualityAgent":
            return autogen.Tool(
                name="check_quality",
                description="Check text quality",
                function=lambda x: f"Quality checked content: {x}"
            )
        return None

    # Instantiate and configure agents
    agents = {}
    for name, info in AGENT_CONFIGS.items():
        if name == "FileReaderAgent":
            agents[name] = FileReaderAgent(
                name=name,
                description=info.get("description",""),
                manifest=file_manifest,
                file_log=FILE_LOG
            )
        elif name == "RootCauseAnalyzerAgent":
            agents[name] = RootCauseAnalyzerAgent(
                name=name,
                model_client=openai_client,
                system_message=info.get("system_message",""),
                reflect_on_tool_use=True,  # Enable reflection for root cause analysis
                cache=llm_cache  # Enable caching
            )
        else:
            # Get appropriate tools for the agent
            tools = [get_tool_for_agent(name)]
            
            agents[name] = AnalyticsAssistantAgent(
                name=name,
                model_client=openai_client,
                system_message=info.get("system_message",""),
                tools=tools,
                reflect_on_tool_use=True,  # Enable reflection for all other agents
                cache=llm_cache  # Enable caching
            )
        logger.info(f"Instantiated agent: {name} with analytics, tool reflection, and caching enabled")

    # Get references to key agents
    coordinator = agents["CoordinatorAgent"]
    file_reader = agents["FileReaderAgent"]
    writer = agents["WriterAgent"]
    verifier = agents["InformationVerifierAgent"]
    quality_checker = agents["TextQualityAgent"]
    root_cause_analyzer = agents["RootCauseAnalyzerAgent"]

    # Create a group chat
    groupchat = GroupChat(
        agents=[coordinator, file_reader, writer, verifier, quality_checker],
        messages=[],
        max_round=max_rounds
    )

    # Start the conversation
    coordinator.initiate_chat(
        groupchat,
        message=f"Let's work on the task: {TASK_DESCRIPTION}"
    )

    # Analyze the conversation
    root_cause_analyzer.analyze_conversation(groupchat.messages)

    # Print cache statistics
    print("\n--- CACHE STATISTICS ---")
    cache_stats = llm_cache.get_stats()
    print(f"Cache Size: {cache_stats['size']}/{cache_stats['max_size']}")
    print(f"Oldest Entry: {cache_stats['oldest_entry']}")
    print(f"Newest Entry: {cache_stats['newest_entry']}")
    if "llm_params" in cache_stats:
        print("\nLLM Parameters:")
        for param, value in cache_stats["llm_params"].items():
            print(f"  {param}: {value}")

    # Print per-agent cache statistics
    print("\n--- PER-AGENT CACHE STATISTICS ---")
    for name, agent in agents.items():
        if isinstance(agent, AnalyticsAssistantAgent):
            metrics = agent.get_performance_metrics()
            print(f"\n{name}:")
            print(f"Cache Hit Rate: {metrics.cache_hit_rate:.2%}")
            print(f"Exact Match Hits: {metrics.exact_match_hits}")
            print(f"Similarity Hits: {metrics.similarity_hits}")
            print(f"Cache Misses: {metrics.cache_misses}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Fatal error: {e}")
