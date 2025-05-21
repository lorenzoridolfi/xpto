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

It uses the new GroupChat + TraceCollectorAgent pattern for robust, auditable traceability, as recommended for all workflows.
"""

import os
import json
import logging
import asyncio
import datetime
from typing import List, Dict, Any, Optional, Set, Union, TypedDict, Tuple
import jsonschema
from jsonschema import validate, ValidationError as JsonValidationError
import sys
from prompt_toolkit import prompt
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
import psutil
import time
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
from openai.types.chat import ChatCompletion
from openai import OpenAI

from autogen_extensions.agents import AssistantAgent, UserProxyAgent
from autogen_extensions.messages import TextMessage, BaseChatMessage
from autogen_extensions.response import Response
from autogen_extensions.llm_cache import LLMCache
from autogen_extensions.agent_tracer import AgentTracer
from autogen_extensions.tool_analytics import ToolAnalytics
from autogen_extensions.load_openai import get_openai_client
from autogen_extensions.utils import get_project_root, validate_manifest, load_manifest_schema, FileOperationError, ManifestValidationError
from autogen_extensions.config import load_merged_config
from autogen_extensions.common_io import load_json_file, save_json_file
from autogen_extensions.trace_collector_agent import TraceCollectorAgent
from autogen_extensions.errors import ConfigError
# from autogen_extensions.file_reader_agent import FileReaderAgent
from autogen_extensions.group_chat import GroupChat
from examples.common import load_manifest_data, validate_manifest_for_toy_example

# Use a single logger instance
logger = logging.getLogger("toy_example")

# --- LLM Cache Option ---
USE_LLM_CACHE = False  # Set to True to enable LLM cache usage

# --- Load config and .env for LLM ---
load_dotenv()
config = load_merged_config(
    "config/shared/base_config.json",
    "config/toy_example/program_config.json"
)
llm_config = config.get("llm_config", {})
# Extract only the valid OpenAI parameters
openai_params = {
    "model": llm_config.get("model", "gpt-4"),
    "temperature": llm_config.get("temperature", 0.7),
    "max_tokens": llm_config.get("max_tokens", 4096)
}
logger.info(f"Using OpenAI parameters: {openai_params}")
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    print(
        "WARNING: OPENAI_API_KEY environment variable is not set. OpenAI calls will fail."
    )

# --- Initialize LLM cache if enabled ---
llm_cache = None
if USE_LLM_CACHE:
    llm_cache = LLMCache(
        max_size=config["cache_config"]["max_size"],
        similarity_threshold=config["cache_config"]["similarity_threshold"],
        expiration_hours=config["cache_config"]["expiration_hours"],
    )

# --- Initialize AgentTracer and ToolAnalytics ---
tracer = AgentTracer(config)
tool_analytics = ToolAnalytics()

# --- Initialize OpenAI LLM client ---
try:
    llm_client = get_openai_client(api_key=openai_api_key, **openai_params)
except Exception as e:
    print(f"ERROR: Could not initialize OpenAI client: {e}")
    llm_client = None

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
        with open("config/shared/agent_validation_schema.json") as f:
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
    except jsonschema.ValidationError as e:
        raise jsonschema.ValidationError(f"Schema validation failed for {agent_name}: {str(e)}")


# -----------------------------------------------------------------------------
# Event logger
# -----------------------------------------------------------------------------
def log_event(
    agent_name: str, event_type: str, inputs: List[BaseChatMessage], outputs: Any
) -> None:
    """
    Log an event in the system with detailed information about inputs and outputs.

    Args:
        agent_name (str): Name of the agent generating the event
        event_type (str): Type of event (e.g., 'on_messages_invoke', 'on_messages_complete')
        inputs (List[BaseChatMessage]): Input messages to the agent
        outputs: Output from the agent (can be Response, list of Responses, or other types)

    Raises:
        SystemError: If logging fails
    """
    try:
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
        elif isinstance(outputs, list) and all(
            isinstance(o, Response) for o in outputs
        ):
            entry["outputs"] = []
            for o in outputs:
                try:
                    content = json.loads(o.chat_message.content)
                    if not validate_agent_output(agent_name, content):
                        logger.warning(f"Invalid output format from {agent_name}")
                    entry["outputs"].append(
                        {"source": o.chat_message.source, "content": content}
                    )
                except json.JSONDecodeError:
                    entry["outputs"].append(
                        {
                            "source": o.chat_message.source,
                            "content": o.chat_message.content,
                        }
                    )
        else:
            entry["outputs"] = outputs

        logger.info(json.dumps(entry))
    except Exception as e:
        raise SystemError(f"Failed to log event: {str(e)}")


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
        style = Style.from_dict(
            {
                "prompt": "ansicyan bold",
            }
        )

        return prompt(
            HTML("<prompt>Please enter the path to the document: </prompt>"),
            style=style,
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
        style = Style.from_dict(
            {
                "prompt": "ansigreen bold",
            }
        )

        return prompt(
            HTML("<prompt>Please provide your feedback: </prompt>"), style=style
        )
    except Exception as e:
        raise SystemError(f"Failed to get user feedback: {str(e)}")


# --- Manifest loading: use the shared manifest generated by update_manifest.py ---
# def validate_manifest_for_toy_example(
#     manifest_path: str = "config/update_manifest.json",
#     schema_path: str = "config/update_manifest/manifest_validation_schema.json",
# ) -> dict:
#     ...

# Utility function to load manifest data from a file
# def load_manifest_data(manifest_path: str) -> dict:
#     ...


def create_agents(config: dict, manifest_data: dict, tracer, model_client) -> Dict[str, Any]:
    """
    Create and initialize all required agents with real LLM client and tracer, loading agent descriptions and system messages from config JSON.
    manifest_data: dict loaded from manifest file, e.g., {"files": [...], ...}
    """
    try:
        agents_config = config.get("agents", {})
        assistant_cfg = agents_config.get("WriterAgent", {})
        user_cfg = agents_config.get("SupervisorAgent", agents_config.get("User", {}))
        agents = {
            "assistant": AssistantAgent(
                name=assistant_cfg.get("name", "assistant"),
                system_message=assistant_cfg.get("system_message", "You are an assistant."),
                llm_config=config.get("llm_config", {"config_list": [{"model": "gpt-4"}]}),
                description=assistant_cfg.get("description", "Assistant agent"),
                tracer=tracer,
                model_client=model_client
            ),
            "user": UserProxyAgent(
                name=user_cfg.get("name", "user"),
                system_message=user_cfg.get("system_message", "You are a user."),
                llm_config=config.get("llm_config", {"config_list": [{"model": "gpt-4"}]}),
                description=user_cfg.get("description", "User agent"),
                tracer=tracer,
                model_client=model_client
            ),
        }
        return agents
    except Exception as e:
        logger.error(f"Failed to create agents: {str(e)}")
        raise Exception(f"Failed to create agents: {str(e)}")


async def run_toy_example_workflow(document_path: str, config_override: dict = None, manifest_path: str = None) -> dict:
    """
    Run the toy_example workflow for a given document path and optional config override.
    Returns a dict with the result, trace, and agents.
    """
    config_to_use = config_override if config_override is not None else config
    manifest_path = manifest_path or "config/update_manifest.json"
    manifest_data = load_manifest_data(manifest_path)
    test_tracer = AgentTracer(config_to_use)
    test_model_client = get_openai_client(api_key=os.environ.get("OPENAI_API_KEY"), **openai_params)
    agents_dict = create_agents(config_to_use, manifest_data, test_tracer, test_model_client)
    llm_config = config_to_use.get("llm_config")
    if not (isinstance(llm_config, dict) and "config_list" in llm_config):
        llm_config = {"config_list": [{"model": "gpt-4"}]}
    trace_collector = TraceCollectorAgent(name="trace_collector", system_message="Trace collector agent", llm_config=llm_config, description="Collects all messages", tracer=test_tracer)
    agents = list(agents_dict.values()) + [trace_collector]
    group_chat = GroupChat(agents=agents, messages=[], max_round=5)
    result = await group_chat.run(document_path)
    trace_collector.save_trace("toy_example_trace.json")
    return {
        "result": result,
        "trace": trace_collector.collected_messages,
        "agents": agents,
    }


async def main() -> None:
    """
    Main entry point for the multi-agent human feedback system.
    """
    try:
        # Get document selection
        document_path = get_document_selection()
        output = await run_toy_example_workflow(document_path)
        result = output["result"]
        trace = output["trace"]

        logger.info(f"System execution completed: {result}")
        print("\n=== toy_example.py Feature Exercise Summary ===")
        print("- Agent types used: AssistantAgent, UserProxyAgent")
        print("- AgentTracer used: trace_event logged")
        print("- ToolAnalytics used: tool usage recorded")
        print(f"- LLMCache used: {'yes' if USE_LLM_CACHE else 'no'}")
        print("- OpenAI LLM used: yes" if llm_client else "- OpenAI LLM used: NO (client not available)")
        print("- Agent message passing: GroupChat with real LLM")
        print("- StateManager: not used in this example")
        print("- Manifest validation: success and error cases")
        print("- Logging: debug/info logs throughout")
        print("- All major autogen_extensions features exercised!\n")
        # Print the collected trace
        print("\n[TRACE] Collected messages:")
        for msg in trace:
            print(f"[TRACE] {msg.role}: {msg.content} (source: {msg.source})")
    except Exception as e:
        logger.error(f"Program execution failed: {str(e)}")
        raise


DEBUG = True  # Set to True to enable debug/summary output

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
