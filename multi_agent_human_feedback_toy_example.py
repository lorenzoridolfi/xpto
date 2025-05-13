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

from autogen_agentchat.agents import BaseChatAgent, AssistantAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import TextMessage, BaseChatMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from json_validator_tool import get_tool_for_agent
from tool_analytics import ToolAnalytics, ToolUsageMetrics
from analytics_assistant_agent import AnalyticsAssistantAgent

# -----------------------------------------------------------------------------
# Global logs
# -----------------------------------------------------------------------------
# Lists to store system-wide logging information
ROOT_CAUSE_DATA: List[dict] = []  # Stores detailed event data for root cause analysis
FILE_LOG: List[str] = []         # Tracks file operations
ACTION_LOG: List[str] = []       # Records agent actions and decisions

# Global logger instance
logger = logging.getLogger("MultiAgentSystem")

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
        response = await super().run(task=prompt)
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

    # Load and validate configuration
    config_file = "multi_agent_human_feedback_toy_example.json"
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {config_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in configuration file: {config_file}")
        return

    # Setup logging
    setup_logging(config)

    # Update agent system messages with subject
    for agent_name, agent_config in config.get("agents", {}).items():
        if "system_message" in agent_config:
            agent_config["system_message"] = agent_config["system_message"].replace(
                "{subject}", subject
            )

    # Write configuration files for runtime use
    with open(config["output_files"]["agent_config"], "w", encoding="utf-8") as fcfg:
        json.dump(config, fcfg, indent=2, ensure_ascii=False)
    with open(config["output_files"]["agents_configuration"], "w", encoding="utf-8") as fcfg2:
        json.dump(config, fcfg2, indent=2, ensure_ascii=False)

    # Extract configuration parameters
    task_description = config.get("task_description", "")
    hierarchy = config.get("hierarchy", [])
    agent_configs = config.get("agents", {})
    file_manifest = config.get("file_manifest", [])
    max_rounds = config.get("max_rounds", 5)
    llm_config = config.get("llm_config", {})

    logger.info(f"Task description: {task_description}")
    logger.info(f"Agent hierarchy: {hierarchy}")

    # Initialize OpenAI client with configuration
    openai_client = OpenAIChatCompletionClient(
        model=llm_config.get("model", "gpt-4"),
        temperature=llm_config.get("temperature", 0.7),
        max_tokens=llm_config.get("max_tokens", 4096),
        top_p=llm_config.get("top_p", 1.0),
        frequency_penalty=llm_config.get("frequency_penalty", 0.0),
        presence_penalty=llm_config.get("presence_penalty", 0.0)
    )

    # Instantiate and configure agents
    agents = {}
    for name, info in agent_configs.items():
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
                reflect_on_tool_use=True  # Enable reflection for root cause analysis
            )
        else:
            # Get appropriate tools for the agent
            tools = [get_tool_for_agent(name)]
            
            agents[name] = AnalyticsAssistantAgent(
                name=name,
                model_client=openai_client,
                system_message=info.get("system_message",""),
                tools=tools,
                reflect_on_tool_use=True  # Enable reflection for all other agents
            )
        logger.info(f"Instantiated agent: {name} with analytics and tool reflection enabled")

    # Get references to key agents
    coordinator = agents.get("CoordinatorAgent")
    file_reader = agents.get("FileReaderAgent")
    writer = agents.get("WriterAgent")
    info_verifier = agents.get("InformationVerifierAgent")
    text_quality = agents.get("TextQualityAgent")
    root_cause = agents.get("RootCauseAnalyzerAgent")

    # Initialize state variables
    accumulated = []
    final_summary = ""
    info_reply = ""
    quality_reply = ""

    # Run iterative improvement rounds
    for round_idx in range(1, max_rounds+1):
        logger.info(f"=== Round {round_idx} ===")
        
        # Get file reading instructions from coordinator
        already = ", ".join(file_reader.files_read) or "none"
        coord_res = await coordinator.run(task=f"Already read: {already}")
        instr = coord_res.messages[-1].content.strip()
        ACTION_LOG.append(f"CoordinatorAgent: {instr}")
        logger.info(f"CoordinatorAgent instructs: {instr}")
        
        # Break if no more files to process
        if instr.upper() == "NO_FILE": break
        
        # Read and process files
        fr_res = await file_reader.run(messages=[TextMessage(content=instr, source="CoordinatorAgent")])
        payload = fr_res.messages[-1].content
        ACTION_LOG.append("FileReaderAgent: batch read")
        if payload == "NO_FILE": break
        accumulated.append(payload)
        
        # Generate and verify content
        w_res = await writer.run(task="\n\n".join(accumulated))
        summary = w_res.messages[-1].content
        final_summary = summary
        ACTION_LOG.append(f"WriterAgent: {summary}")
        logger.info("WriterAgent produced summary")
        
        # Verify information and quality
        iv_res = await info_verifier.run(task=summary)
        info_reply = iv_res.messages[-1].content.strip()
        ACTION_LOG.append(f"InformationVerifierAgent: {info_reply}")
        logger.info(f"InformationVerifierAgent: {info_reply}")
        
        tq_res = await text_quality.run(task=summary)
        quality_reply = tq_res.messages[-1].content.strip()
        ACTION_LOG.append(f"TextQualityAgent: {quality_reply}")
        logger.info(f"TextQualityAgent: {quality_reply}")
        
        # Break if both verifiers approve
        if info_reply == "TERMINATE" and quality_reply == "TERMINATE": break

        # Collect validation summaries at the end of each round
        validation_summaries = {}
        for name, agent in agents.items():
            if isinstance(agent, AnalyticsAssistantAgent):
                validation_summaries[name] = agent.get_analytics_summary()
        
        # Log validation summaries
        logger.info(f"Round {round_idx} validation summaries:")
        for name, summary in validation_summaries.items():
            logger.info(f"{name}: {json.dumps(summary, indent=2)}")

    # Collect and process user feedback
    print("\n=== Final Summary ===\n")
    print(final_summary)
    user_fb = input("\nPlease provide your feedback:\n> ").strip()
    ACTION_LOG.append(f"User: {user_fb}")
    logger.info(f"User feedback: {user_fb}")

    # Collect tool analytics data from all agents
    tool_analytics_data = {}
    for name, agent in agents.items():
        if isinstance(agent, AnalyticsAssistantAgent):
            tool_analytics_data[name] = {
                "analytics_summary": agent.get_analytics_summary(),
                "performance_metrics": agent.performance_metrics,
                "interaction_history": agent.interaction_history
            }

    # Prepare comprehensive analysis data
    analysis_data = {
        "configuration": config,
        "user_feedback": user_fb,
        "event_logs": ACTION_LOG,
        "tool_analytics": tool_analytics_data,
        "root_cause_data": ROOT_CAUSE_DATA,
        "file_operations": FILE_LOG
    }

    # Perform root cause analysis with enhanced data
    rc_prompt = f"""
    Analyze the following comprehensive system data and provide detailed insights:

    1. Configuration and Setup:
    {json.dumps(config, indent=2)}

    2. User Feedback:
    {user_fb}

    3. Tool Usage Analytics:
    {json.dumps(tool_analytics_data, indent=2)}

    4. Event Logs and Actions:
    {json.dumps(ACTION_LOG, indent=2)}

    5. File Operations:
    {json.dumps(FILE_LOG, indent=2)}

    Please analyze:
    1. Tool Usage Effectiveness:
       - Success rates and patterns
       - Optimization opportunities
       - Tool combinations and sequences
       - Context awareness and timing

    2. System Performance:
       - Response times and efficiency
       - Error patterns and prevention
       - Resource utilization
       - Bottlenecks and improvements

    3. User Experience:
       - Feedback alignment with tool usage
       - Quality of outputs
       - System responsiveness
       - Areas for improvement

    4. Recommendations:
       - Tool usage optimization
       - System improvements
       - Process enhancements
       - Future considerations

    Provide a detailed analysis focusing on actionable improvements and specific recommendations.
    """

    rc_res = await root_cause.run(task=rc_prompt)
    rc_output = rc_res.messages[-1].content
    
    # Save comprehensive analysis results
    analysis_results = {
        "analysis": rc_output,
        "events": ROOT_CAUSE_DATA,
        "tool_analytics": tool_analytics_data,
        "user_feedback": user_fb,
        "event_logs": ACTION_LOG,
        "file_operations": FILE_LOG,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    }

    with open(config["output_files"]["root_cause"], "w", encoding="utf-8") as jf:
        json.dump(analysis_results, jf, indent=2, ensure_ascii=False)
    logger.info("Wrote detailed root_cause.json with comprehensive analysis")

    print("\n--- ROOT CAUSE ANALYSIS SUMMARY ---")
    print(rc_output)

    # Print tool usage insights
    print("\n--- TOOL USAGE INSIGHTS ---")
    for agent_name, data in tool_analytics_data.items():
        if data["performance_metrics"]:
            metrics = data["performance_metrics"]
            print(f"\n{agent_name} Tool Usage:")
            print(f"Total Tools Used: {metrics['total_tools_used']}")
            print(f"Total Calls: {metrics['total_calls']}")
            print(f"Average Success Rate: {metrics['average_success_rate']:.2%}")
            
            if metrics["optimization_suggestions"]:
                print("\nOptimization Suggestions:")
                for suggestion in metrics["optimization_suggestions"]:
                    print(f"- {suggestion['tool']}: {suggestion['suggestion']}")
                    print(f"  Reason: {suggestion.get('reason', 'N/A')}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Fatal error: {e}")
