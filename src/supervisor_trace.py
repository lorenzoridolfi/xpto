"""
Supervisor Trace Module

This module provides tracing functionality for supervisor decisions and agent interactions.
It implements a robust tracing system that captures:
- Task execution flow
- Agent creation and interactions
- Supervisor decisions
- Error occurrences
- Performance metrics

The module uses a structured logging approach to maintain detailed traces of system behavior.
"""

from typing import Dict, Any, List, Optional, TypedDict, Union
from datetime import datetime
from autogen import Agent, GroupChat
from .logger import LoggerMixin


# Custom Exceptions
class TraceError(Exception):
    """Base exception for trace-related errors."""

    pass


class TraceConfigurationError(TraceError):
    """Raised when trace configuration is invalid."""

    pass


class TraceOperationError(TraceError):
    """Raised when trace operations fail."""

    pass


# Type Definitions
class TaskDict(TypedDict):
    id: str
    type: str
    status: Optional[str]


class TraceEvent(TypedDict):
    type: str
    timestamp: str
    agent_name: Optional[str]
    agent_type: Optional[str]
    config: Optional[Dict[str, Any]]
    error_type: Optional[str]
    error_message: Optional[str]
    context: Optional[Dict[str, Any]]


class AgentInteraction(TypedDict):
    timestamp: str
    sender: str
    recipient: str
    message: str


class Decision(TypedDict):
    timestamp: str
    type: str
    context: Dict[str, Any]


class Trace(TypedDict):
    task_id: str
    task_type: str
    start_time: str
    end_time: Optional[str]
    status: Optional[str]
    result: Optional[Dict[str, Any]]
    events: List[TraceEvent]
    agent_interactions: List[AgentInteraction]
    decisions: List[Decision]


class TraceMetrics(TypedDict):
    total_traces: int
    active_trace: bool
    total_events: int
    total_interactions: int
    total_decisions: int
    error_count: int


class SupervisorTrace(LoggerMixin):
    """Trace class for tracking supervisor decisions and agent interactions.

    This class provides comprehensive tracing capabilities for monitoring and debugging
    the supervisor system. It tracks task execution, agent interactions, decisions,
    and errors in a structured format.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the supervisor trace.

        Args:
            config: Configuration dictionary containing trace settings

        Raises:
            TraceConfigurationError: If configuration is invalid
        """
        try:
            super().__init__()
            self.config = config
            self.traces: List[Trace] = []
            self.current_trace: Optional[Trace] = None
            self.trace_enabled: bool = config.get("tracing", {}).get("enabled", True)
            self.trace_level: str = config.get("tracing", {}).get("level", "INFO")
        except Exception as e:
            raise TraceConfigurationError(f"Failed to initialize trace: {str(e)}")

    def start_trace(self, task: TaskDict) -> None:
        """Start a new trace for a task.

        Args:
            task: Task dictionary to trace

        Raises:
            TraceOperationError: If trace initialization fails
        """
        if not self.trace_enabled:
            return

        try:
            self.current_trace = {
                "task_id": task.get("id", str(datetime.now().timestamp())),
                "task_type": task.get("type"),
                "start_time": datetime.now().isoformat(),
                "events": [],
                "agent_interactions": [],
                "decisions": [],
            }
            self.traces.append(self.current_trace)

            self.log_info(
                "Started trace",
                task_id=self.current_trace["task_id"],
                task_type=self.current_trace["task_type"],
            )
        except Exception as e:
            raise TraceOperationError(f"Failed to start trace: {str(e)}")

    def end_trace(self, result: Dict[str, Any]) -> None:
        """End the current trace with a result.

        Args:
            result: Task result dictionary

        Raises:
            TraceOperationError: If trace completion fails
        """
        if not self.trace_enabled or not self.current_trace:
            return

        try:
            self.current_trace["end_time"] = datetime.now().isoformat()
            self.current_trace["result"] = result
            self.current_trace["status"] = result.get("status", "unknown")

            self.log_info(
                "Ended trace",
                task_id=self.current_trace["task_id"],
                status=self.current_trace["status"],
            )

            self.current_trace = None
        except Exception as e:
            raise TraceOperationError(f"Failed to end trace: {str(e)}")

    def trace_agent_creation(self, agent: Agent, config: Dict[str, Any]) -> None:
        """Trace agent creation.

        Args:
            agent: Created agent instance
            config: Agent configuration

        Raises:
            TraceOperationError: If agent creation tracing fails
        """
        if not self.trace_enabled or not self.current_trace:
            return

        try:
            event: TraceEvent = {
                "type": "agent_creation",
                "timestamp": datetime.now().isoformat(),
                "agent_name": agent.name,
                "agent_type": getattr(agent, "type", "unknown"),
                "config": config,
                "error_type": None,
                "error_message": None,
                "context": None,
            }

            self.current_trace["events"].append(event)
            self.log_debug("Traced agent creation", agent_name=agent.name)
        except Exception as e:
            raise TraceOperationError(f"Failed to trace agent creation: {str(e)}")

    def trace_agent_interaction(
        self, sender: Agent, recipient: Agent, message: str
    ) -> None:
        """Trace interaction between agents.

        Args:
            sender: Sending agent
            recipient: Receiving agent
            message: Message content

        Raises:
            TraceOperationError: If interaction tracing fails
        """
        if not self.trace_enabled or not self.current_trace:
            return

        try:
            interaction: AgentInteraction = {
                "timestamp": datetime.now().isoformat(),
                "sender": sender.name,
                "recipient": recipient.name,
                "message": message,
            }

            self.current_trace["agent_interactions"].append(interaction)
            self.log_debug(
                "Traced agent interaction", sender=sender.name, recipient=recipient.name
            )
        except Exception as e:
            raise TraceOperationError(f"Failed to trace agent interaction: {str(e)}")

    def trace_decision(self, decision_type: str, context: Dict[str, Any]) -> None:
        """Trace a supervisor decision.

        Args:
            decision_type: Type of decision
            context: Decision context

        Raises:
            TraceOperationError: If decision tracing fails
        """
        if not self.trace_enabled or not self.current_trace:
            return

        try:
            decision: Decision = {
                "timestamp": datetime.now().isoformat(),
                "type": decision_type,
                "context": context,
            }

            self.current_trace["decisions"].append(decision)
            self.log_debug("Traced decision", decision_type=decision_type)
        except Exception as e:
            raise TraceOperationError(f"Failed to trace decision: {str(e)}")

    def trace_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Trace an error occurrence.

        Args:
            error: Exception that occurred
            context: Error context

        Raises:
            TraceOperationError: If error tracing fails
        """
        if not self.trace_enabled or not self.current_trace:
            return

        try:
            error_event: TraceEvent = {
                "type": "error",
                "timestamp": datetime.now().isoformat(),
                "agent_name": None,
                "agent_type": None,
                "config": None,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context,
            }

            self.current_trace["events"].append(error_event)
            self.log_error(
                "Traced error",
                error_type=error_event["error_type"],
                error_message=error_event["error_message"],
            )
        except Exception as e:
            raise TraceOperationError(f"Failed to trace error: {str(e)}")

    def get_trace(self, task_id: str) -> Optional[Trace]:
        """Get trace for a specific task.

        Args:
            task_id: Task identifier

        Returns:
            Trace dictionary or None if not found

        Raises:
            TraceOperationError: If trace retrieval fails
        """
        try:
            for trace in self.traces:
                if trace["task_id"] == task_id:
                    return trace
            return None
        except Exception as e:
            raise TraceOperationError(f"Failed to get trace: {str(e)}")

    def get_all_traces(self) -> List[Trace]:
        """Get all traces.

        Returns:
            List of trace dictionaries

        Raises:
            TraceOperationError: If trace retrieval fails
        """
        try:
            return self.traces
        except Exception as e:
            raise TraceOperationError(f"Failed to get all traces: {str(e)}")

    def clear_traces(self) -> None:
        """Clear all traces.

        Raises:
            TraceOperationError: If trace clearing fails
        """
        try:
            self.traces = []
            self.current_trace = None
            self.log_info("Cleared all traces")
        except Exception as e:
            raise TraceOperationError(f"Failed to clear traces: {str(e)}")

    def get_trace_metrics(self) -> TraceMetrics:
        """Get metrics about traces.

        Returns:
            Dictionary containing trace metrics

        Raises:
            TraceOperationError: If metrics calculation fails
        """
        try:
            return {
                "total_traces": len(self.traces),
                "active_trace": bool(self.current_trace),
                "total_events": sum(len(t["events"]) for t in self.traces),
                "total_interactions": sum(
                    len(t["agent_interactions"]) for t in self.traces
                ),
                "total_decisions": sum(len(t["decisions"]) for t in self.traces),
                "error_count": sum(
                    1 for t in self.traces for e in t["events"] if e["type"] == "error"
                ),
            }
        except Exception as e:
            raise TraceOperationError(f"Failed to get trace metrics: {str(e)}")
