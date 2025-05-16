from typing import Dict, Any, List, Optional
from datetime import datetime
from autogen import Agent, GroupChat
from .logger import LoggerMixin

class SupervisorTrace(LoggerMixin):
    """Trace class for tracking supervisor decisions and agent interactions."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the supervisor trace.
        
        Args:
            config: Configuration dictionary containing trace settings
        """
        super().__init__()
        self.config = config
        self.traces = []
        self.current_trace = None
        self.trace_enabled = config.get("tracing", {}).get("enabled", True)
        self.trace_level = config.get("tracing", {}).get("level", "INFO")
        
    def start_trace(self, task: Dict[str, Any]) -> None:
        """Start a new trace for a task.
        
        Args:
            task: Task dictionary to trace
        """
        if not self.trace_enabled:
            return
            
        self.current_trace = {
            "task_id": task.get("id", str(datetime.now().timestamp())),
            "task_type": task.get("type"),
            "start_time": datetime.now().isoformat(),
            "events": [],
            "agent_interactions": [],
            "decisions": []
        }
        self.traces.append(self.current_trace)
        
        self.log_info(
            "Started trace",
            task_id=self.current_trace["task_id"],
            task_type=self.current_trace["task_type"]
        )
        
    def end_trace(self, result: Dict[str, Any]) -> None:
        """End the current trace with a result.
        
        Args:
            result: Task result dictionary
        """
        if not self.trace_enabled or not self.current_trace:
            return
            
        self.current_trace["end_time"] = datetime.now().isoformat()
        self.current_trace["result"] = result
        self.current_trace["status"] = result.get("status", "unknown")
        
        self.log_info(
            "Ended trace",
            task_id=self.current_trace["task_id"],
            status=self.current_trace["status"]
        )
        
        self.current_trace = None
        
    def trace_agent_creation(self, agent: Agent, config: Dict[str, Any]) -> None:
        """Trace agent creation.
        
        Args:
            agent: Created agent instance
            config: Agent configuration
        """
        if not self.trace_enabled or not self.current_trace:
            return
            
        event = {
            "type": "agent_creation",
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent.name,
            "agent_type": getattr(agent, "type", "unknown"),
            "config": config
        }
        
        self.current_trace["events"].append(event)
        self.log_debug("Traced agent creation", agent_name=agent.name)
        
    def trace_agent_interaction(self, sender: Agent, recipient: Agent, message: str) -> None:
        """Trace interaction between agents.
        
        Args:
            sender: Sending agent
            recipient: Receiving agent
            message: Message content
        """
        if not self.trace_enabled or not self.current_trace:
            return
            
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "sender": sender.name,
            "recipient": recipient.name,
            "message": message
        }
        
        self.current_trace["agent_interactions"].append(interaction)
        self.log_debug(
            "Traced agent interaction",
            sender=sender.name,
            recipient=recipient.name
        )
        
    def trace_decision(self, decision_type: str, context: Dict[str, Any]) -> None:
        """Trace a supervisor decision.
        
        Args:
            decision_type: Type of decision
            context: Decision context
        """
        if not self.trace_enabled or not self.current_trace:
            return
            
        decision = {
            "timestamp": datetime.now().isoformat(),
            "type": decision_type,
            "context": context
        }
        
        self.current_trace["decisions"].append(decision)
        self.log_debug("Traced decision", decision_type=decision_type)
        
    def trace_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """Trace an error occurrence.
        
        Args:
            error: Exception that occurred
            context: Error context
        """
        if not self.trace_enabled or not self.current_trace:
            return
            
        error_event = {
            "type": "error",
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context
        }
        
        self.current_trace["events"].append(error_event)
        self.log_error(
            "Traced error",
            error_type=error_event["error_type"],
            error_message=error_event["error_message"]
        )
        
    def get_trace(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get trace for a specific task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Trace dictionary or None if not found
        """
        for trace in self.traces:
            if trace["task_id"] == task_id:
                return trace
        return None
        
    def get_all_traces(self) -> List[Dict[str, Any]]:
        """Get all traces.
        
        Returns:
            List of trace dictionaries
        """
        return self.traces
        
    def clear_traces(self) -> None:
        """Clear all traces."""
        self.traces = []
        self.current_trace = None
        self.log_info("Cleared all traces")
        
    def get_trace_metrics(self) -> Dict[str, Any]:
        """Get metrics about traces.
        
        Returns:
            Dictionary containing trace metrics
        """
        return {
            "total_traces": len(self.traces),
            "active_trace": bool(self.current_trace),
            "total_events": sum(len(t["events"]) for t in self.traces),
            "total_interactions": sum(len(t["agent_interactions"]) for t in self.traces),
            "total_decisions": sum(len(t["decisions"]) for t in self.traces),
            "error_count": sum(1 for t in self.traces for e in t["events"] if e["type"] == "error")
        } 