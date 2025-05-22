import datetime
import json
from typing import List, Dict, Any, Optional
from autogen import GroupChat

class TraceCollectorAgent:
    """Agent that collects all messages and actions for tracing purposes."""
    def __init__(self, name: str = "trace_collector"):
        self.name = name
        self.collected_actions: List[Dict[str, Any]] = []

    def observe(self, action_type: str, details: Dict[str, Any], agent_name: str):
        self.collected_actions.append({
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "action_type": action_type,
            "agent": agent_name,
            **details
        })

    def get_trace(self) -> List[Dict[str, Any]]:
        return self.collected_actions

    def save_trace(self, trace_path: str):
        trace = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "stats": {
                "total_actions": len(self.collected_actions),
                "agents": list(set(action["agent"] for action in self.collected_actions))
            },
            "actions": self.collected_actions
        }
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2, ensure_ascii=False)

class AutoTracingGroupChat(GroupChat):
    """
    GroupChat that automatically traces all agent actions and messages using a TraceCollectorAgent.
    No manual logging or user intervention required.
    """
    def __init__(self, agents, trace_path: str, *args, **kwargs):
        self.trace_collector = TraceCollectorAgent()
        super().__init__(agents=agents, *args, **kwargs)
        self.trace_path = trace_path

    def send(self, message: str, sender: Optional[str] = None, **kwargs):
        self.trace_collector.observe(
            "message_sent",
            {"sender": sender, "message_length": len(message), **kwargs},
            sender or "Unknown"
        )
        return super().send(message, sender, **kwargs)

    def receive(self, message: str, sender: Optional[str] = None, **kwargs):
        self.trace_collector.observe(
            "message_received",
            {"sender": sender, "message_length": len(message), **kwargs},
            self.name
        )
        return super().receive(message, sender, **kwargs)

    def agent_action(self, action_type: str, details: Dict[str, Any], agent_name: str):
        """Call this from agent methods to log any custom action."""
        self.trace_collector.observe(action_type, details, agent_name)

    def reset(self):
        self.trace_collector.save_trace(self.trace_path)
        super().reset()

    def get_trace(self):
        return self.trace_collector.get_trace()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reset() 