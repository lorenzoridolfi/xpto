import datetime
import json
from typing import List, Dict, Any, Optional
from autogen import GroupChat

class TracedGroupChat(GroupChat):
    """A GroupChat that automatically traces all interactions."""
    
    def __init__(self, *args, trace_path: str, **kwargs):
        """Initialize the traced group chat.
        
        Args:
            trace_path: Path where to save the trace file
            *args: Arguments to pass to GroupChat
            **kwargs: Keyword arguments to pass to GroupChat
        """
        super().__init__(*args, **kwargs)
        self.trace_path = trace_path
        self.action_trace: List[Dict[str, Any]] = []
        
    def _log_action(self, action_type: str, details: Dict[str, Any], agent_name: str) -> None:
        """Log an action to the trace with timestamp and agent information."""
        self.action_trace.append({
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "action_type": action_type,
            "agent": agent_name,
            **details
        })
        
    def _save_trace(self) -> None:
        """Save the current trace to the trace file."""
        trace = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "stats": {
                "total_actions": len(self.action_trace),
                "agents": list(set(action["agent"] for action in self.action_trace))
            },
            "actions": self.action_trace
        }
        with open(self.trace_path, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2, ensure_ascii=False)
            
    def reset(self) -> None:
        """Reset the group chat and save the current trace."""
        self._save_trace()
        super().reset()
        
    def select_speaker(self, *args, **kwargs):
        """Override select_speaker to trace speaker selection."""
        speaker = super().select_speaker(*args, **kwargs)
        if speaker:
            self._log_action(
                "speaker_selected",
                {"selected_speaker": speaker.name},
                "GroupChat"
            )
        return speaker
        
    def send(self, message: str, sender: Optional[str] = None, **kwargs):
        """Override send to trace message sending."""
        self._log_action(
            "message_sent",
            {
                "sender": sender,
                "message_length": len(message),
                **kwargs
            },
            sender or "Unknown"
        )
        return super().send(message, sender, **kwargs)
        
    def receive(self, message: str, sender: Optional[str] = None, **kwargs):
        """Override receive to trace message receiving."""
        self._log_action(
            "message_received",
            {
                "sender": sender,
                "message_length": len(message),
                **kwargs
            },
            self.name
        )
        return super().receive(message, sender, **kwargs) 