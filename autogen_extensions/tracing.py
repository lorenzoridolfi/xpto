import datetime
import json
from typing import List, Dict, Any, Optional

class TracingMixin:
    """
    Mixin for trace logging. Provides log and save methods for traceability.
    Can be used by any agent, orchestrator, or group chat.
    """
    def __init__(self, trace_path: Optional[str] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_path = trace_path
        self.trace: List[Dict[str, Any]] = []

    def log_event(self, event_type: str, details: Dict[str, Any]):
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "event_type": event_type,
            **details,
        }
        self.trace.append(entry)

    def save_trace(self, path: Optional[str] = None):
        if path is not None:
            self.trace_path = path
        if not self.trace_path:
            raise ValueError("No trace_path specified for saving trace.")
        with open(self.trace_path, "w", encoding="utf-8") as f:
            json.dump(self.trace, f, indent=2, ensure_ascii=False) 