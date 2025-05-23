from typing import Protocol, Dict, Optional, Any
from dataclasses import dataclass
from uuid import UUID
from datetime import datetime


@dataclass
class FeedbackEntry:
    """Data class representing a feedback entry."""

    id: UUID
    query: str
    response: str
    feedback: Optional[str]
    timestamp: datetime
    metadata: Dict[str, Any]


class QueryHandler(Protocol):
    """Protocol for handling queries."""

    async def handle_query(self, query: str) -> str:
        """Handle a query and return a response."""
        ...


class FeedbackHandler(Protocol):
    """Protocol for handling feedback."""

    async def handle_feedback(self, entry_id: UUID, feedback: str) -> None:
        """Handle feedback for a specific entry."""
        ...
