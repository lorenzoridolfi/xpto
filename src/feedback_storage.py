from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Optional, Any
from uuid import UUID, uuid4

from .feedback_protocols import FeedbackEntry

class FeedbackStorage(ABC):
    """Abstract base class for feedback storage."""
    
    @abstractmethod
    async def store(self, query: str, response: str, metadata: Optional[Dict[str, Any]] = None) -> UUID:
        """Store a query-response pair and return its ID."""
        ...
    
    @abstractmethod
    async def get(self, entry_id: UUID) -> Optional[FeedbackEntry]:
        """Retrieve a feedback entry by ID."""
        ...
    
    @abstractmethod
    async def purge_by_time(self, older_than: datetime) -> int:
        """Purge entries older than the specified time. Returns number of entries purged."""
        ...
    
    @abstractmethod
    async def purge_by_count(self, keep_last_n: int) -> int:
        """Purge entries keeping only the last N entries. Returns number of entries purged."""
        ...

class InMemoryFeedbackStorage(FeedbackStorage):
    """Simple in-memory implementation of FeedbackStorage."""
    
    def __init__(self):
        self._storage: Dict[UUID, FeedbackEntry] = {}
    
    async def store(self, query: str, response: str, metadata: Optional[Dict[str, Any]] = None) -> UUID:
        """Store a query-response pair in memory."""
        entry_id = uuid4()
        entry = FeedbackEntry(
            id=entry_id,
            query=query,
            response=response,
            feedback=None,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        self._storage[entry_id] = entry
        return entry_id
    
    async def get(self, entry_id: UUID) -> Optional[FeedbackEntry]:
        """Retrieve a feedback entry by ID."""
        return self._storage.get(entry_id)
    
    async def purge_by_time(self, older_than: datetime) -> int:
        """Stub implementation - does nothing."""
        return 0
    
    async def purge_by_count(self, keep_last_n: int) -> int:
        """Stub implementation - does nothing."""
        return 0 