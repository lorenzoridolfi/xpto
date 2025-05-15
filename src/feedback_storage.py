from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from uuid import UUID, uuid4

from .feedback_protocols import FeedbackEntry
from .logger import LoggerMixin
from .config import config

class FeedbackStorage(ABC):
    """Abstract base class for feedback storage."""
    
    @abstractmethod
    async def store(self, entry_id: UUID, query: str, response: str, metadata: Optional[Dict[str, Any]] = None) -> UUID:
        """Store a query-response pair and return its ID.
        
        Args:
            entry_id: The UUID to use for the entry
            query: The query string
            response: The response string
            metadata: Optional metadata dictionary
            
        Returns:
            The UUID of the stored entry
        """
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

class InMemoryFeedbackStorage(FeedbackStorage, LoggerMixin):
    """Simple in-memory implementation of FeedbackStorage with logging."""
    
    def __init__(self):
        super().__init__()
        self._storage: Dict[UUID, FeedbackEntry] = {}
        self._config = config.get_storage_config()
        self.log_info("Initialized InMemoryFeedbackStorage", config=self._config)
    
    async def store(self, entry_id: UUID, query: str, response: str, metadata: Optional[Dict[str, Any]] = None) -> UUID:
        """Store a query-response pair in memory.
        
        Args:
            entry_id: The UUID to use for the entry
            query: The query string
            response: The response string
            metadata: Optional metadata dictionary
            
        Returns:
            The UUID of the stored entry
            
        Raises:
            ValueError: If an entry with the given ID already exists
        """
        try:
            if entry_id in self._storage:
                error_msg = f"Entry with ID {entry_id} already exists"
                self.log_error(error_msg)
                raise ValueError(error_msg)
            
            # Check if we need to purge old entries
            if len(self._storage) >= self._config.get("max_entries", 1000):
                await self.purge_by_count(self._config.get("keep_last_n_entries", 100))
                
            entry = FeedbackEntry(
                id=entry_id,
                query=query,
                response=response,
                feedback=None,
                timestamp=datetime.utcnow(),
                metadata=metadata or {}
            )
            self._storage[entry_id] = entry
            
            self.log_info(
                "Stored new entry",
                entry_id=str(entry_id),
                query_length=len(query),
                response_length=len(response),
                has_metadata=bool(metadata)
            )
            return entry_id
            
        except Exception as e:
            self.log_error(
                "Failed to store entry",
                entry_id=str(entry_id),
                error=str(e),
                query_length=len(query),
                response_length=len(response)
            )
            raise
    
    async def get(self, entry_id: UUID) -> Optional[FeedbackEntry]:
        """Retrieve a feedback entry by ID."""
        try:
            entry = self._storage.get(entry_id)
            if entry:
                self.log_info(
                    "Retrieved entry",
                    entry_id=str(entry_id),
                    has_feedback=bool(entry.feedback)
                )
            else:
                self.log_warning(
                    "Entry not found",
                    entry_id=str(entry_id)
                )
            return entry
            
        except Exception as e:
            self.log_error(
                "Failed to retrieve entry",
                entry_id=str(entry_id),
                error=str(e)
            )
            raise
    
    async def purge_by_time(self, older_than: datetime) -> int:
        """Purge entries older than the specified time."""
        try:
            old_entries = [
                entry_id for entry_id, entry in self._storage.items()
                if entry.timestamp < older_than
            ]
            for entry_id in old_entries:
                del self._storage[entry_id]
            
            self.log_info(
                "Purged entries by time",
                older_than=older_than.isoformat(),
                purged_count=len(old_entries)
            )
            return len(old_entries)
            
        except Exception as e:
            self.log_error(
                "Failed to purge entries by time",
                older_than=older_than.isoformat(),
                error=str(e)
            )
            raise
    
    async def purge_by_count(self, keep_last_n: int) -> int:
        """Purge entries keeping only the last N entries."""
        try:
            if len(self._storage) <= keep_last_n:
                return 0
                
            # Sort entries by timestamp and keep only the most recent ones
            sorted_entries = sorted(
                self._storage.items(),
                key=lambda x: x[1].timestamp,
                reverse=True
            )
            
            entries_to_remove = sorted_entries[keep_last_n:]
            for entry_id, _ in entries_to_remove:
                del self._storage[entry_id]
            
            self.log_info(
                "Purged entries by count",
                keep_last_n=keep_last_n,
                purged_count=len(entries_to_remove)
            )
            return len(entries_to_remove)
            
        except Exception as e:
            self.log_error(
                "Failed to purge entries by count",
                keep_last_n=keep_last_n,
                error=str(e)
            )
            raise
    
    def __len__(self) -> int:
        """Get the number of stored entries."""
        return len(self._storage)
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about the storage."""
        stats = {
            "total_entries": len(self._storage),
            "entries_with_feedback": sum(1 for e in self._storage.values() if e.feedback),
            "oldest_entry": min((e.timestamp for e in self._storage.values()), default=None),
            "newest_entry": max((e.timestamp for e in self._storage.values()), default=None),
            "max_entries": self._config.get("max_entries", 1000),
            "purge_older_than_hours": self._config.get("purge_older_than_hours", 24),
            "keep_last_n_entries": self._config.get("keep_last_n_entries", 100)
        }
        self.log_info("Retrieved storage stats", **stats)
        return stats 