from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Optional, Any
from uuid import UUID, uuid4
import logging
from logging.handlers import RotatingFileHandler
import os

from .feedback_protocols import FeedbackEntry

# Configure logging
logger = logging.getLogger("feedback_storage")
logger.setLevel(logging.DEBUG)

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Add file handler
file_handler = RotatingFileHandler(
    "logs/storage.log",
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
logger.addHandler(file_handler)

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
logger.addHandler(console_handler)

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

class InMemoryFeedbackStorage(FeedbackStorage):
    """Simple in-memory implementation of FeedbackStorage with logging."""
    
    def __init__(self):
        self._storage: Dict[UUID, FeedbackEntry] = {}
        logger.info("Initialized InMemoryFeedbackStorage")
    
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
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            entry = FeedbackEntry(
                id=entry_id,
                query=query,
                response=response,
                feedback=None,
                timestamp=datetime.utcnow(),
                metadata=metadata or {}
            )
            self._storage[entry_id] = entry
            
            logger.info(
                "Stored new entry",
                extra={
                    "entry_id": str(entry_id),
                    "query_length": len(query),
                    "response_length": len(response),
                    "has_metadata": bool(metadata)
                }
            )
            return entry_id
            
        except Exception as e:
            logger.error(
                "Failed to store entry",
                extra={
                    "entry_id": str(entry_id),
                    "error": str(e),
                    "query_length": len(query),
                    "response_length": len(response)
                },
                exc_info=True
            )
            raise
    
    async def get(self, entry_id: UUID) -> Optional[FeedbackEntry]:
        """Retrieve a feedback entry by ID."""
        try:
            entry = self._storage.get(entry_id)
            if entry:
                logger.info(
                    "Retrieved entry",
                    extra={
                        "entry_id": str(entry_id),
                        "has_feedback": bool(entry.feedback)
                    }
                )
            else:
                logger.warning(
                    "Entry not found",
                    extra={"entry_id": str(entry_id)}
                )
            return entry
            
        except Exception as e:
            logger.error(
                "Failed to retrieve entry",
                extra={"entry_id": str(entry_id), "error": str(e)},
                exc_info=True
            )
            raise
    
    async def purge_by_time(self, older_than: datetime) -> int:
        """Stub implementation - does nothing."""
        logger.info(
            "Purge by time called (stub implementation)",
            extra={"older_than": older_than.isoformat()}
        )
        return 0
    
    async def purge_by_count(self, keep_last_n: int) -> int:
        """Stub implementation - does nothing."""
        logger.info(
            "Purge by count called (stub implementation)",
            extra={"keep_last_n": keep_last_n}
        )
        return 0
    
    def __len__(self) -> int:
        """Get the number of stored entries."""
        return len(self._storage)
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about the storage."""
        stats = {
            "total_entries": len(self._storage),
            "entries_with_feedback": sum(1 for e in self._storage.values() if e.feedback),
            "oldest_entry": min((e.timestamp for e in self._storage.values()), default=None),
            "newest_entry": max((e.timestamp for e in self._storage.values()), default=None)
        }
        logger.info("Retrieved storage stats", extra=stats)
        return stats 