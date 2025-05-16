"""
Feedback Storage Module

This module provides abstract and concrete implementations for storing and managing
feedback entries in a multi-agent system. It includes:
- Abstract base class for feedback storage
- In-memory implementation with automatic purging
- Statistics and monitoring capabilities
- Error handling and logging

The storage system supports both time-based and count-based purging strategies
to manage memory usage and maintain performance.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List, TypedDict, Union
from uuid import UUID, uuid4

from .feedback_protocols import FeedbackEntry
from .logger import LoggerMixin
from .config import config

# Custom Exceptions
class StorageError(Exception):
    """Base exception for storage-related errors."""
    pass

class StorageConfigurationError(StorageError):
    """Raised when storage configuration is invalid."""
    pass

class StorageOperationError(StorageError):
    """Raised when storage operations fail."""
    pass

class DuplicateEntryError(StorageError):
    """Raised when attempting to store a duplicate entry."""
    pass

# Type Definitions
class StorageConfig(TypedDict):
    max_entries: int
    keep_last_n_entries: int
    purge_older_than_hours: int

class StorageStats(TypedDict):
    total_entries: int
    entries_with_feedback: int
    oldest_entry: Optional[datetime]
    newest_entry: Optional[datetime]
    max_entries: int
    purge_older_than_hours: int
    keep_last_n_entries: int

class FeedbackStorage(ABC):
    """Abstract base class for feedback storage.
    
    This class defines the interface for storing and retrieving feedback entries
    in a multi-agent system. Implementations must provide methods for storing,
    retrieving, and purging entries.
    """
    
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
            
        Raises:
            DuplicateEntryError: If an entry with the given ID already exists
            StorageOperationError: If storage operation fails
        """
        ...
    
    @abstractmethod
    async def get(self, entry_id: UUID) -> Optional[FeedbackEntry]:
        """Retrieve a feedback entry by ID.
        
        Args:
            entry_id: The UUID of the entry to retrieve
            
        Returns:
            The feedback entry if found, None otherwise
            
        Raises:
            StorageOperationError: If retrieval operation fails
        """
        ...
    
    @abstractmethod
    async def purge_by_time(self, older_than: datetime) -> int:
        """Purge entries older than the specified time.
        
        Args:
            older_than: The cutoff datetime for purging
            
        Returns:
            Number of entries purged
            
        Raises:
            StorageOperationError: If purge operation fails
        """
        ...
    
    @abstractmethod
    async def purge_by_count(self, keep_last_n: int) -> int:
        """Purge entries keeping only the last N entries.
        
        Args:
            keep_last_n: Number of most recent entries to keep
            
        Returns:
            Number of entries purged
            
        Raises:
            StorageOperationError: If purge operation fails
        """
        ...

class InMemoryFeedbackStorage(FeedbackStorage, LoggerMixin):
    """Simple in-memory implementation of FeedbackStorage with logging.
    
    This class provides an in-memory storage solution for feedback entries with
    automatic purging based on both time and count thresholds. It includes
    comprehensive logging and error handling.
    """
    
    def __init__(self) -> None:
        """Initialize the in-memory storage.
        
        Raises:
            StorageConfigurationError: If configuration is invalid
        """
        try:
            super().__init__()
            self._storage: Dict[UUID, FeedbackEntry] = {}
            self._config = config.get_storage_config()
            self._validate_config()
            self.log_info("Initialized InMemoryFeedbackStorage", config=self._config)
        except Exception as e:
            raise StorageConfigurationError(f"Failed to initialize storage: {str(e)}")
    
    def _validate_config(self) -> None:
        """Validate storage configuration.
        
        Raises:
            StorageConfigurationError: If configuration is invalid
        """
        required_fields = ["max_entries", "keep_last_n_entries", "purge_older_than_hours"]
        for field in required_fields:
            if field not in self._config:
                raise StorageConfigurationError(f"Missing required config field: {field}")
    
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
            DuplicateEntryError: If an entry with the given ID already exists
            StorageOperationError: If storage operation fails
        """
        try:
            if entry_id in self._storage:
                error_msg = f"Entry with ID {entry_id} already exists"
                self.log_error(error_msg)
                raise DuplicateEntryError(error_msg)
            
            # Check if we need to purge old entries
            if len(self._storage) >= self._config["max_entries"]:
                await self.purge_by_count(self._config["keep_last_n_entries"])
                
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
            
        except DuplicateEntryError:
            raise
        except Exception as e:
            self.log_error(
                "Failed to store entry",
                entry_id=str(entry_id),
                error=str(e),
                query_length=len(query),
                response_length=len(response)
            )
            raise StorageOperationError(f"Failed to store entry: {str(e)}")
    
    async def get(self, entry_id: UUID) -> Optional[FeedbackEntry]:
        """Retrieve a feedback entry by ID.
        
        Args:
            entry_id: The UUID of the entry to retrieve
            
        Returns:
            The feedback entry if found, None otherwise
            
        Raises:
            StorageOperationError: If retrieval operation fails
        """
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
            raise StorageOperationError(f"Failed to retrieve entry: {str(e)}")
    
    async def purge_by_time(self, older_than: datetime) -> int:
        """Purge entries older than the specified time.
        
        Args:
            older_than: The cutoff datetime for purging
            
        Returns:
            Number of entries purged
            
        Raises:
            StorageOperationError: If purge operation fails
        """
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
            raise StorageOperationError(f"Failed to purge entries by time: {str(e)}")
    
    async def purge_by_count(self, keep_last_n: int) -> int:
        """Purge entries keeping only the last N entries.
        
        Args:
            keep_last_n: Number of most recent entries to keep
            
        Returns:
            Number of entries purged
            
        Raises:
            StorageOperationError: If purge operation fails
        """
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
            raise StorageOperationError(f"Failed to purge entries by count: {str(e)}")
    
    def __len__(self) -> int:
        """Get the number of stored entries.
        
        Returns:
            Number of entries in storage
        """
        return len(self._storage)
    
    def get_storage_stats(self) -> StorageStats:
        """Get statistics about the storage.
        
        Returns:
            Dictionary containing storage statistics
            
        Raises:
            StorageOperationError: If stats retrieval fails
        """
        try:
            stats: StorageStats = {
                "total_entries": len(self._storage),
                "entries_with_feedback": sum(1 for e in self._storage.values() if e.feedback),
                "oldest_entry": min((e.timestamp for e in self._storage.values()), default=None),
                "newest_entry": max((e.timestamp for e in self._storage.values()), default=None),
                "max_entries": self._config["max_entries"],
                "purge_older_than_hours": self._config["purge_older_than_hours"],
                "keep_last_n_entries": self._config["keep_last_n_entries"]
            }
            self.log_info("Retrieved storage stats", **stats)
            return stats
        except Exception as e:
            self.log_error("Failed to get storage stats", error=str(e))
            raise StorageOperationError(f"Failed to get storage stats: {str(e)}") 