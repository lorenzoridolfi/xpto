"""
Feedback Manager Module

This module provides a manager class that coordinates query processing, feedback handling,
and storage operations. It combines the functionality of query handlers, feedback handlers,
and storage systems to provide a unified interface for managing user queries and feedback.

The manager ensures proper coordination between different components and maintains
data consistency across the system.
"""

from typing import Optional, Dict, Any, TypedDict
from uuid import UUID

from .feedback_protocols import QueryHandler, FeedbackHandler
from .feedback_storage import FeedbackStorage, StorageError

# Custom Exceptions
class FeedbackManagerError(Exception):
    """Base exception for feedback manager-related errors."""
    pass

class QueryProcessingError(FeedbackManagerError):
    """Raised when query processing fails."""
    pass

class FeedbackProcessingError(FeedbackManagerError):
    """Raised when feedback processing fails."""
    pass

class EntryNotFoundError(FeedbackManagerError):
    """Raised when a requested entry is not found."""
    pass

# Type Definitions
class QueryMetadata(TypedDict, total=False):
    """Type definition for query metadata."""
    user_id: Optional[str]
    timestamp: Optional[str]
    context: Optional[Dict[str, Any]]
    priority: Optional[int]

class QueryFeedbackManager:
    """Manager class that combines query handling, feedback handling, and storage.
    
    This class coordinates the interaction between query handlers, feedback handlers,
    and storage systems. It ensures proper processing of queries and feedback,
    maintaining data consistency and providing error handling.
    """
    
    def __init__(
        self,
        query_handler: QueryHandler,
        feedback_handler: FeedbackHandler,
        storage: FeedbackStorage
    ) -> None:
        """Initialize the feedback manager.
        
        Args:
            query_handler: Handler for processing queries
            feedback_handler: Handler for processing feedback
            storage: Storage system for queries and feedback
            
        Raises:
            FeedbackManagerError: If initialization fails
        """
        try:
            self.query_handler = query_handler
            self.feedback_handler = feedback_handler
            self.storage = storage
        except Exception as e:
            raise FeedbackManagerError(f"Failed to initialize feedback manager: {str(e)}")
    
    async def process_query(self, query: str, metadata: Optional[QueryMetadata] = None) -> UUID:
        """Process a query and store the result.
        
        Args:
            query: The query string to process
            metadata: Optional metadata about the query
            
        Returns:
            UUID of the stored entry
            
        Raises:
            QueryProcessingError: If query processing fails
            StorageError: If storage operation fails
        """
        try:
            response = await self.query_handler.handle_query(query)
            entry_id = await self.storage.store(query, response, metadata)
            return entry_id
        except Exception as e:
            raise QueryProcessingError(f"Failed to process query: {str(e)}")
    
    async def process_feedback(self, entry_id: UUID, feedback: str) -> None:
        """Process feedback for a specific entry.
        
        Args:
            entry_id: UUID of the entry to process feedback for
            feedback: The feedback string
            
        Raises:
            EntryNotFoundError: If the entry is not found
            FeedbackProcessingError: If feedback processing fails
            StorageError: If storage operation fails
        """
        try:
            entry = await self.storage.get(entry_id)
            if entry is None:
                raise EntryNotFoundError(f"No entry found with ID {entry_id}")
            
            await self.feedback_handler.handle_feedback(entry_id, feedback)
            
            # Update the entry with feedback
            entry.feedback = feedback
            await self.storage.store(entry.query, entry.response, entry.metadata)
        except EntryNotFoundError:
            raise
        except Exception as e:
            raise FeedbackProcessingError(f"Failed to process feedback: {str(e)}") 