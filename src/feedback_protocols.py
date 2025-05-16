"""
Feedback Protocols Module

This module defines the core protocols and data structures for the feedback system.
It includes:
- Feedback entry data structure
- Query handler protocol
- Feedback handler protocol

These protocols define the interfaces that components must implement to participate
in the feedback system, ensuring consistent behavior across different implementations.
"""

from typing import Protocol, Dict, Optional, Any, TypedDict, runtime_checkable
from dataclasses import dataclass
from uuid import UUID
from datetime import datetime

# Custom Exceptions
class FeedbackError(Exception):
    """Base exception for feedback-related errors."""
    pass

class QueryError(FeedbackError):
    """Raised when query handling fails."""
    pass

class FeedbackProcessingError(FeedbackError):
    """Raised when feedback processing fails."""
    pass

# Type Definitions
class FeedbackMetadata(TypedDict, total=False):
    """Type definition for feedback metadata."""
    user_id: Optional[str]
    session_id: Optional[str]
    context: Optional[Dict[str, Any]]
    priority: Optional[int]
    tags: Optional[list[str]]

@dataclass
class FeedbackEntry:
    """Data class representing a feedback entry.
    
    This class encapsulates all information related to a single feedback entry,
    including the original query, response, feedback, and associated metadata.
    
    Attributes:
        id: Unique identifier for the entry
        query: Original query string
        response: Response to the query
        feedback: Optional feedback string
        timestamp: When the entry was created
        metadata: Additional metadata about the entry
    """
    id: UUID
    query: str
    response: str
    feedback: Optional[str]
    timestamp: datetime
    metadata: Dict[str, Any]

@runtime_checkable
class QueryHandler(Protocol):
    """Protocol for handling queries.
    
    This protocol defines the interface that query handlers must implement.
    Query handlers are responsible for processing user queries and generating
    appropriate responses.
    """
    
    async def handle_query(self, query: str) -> str:
        """Handle a query and return a response.
        
        Args:
            query: The query string to process
            
        Returns:
            Response string
            
        Raises:
            QueryError: If query processing fails
        """
        ...

@runtime_checkable
class FeedbackHandler(Protocol):
    """Protocol for handling feedback.
    
    This protocol defines the interface that feedback handlers must implement.
    Feedback handlers are responsible for processing user feedback and updating
    the system accordingly.
    """
    
    async def handle_feedback(self, entry_id: UUID, feedback: str) -> None:
        """Handle feedback for a specific entry.
        
        Args:
            entry_id: UUID of the entry to process feedback for
            feedback: The feedback string
            
        Raises:
            FeedbackProcessingError: If feedback processing fails
        """
        ... 