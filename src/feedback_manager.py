from typing import Optional, Dict, Any
from uuid import UUID

from .feedback_protocols import QueryHandler, FeedbackHandler
from .feedback_storage import FeedbackStorage

class QueryFeedbackManager:
    """Manager class that combines query handling, feedback handling, and storage."""
    
    def __init__(
        self,
        query_handler: QueryHandler,
        feedback_handler: FeedbackHandler,
        storage: FeedbackStorage
    ):
        self.query_handler = query_handler
        self.feedback_handler = feedback_handler
        self.storage = storage
    
    async def process_query(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> UUID:
        """Process a query and store the result."""
        response = await self.query_handler.handle_query(query)
        entry_id = await self.storage.store(query, response, metadata)
        return entry_id
    
    async def process_feedback(self, entry_id: UUID, feedback: str) -> None:
        """Process feedback for a specific entry."""
        entry = await self.storage.get(entry_id)
        if entry is None:
            raise ValueError(f"No entry found with ID {entry_id}")
        
        await self.feedback_handler.handle_feedback(entry_id, feedback)
        
        # Update the entry with feedback
        entry.feedback = feedback
        await self.storage.store(entry.query, entry.response, entry.metadata) 