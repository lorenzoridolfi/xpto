from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from uuid import UUID

from .feedback_protocols import QueryHandler, FeedbackHandler
from .feedback_storage import InMemoryFeedbackStorage
from .feedback_manager import QueryFeedbackManager

app = FastAPI(title="Query Feedback API")


# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    metadata: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    entry_id: UUID
    response: str


class FeedbackRequest(BaseModel):
    feedback: str


class FeedbackResponse(BaseModel):
    entry_id: UUID
    status: str


# Example implementations of the protocols
class OpenAIQueryHandler(QueryHandler):
    async def handle_query(self, query: str) -> str:
        # This would be your actual OpenAI implementation
        return f"Response to: {query}"


class LoggingFeedbackHandler(FeedbackHandler):
    async def handle_feedback(self, entry_id: UUID, feedback: str) -> None:
        # This would be your actual feedback handling implementation
        print(f"Received feedback for {entry_id}: {feedback}")


# Initialize components
storage = InMemoryFeedbackStorage()
query_handler = OpenAIQueryHandler()
feedback_handler = LoggingFeedbackHandler()
manager = QueryFeedbackManager(query_handler, feedback_handler, storage)


@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest) -> QueryResponse:
    """Handle a new query."""
    try:
        entry_id = await manager.process_query(request.query, request.metadata)
        entry = await storage.get(entry_id)
        if entry is None:
            raise HTTPException(status_code=500, detail="Failed to store query")

        return QueryResponse(entry_id=entry_id, response=entry.response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback/{entry_id}", response_model=FeedbackResponse)
async def handle_feedback(entry_id: UUID, request: FeedbackRequest) -> FeedbackResponse:
    """Handle feedback for a specific query."""
    try:
        await manager.process_feedback(entry_id, request.feedback)
        return FeedbackResponse(entry_id=entry_id, status="success")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query/{entry_id}")
async def get_query(entry_id: UUID):
    """Retrieve a query and its feedback."""
    entry = await storage.get(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Query not found")
    return entry
