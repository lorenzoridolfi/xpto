# Feedback System Documentation

## Overview
The feedback system is designed to handle queries, store responses, and manage user feedback in a modular and extensible way. The system is built using protocols, abstract classes, and concrete implementations to ensure flexibility and maintainability.

## Architecture

### Core Components

#### 1. Protocols (`feedback_protocols.py`)
The system defines two main protocols:

- **QueryHandler**
  ```python
  class QueryHandler(Protocol):
      async def handle_query(self, query: str) -> str:
          """Handle a query and return a response."""
  ```
  - Responsible for processing user queries
  - Returns a response string
  - Can be implemented for different query processing strategies (e.g., OpenAI, custom logic)

- **FeedbackHandler**
  ```python
  class FeedbackHandler(Protocol):
      async def handle_feedback(self, entry_id: UUID, feedback: str) -> None:
          """Handle feedback for a specific entry."""
  ```
  - Processes user feedback for specific entries
  - Can be implemented for different feedback handling strategies (e.g., logging, analytics)

#### 2. Storage (`feedback_storage.py`)
The storage system is built around the `FeedbackStorage` abstract base class:

```python
class FeedbackStorage(ABC):
    async def store(self, query: str, response: str, metadata: Optional[Dict[str, Any]] = None) -> UUID
    async def get(self, entry_id: UUID) -> Optional[FeedbackEntry]
    async def purge_by_time(self, older_than: datetime) -> int
    async def purge_by_count(self, keep_last_n: int) -> int
```

Features:
- Store query-response pairs with metadata
- Retrieve entries by ID
- Purge old entries by time or count
- Extensible for different storage backends

Current implementations:
- `InMemoryFeedbackStorage`: Simple in-memory storage with stub purge functions

#### 3. Manager (`feedback_manager.py`)
The `QueryFeedbackManager` class orchestrates the interaction between components:

```python
class QueryFeedbackManager:
    def __init__(
        self,
        query_handler: QueryHandler,
        feedback_handler: FeedbackHandler,
        storage: FeedbackStorage
    )
```

Features:
- Processes queries and stores responses
- Handles feedback for specific entries
- Manages the interaction between handlers and storage
- Provides a clean interface for the API layer

## Data Model

### FeedbackEntry
```python
@dataclass
class FeedbackEntry:
    id: UUID
    query: str
    response: str
    feedback: Optional[str]
    timestamp: datetime
    metadata: Dict[str, Any]
```

Fields:
- `id`: Unique identifier for the entry
- `query`: Original user query
- `response`: System response to the query
- `feedback`: Optional user feedback
- `timestamp`: When the entry was created
- `metadata`: Additional information about the entry

## API Endpoints

### 1. Submit Query
```http
POST /query
Content-Type: application/json

{
    "query": "string",
    "metadata": {
        "key": "value"
    }
}
```

Response:
```json
{
    "entry_id": "uuid",
    "response": "string"
}
```

### 2. Submit Feedback
```http
POST /feedback/{entry_id}
Content-Type: application/json

{
    "feedback": "string"
}
```

Response:
```json
{
    "entry_id": "uuid",
    "status": "success"
}
```

### 3. Get Query
```http
GET /query/{entry_id}
```

Response:
```json
{
    "id": "uuid",
    "query": "string",
    "response": "string",
    "feedback": "string",
    "timestamp": "datetime",
    "metadata": {
        "key": "value"
    }
}
```

## Error Handling

The system handles various error cases:

1. **Query Processing**
   - Invalid query format
   - Query processing failures
   - Storage errors

2. **Feedback Processing**
   - Invalid entry ID
   - Feedback processing failures
   - Storage errors

3. **Storage Operations**
   - Entry not found
   - Storage system errors
   - Purge operation failures

## Extending the System

### Adding New Query Handlers
1. Implement the `QueryHandler` protocol
2. Add your custom query processing logic
3. Use the new handler in the `QueryFeedbackManager`

### Adding New Feedback Handlers
1. Implement the `FeedbackHandler` protocol
2. Add your custom feedback processing logic
3. Use the new handler in the `QueryFeedbackManager`

### Adding New Storage Backends
1. Extend the `FeedbackStorage` abstract base class
2. Implement all required methods
3. Add your storage-specific logic
4. Use the new storage in the `QueryFeedbackManager`

## Best Practices

1. **Error Handling**
   - Always handle exceptions appropriately
   - Provide meaningful error messages
   - Log errors for debugging

2. **Storage**
   - Implement proper cleanup strategies
   - Handle storage failures gracefully
   - Consider data persistence requirements

3. **Performance**
   - Use async/await for I/O operations
   - Implement proper caching strategies
   - Monitor storage usage

4. **Security**
   - Validate input data
   - Sanitize user feedback
   - Implement proper access control

## Example Usage

```python
# Initialize components
storage = InMemoryFeedbackStorage()
query_handler = OpenAIQueryHandler()
feedback_handler = LoggingFeedbackHandler()
manager = QueryFeedbackManager(query_handler, feedback_handler, storage)

# Process a query
entry_id = await manager.process_query("What is the weather?", {"source": "web"})

# Process feedback
await manager.process_feedback(entry_id, "The response was helpful")
``` 