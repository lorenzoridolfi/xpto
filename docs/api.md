# API Documentation

## Overview
The Feedback System API provides endpoints for submitting queries, providing feedback, and retrieving query information. The API is built using FastAPI and follows RESTful principles.

## Base URL
```
http://localhost:8000
```

## Authentication
Currently, the API does not require authentication. Future versions may implement authentication mechanisms.

## Endpoints

### 1. Submit Query
Submit a new query to the system.

```http
POST /query
```

#### Request Body
```json
{
    "query": "string",
    "metadata": {
        "key": "value"
    }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| query | string | Yes | The query to process |
| metadata | object | No | Additional information about the query |

#### Response
```json
{
    "entry_id": "uuid",
    "response": "string"
}
```

| Field | Type | Description |
|-------|------|-------------|
| entry_id | string | UUID of the created entry |
| response | string | System's response to the query |

#### Status Codes
- `200 OK`: Query processed successfully
- `500 Internal Server Error`: Server-side error

#### Example
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{
         "query": "What is the weather?",
         "metadata": {
             "source": "web",
             "user_id": "123"
         }
     }'
```

### 2. Submit Feedback
Provide feedback for a specific query.

```http
POST /feedback/{entry_id}
```

#### Path Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| entry_id | string | UUID of the entry to provide feedback for |

#### Request Body
```json
{
    "feedback": "string"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| feedback | string | Yes | The feedback message |

#### Response
```json
{
    "entry_id": "uuid",
    "status": "success"
}
```

| Field | Type | Description |
|-------|------|-------------|
| entry_id | string | UUID of the entry |
| status | string | Status of the feedback submission |

#### Status Codes
- `200 OK`: Feedback processed successfully
- `404 Not Found`: Entry not found
- `500 Internal Server Error`: Server-side error

#### Example
```bash
curl -X POST "http://localhost:8000/feedback/123e4567-e89b-12d3-a456-426614174000" \
     -H "Content-Type: application/json" \
     -d '{
         "feedback": "The response was helpful"
     }'
```

### 3. Get Query
Retrieve information about a specific query.

```http
GET /query/{entry_id}
```

#### Path Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| entry_id | string | UUID of the entry to retrieve |

#### Response
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

| Field | Type | Description |
|-------|------|-------------|
| id | string | UUID of the entry |
| query | string | Original query |
| response | string | System's response |
| feedback | string | User feedback (if any) |
| timestamp | string | When the entry was created |
| metadata | object | Additional information |

#### Status Codes
- `200 OK`: Entry retrieved successfully
- `404 Not Found`: Entry not found
- `500 Internal Server Error`: Server-side error

#### Example
```bash
curl "http://localhost:8000/query/123e4567-e89b-12d3-a456-426614174000"
```

## Error Responses

All endpoints may return the following error responses:

### 404 Not Found
```json
{
    "detail": "Query not found"
}
```

### 500 Internal Server Error
```json
{
    "detail": "Error message"
}
```

## Rate Limiting
Currently, the API does not implement rate limiting. Future versions may add rate limiting to prevent abuse.

## Best Practices

1. **Error Handling**
   - Always check response status codes
   - Handle errors gracefully
   - Implement retry logic for transient failures

2. **Performance**
   - Cache responses when appropriate
   - Minimize payload size
   - Use compression for large responses

3. **Security**
   - Validate input data
   - Sanitize user input
   - Use HTTPS in production

## Future Enhancements

1. **Authentication**
   - API key authentication
   - OAuth 2.0 support
   - Role-based access control

2. **Rate Limiting**
   - Per-user rate limits
   - IP-based rate limiting
   - Quota management

3. **Monitoring**
   - Request/response logging
   - Performance metrics
   - Error tracking

4. **Caching**
   - Response caching
   - Cache invalidation
   - Cache headers 