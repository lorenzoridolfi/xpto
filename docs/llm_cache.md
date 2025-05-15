# LLM Cache System

## Overview

The LLM cache system (`src/llm_cache.py`) provides sophisticated caching capabilities to reduce API calls and improve performance.

## Key Features

1. **Exact Match Caching**
   - Uses message hashing for exact matches
   - Maintains cache entry metadata
   - Tracks cache hits and misses

2. **Similarity-Based Caching**
   - Uses TF-IDF and cosine similarity
   - Configurable similarity threshold
   - Multi-language support
   - Handles similar but not identical requests

3. **Cache Management**
   - Automatic expiration of old entries
   - Size-based cleanup
   - Hit-based retention
   - Performance optimization

4. **Metadata Tracking**
   - Cache hit/miss statistics
   - Usage patterns
   - Performance metrics
   - Cost savings

## Usage Example

```python
from src.llm_cache import LLMCache, LLMParams

# Initialize cache
cache = LLMCache(
    max_size=1000,
    similarity_threshold=0.85,
    expiration_hours=24
)

# Set LLM parameters
params = LLMParams(
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
)

# Check cache
response = cache.get(messages, query_text)
if response is None:
    # Make API call
    response = make_llm_call(messages, params)
    # Cache response
    cache.set(messages, response, metadata={"params": params.to_dict()})
```

## Configuration

```python
cache_config = {
    "max_size": 1000,              # Maximum cache entries
    "similarity_threshold": 0.85,   # Similarity threshold
    "expiration_hours": 24,        # Cache entry expiration
    "language": "english"          # Language for tokenization
}
```

## Best Practices

1. **Cache Usage**
   - Set appropriate cache size and expiration
   - Monitor cache hit rates
   - Clean cache periodically
   - Use similarity caching for similar requests

2. **Error Handling**
   - Handle cache misses gracefully
   - Monitor error patterns
   - Log validation errors 