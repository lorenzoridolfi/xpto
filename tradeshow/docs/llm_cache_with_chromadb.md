# Intelligent LLM Cache System for Autogen Agents with ChromaDB

## Overview

This document describes the architecture and implementation of a robust, intelligent cache system for Autogen agents, leveraging ChromaDB as a vector database with local file storage. The system is designed to minimize LLM API calls by caching responses based on both the fixed and variable components of prompts, ensuring precise and repeatable results.

---

## Architectural Principles

### 1. Separation of Prompt Components
- **Fixed Parts:** Agent descriptions, system messages, configuration texts—these rarely change and define the agent's identity and behavior.
- **Variable Parts:** User inputs, contextual data, and dynamic content—these change per request.

### 2. Hybrid Cache Keying
- **Combined Hash Key:** At startup, each fixed component is individually hashed (e.g., SHA256), and the hashes are concatenated to form a single string. This key uniquely identifies the agent's configuration and system context.
- **Vector Embedding:** The variable part of the prompt is embedded using a configurable embedding model (e.g., OpenAI, SentenceTransformers).
- **Cache Hit Criteria:**
    - High similarity score from ChromaDB vector search (for variable part)
    - Exact match of the combined hash key (for fixed part)

### 3. False Positive Avoidance
- By requiring both a vector similarity match and an exact hash key match, the system avoids returning stale or incorrect responses when agent configuration changes, even if the user input is similar.

---

## ChromaDB Collection Structure

- **Collection Name:** `autogen_llm_cache`
- **Document Fields:**
    - `id`: Unique identifier (e.g., UUID)
    - `embedding`: Vector representation of the variable prompt part
    - `combined_hash_key`: String (concatenated hash of fixed parts)
    - `response`: Cached LLM response
    - `metadata`: Dict (e.g., timestamps, agent name, config version)
- **Indexing:**
    - Vector index on `embedding`
    - Optional secondary index on `combined_hash_key`

---

## Embedding Model Configuration

- **Model:** Configurable (e.g., OpenAI, HuggingFace, local models)
- **Dimension:** Must match ChromaDB collection
- **Batching:** Supported for efficient cache population

---

## Cache Wrapper Class Design

### Class: `AutogenLLMCache`

#### Initialization
- Accepts:
    - List of fixed prompt components (strings)
    - Embedding model instance
    - ChromaDB client/collection
    - Similarity threshold (e.g., 0.90)
- Computes and stores the combined hash key at startup
- Connects to or creates the ChromaDB collection

#### Methods
- `lookup(variable_prompt: str) -> Optional[str]`
    - Embeds the variable prompt
    - Queries ChromaDB for top-N similar embeddings
    - For each candidate, checks if `combined_hash_key` matches
    - Returns cached response if both criteria are met
- `store(variable_prompt: str, response: str)`
    - Embeds the variable prompt
    - Stores embedding, combined hash key, and response in ChromaDB
- `invalidate_on_config_change(new_fixed_parts: List[str])`
    - Recomputes combined hash key
    - Optionally purges or marks old cache entries
- `wrap_agent(agent)`
    - Intercepts agent LLM calls, applies cache lookup/store logic
- `monitor_performance()`
    - Tracks hit/miss rates, average lookup time, and logs statistics

#### Error Handling
- Handles ChromaDB connection errors, embedding failures, and corrupted cache entries gracefully
- Falls back to direct LLM call on error, logs incident

---

## Integration with Autogen Flow

- The cache wrapper is injected into the agent's LLM call pipeline
- On each LLM call:
    1. The fixed parts are already hashed (from startup)
    2. The variable part is embedded and looked up in the cache
    3. On cache hit: return cached response
    4. On miss: call LLM, store result in cache

---

## Example Implementation

```python
import hashlib
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Optional

class AutogenLLMCache:
    def __init__(self, fixed_parts: List[str], embedding_model, similarity_threshold: float = 0.90):
        self.fixed_parts = fixed_parts
        self.combined_hash_key = self._compute_combined_hash(fixed_parts)
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection("autogen_llm_cache")

    def _compute_combined_hash(self, parts: List[str]) -> str:
        hashes = [hashlib.sha256(p.encode()).hexdigest() for p in parts]
        return "|".join(hashes)

    def lookup(self, variable_prompt: str) -> Optional[str]:
        embedding = self.embedding_model.embed(variable_prompt)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=5,
            include=["metadatas", "documents", "distances"]
        )
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            if meta["combined_hash_key"] == self.combined_hash_key and dist >= self.similarity_threshold:
                return doc  # Cached response
        return None

    def store(self, variable_prompt: str, response: str):
        embedding = self.embedding_model.embed(variable_prompt)
        self.collection.add(
            embeddings=[embedding],
            documents=[response],
            metadatas=[{"combined_hash_key": self.combined_hash_key}]
        )

    def invalidate_on_config_change(self, new_fixed_parts: List[str]):
        self.combined_hash_key = self._compute_combined_hash(new_fixed_parts)
        # Optionally: self.collection.delete(where={"combined_hash_key": {"$ne": self.combined_hash_key}})

    def wrap_agent(self, agent):
        orig_llm_call = agent.llm_call
        def cached_llm_call(variable_prompt, *args, **kwargs):
            cached = self.lookup(variable_prompt)
            if cached:
                return cached
            result = orig_llm_call(variable_prompt, *args, **kwargs)
            self.store(variable_prompt, result)
            return result
        agent.llm_call = cached_llm_call

    def monitor_performance(self):
        # Implement hit/miss tracking, timing, and logging
        pass
```

---

## Usage Example

```python
# At application startup
fixed_parts = [agent_description, system_message, config_text]
embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
llm_cache = AutogenLLMCache(fixed_parts, embedding_model)

# Wrap an Autogen agent
llm_cache.wrap_agent(my_autogen_agent)

# On agent LLM call
response = my_autogen_agent.llm_call(user_input)

# On configuration change
llm_cache.invalidate_on_config_change(new_fixed_parts)
```

---

## Monitoring and Performance
- Track cache hit/miss rates and average lookup/store times
- Log cache statistics for analysis and tuning
- Use ChromaDB's built-in tools for collection inspection and maintenance

---

## Benefits of the Hybrid Approach
- **Precision:** Avoids false positives by requiring both vector similarity and exact config match
- **Repeatability:** Ensures responses are tied to the exact agent configuration
- **Efficiency:** Minimizes redundant LLM API calls, reducing cost and latency
- **Robustness:** Handles configuration changes and cache invalidation cleanly

---

## Conclusion

This architecture provides a robust, precise, and efficient LLM cache for Autogen agents, leveraging ChromaDB's vector search and strict configuration keying. It is extensible to other agent frameworks and can be tuned for different workloads and embedding models. 