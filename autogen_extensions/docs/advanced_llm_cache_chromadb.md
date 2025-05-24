# Advanced LLM Cache with ChromaDB: Per-Agent Collections and Custom Hashing

## Overview

This document outlines best practices for implementing an advanced LLM cache using ChromaDB, with a focus on:
- Using a **separate collection (table) per agent type**
- Providing a **custom hash formation function** for each agent
- Ensuring robust, precise, and efficient caching of LLM calls

---

## 1. Rationale: Why Separate Collections and Custom Hashing?

- **Isolation:** Each agent type (e.g., user generator, validator, reviewer, summarizer) may have different prompt structures and cache key logic. Separate collections prevent key collisions and make cache management easier.
- **Customizability:** Each agent can define what constitutes its "fixed" and "variable" prompt parts, and how to hash them for cache keys.
- **Maintainability:** Per-agent collections simplify cache invalidation, inspection, and debugging.

---

## 2. Implementation Pattern

### Per-Agent Collection
- Name each ChromaDB collection after the agent type and program context (e.g., `user_generator_cache`, `reviewer_cache`).
- When initializing the cache for an agent, pass the collection name as a parameter.

### Custom Hash Formation Function
- Each agent (or its cache wrapper) provides a method for generating the combined hash key.
- Use robust text/JSON/dict hashing utilities for consistency.

**Example:**
```python
from autogen_extensions.text_hashing import hash_dict, hash_json, hash_text

class UserGeneratorAgent:
    def get_cache_key(self, segment, schema, system_message):
        # Hash each fixed part, then combine
        return '|'.join([
            hash_dict(segment),
            hash_json(schema),
            hash_text(system_message)
        ])
```

### ChromaDB Integration
- **Initialization:** Each agent or cache wrapper creates/uses its own ChromaDB collection.
- **Cache Lookup:**
    - Use the agent's hash function to generate the key for the fixed parts.
    - Use vector search for the variable part (e.g., user input, context).
    - Require both a high similarity score and an exact hash match for a cache hit.
- **Cache Store:**
    - Store the embedding, hash key, and response in the agent's collection.

---

## 3. Example Table: Agent Types and Cache Keys

| Agent Type         | ChromaDB Collection      | Hash Function Needed? | Example Fixed Parts to Hash         |
|--------------------|-------------------------|-----------------------|-------------------------------------|
| User Generator     | user_generator_cache    | Yes                   | segment, schema, system_message     |
| Validator          | validator_cache         | Yes                   | schema, system_message              |
| Reviewer           | reviewer_cache          | Yes                   | schema, system_message              |
| Summarizer         | summarizer_cache        | Yes                   | summary_prompt, config              |
| ...                | ...                     | ...                   | ...                                 |

---

## 4. Extensibility and Testing

- **Extensibility:**
    - Add new agent types by creating new collections and hash functions as needed.
    - Consider a base cache wrapper class with overridable hash/key methods for DRYness.
- **Testing:**
    - Write unit tests for hash formation to ensure cache consistency and avoid collisions.
    - Test cache hit/miss logic for each agent type.

---

## 5. Recommendations

- **Document the hash function and cache key logic for each agent type.**
- **Use robust, collision-resistant hashing utilities for all fixed/variable parts.**
- **Keep collections isolated per agent type for maintainability and clarity.**
- **Write tests for hash formation and cache integration.**
- **Extend the pattern as new agent types or workflows are added.**

---

## 6. Example: Base Cache Wrapper Class (Pattern)

```python
class BaseAgentLLMCache:
    def __init__(self, collection_name, embedding_model, chroma_client):
        self.collection = chroma_client.get_or_create_collection(collection_name)
        self.embedding_model = embedding_model

    def get_cache_key(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement get_cache_key")

    def lookup(self, variable_prompt, *fixed_parts):
        key = self.get_cache_key(*fixed_parts)
        embedding = self.embedding_model.embed(variable_prompt)
        # ... perform ChromaDB query and match on key ...

    def store(self, variable_prompt, response, *fixed_parts):
        key = self.get_cache_key(*fixed_parts)
        embedding = self.embedding_model.embed(variable_prompt)
        # ... store in ChromaDB ...
```

--- 