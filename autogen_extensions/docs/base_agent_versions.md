# BaseAgent: Legacy vs. Cache-Enabled Versions

## Overview

This document explains the two versions of the `BaseAgent` class available in the codebase:
- The **legacy version** (no LLM cache support)
- The **cache-enabled version** (with LLM cache dependency injection)

This approach allows for backward compatibility while enabling new features and optimizations for LLM-driven workflows.

---

## 1. Legacy BaseAgent (No LLM Cache)

- **Location:** `autogen_extensions/base_agent.py`
- **LLM Caching:** Not supported
- **Constructor:** No `llm_cache` parameter
- **LLM Calls:** Always go directly to the LLM provider

**Example Usage:**
```python
from autogen_extensions.base_agent import BaseAgent

agent = BaseAgent(...)
response = agent.llm_call("What is the capital of France?")
```

---

## 2. Cache-Enabled BaseAgent (Dependency Injection)

- **Location:** `autogen_extensions/base_agent_cache.py`
- **LLM Caching:** Supported via optional `llm_cache` parameter
- **Constructor:** `BaseAgent(llm_cache=None, ...)`
- **LLM Calls:**
    - If `llm_cache` is provided, checks the cache before calling the LLM
    - Stores new results in the cache
    - If `llm_cache` is not provided, behaves like the legacy version

**Example Usage (with cache):**
```python
from autogen_extensions.base_agent_cache import BaseAgent
from autogen_extensions.llm_cache import AutogenLLMCache

llm_cache = AutogenLLMCache(...)
agent = BaseAgent(llm_cache=llm_cache, ...)
response = agent.llm_call("What is the capital of France?")
```

**Example Usage (without cache):**
```python
from autogen_extensions.base_agent_cache import BaseAgent
agent = BaseAgent(...)
response = agent.llm_call("What is the capital of France?")
```

---

## Summary Table

| Feature                | Legacy BaseAgent         | Cache-Enabled BaseAgent      |
|------------------------|-------------------------|-----------------------------|
| LLM cache support      | No                      | Yes (optional)              |
| Backward compatible    | Yes                     | Yes                         |
| Import path            | base_agent.py           | base_agent_cache.py         |
| Migration required     | No                      | Only if you want caching    |
| Recommended for new?   | No                      | Yes                         |

---

## Migration Path

- **Existing code** can continue using the legacy version with no changes.
- **New code** or codebases that want LLM caching should import from `base_agent_cache.py` and (optionally) provide an `llm_cache` instance.
- **Gradual migration** is possible: update imports and add the `llm_cache` parameter where needed.

---

## Recommendations

- Use the **cache-enabled version** for all new development and for any agents where LLM call efficiency, cost, or repeatability is important.
- Use the **legacy version** only for legacy code that cannot be updated or where caching is not needed.
- Document which version is in use in your project for clarity.

--- 