# Incremental Implementation Plan: Idea Brainstorm with Synthetic Users

## Overview
This document outlines a stepwise, modular approach to building an idea-brainstorming system using synthetic users and agent-based architecture. Each step builds on the previous, allowing for incremental development, testing, and extension.

---

## Architecture: Dependency Injection for Execution Strategy and LLM Cache

A key feature of this architecture is the use of **dependency injection** for both the execution strategy (sequential, thread, process, async) and the LLM cache. These dependencies are passed into each layer of the system, enabling:
- **Modularity:** Swap execution modes or cache implementations without changing business logic.
- **Testability:** Inject mocks or simple strategies for unit/integration tests.
- **Flexibility:** Easily scale or adapt to new requirements (e.g., distributed execution, new cache backends).

### How Dependencies Flow

- The **LLM cache** is injected into all agent classes that make LLM calls.
- The **execution strategy** is injected into orchestrator classes (user questioning, batch, brainstormer) to control parallelism at each level.
- Both can be configured at the top level and passed down recursively.

**Diagram:**
```
+-------------------+
|  LLM Cache        |
+-------------------+
         |
         v
+-------------------+
| Execution Strategy|
+-------------------+
         |
         v
+-------------------------------+
| IdeaBrainstormer               |
|  (exec strategy, llm cache)    |
+-------------------------------+
         |
         v
+-------------------------------+
| BatchQuestionOrchestrator      |
|  (exec strategy, llm cache)    |
+-------------------------------+
         |
         v
+-------------------------------+
| UserQuestionSummarizer         |
|  (exec strategy, llm cache)    |
+-------------------------------+
         |
         v
+-------------------+
|   Agents          |
|  (llm cache)      |
+-------------------+
```

---

## Result Collection and Communication in the Spawn-Join Pattern

By default, the **SpawnJoinOrchestrator** and its execution strategies return a list of results after all tasks complete. This is sufficient for most agent-based, LLM, or batch-processing workflows:
- ThreadPool, ProcessPool, and asyncio.gather all return results as a list (order matches submission).
- The orchestrator summarizes or aggregates these results in a single step.

**Advanced Coordination (Optional):**
- If you need to stream results, handle partial failures, or coordinate between processes (e.g., for distributed or stateful workflows), you can extend the execution strategy interface to support callbacks, streaming, or event hooks.
- Only add this complexity if you have a real use case (e.g., very large result sets, need for early aggregation, or distributed execution).

**Recommendation:**
- Keep the default pattern (list of results) for simplicity and testability.
- Extend only if/when you need more advanced result handling or communication.

---

## Step 1: User Questioning and Summarization Class

**Goal:**
Create a class that, given a list of users and a question, asks the question to each user (via an agent) and summarizes the responses.

**Key Features:**
- Accepts:
  - List of user objects (loaded from JSON)
  - A question (string)
  - (Optional) agent configuration
  - **LLM cache (injected)**
  - **Execution strategy (injected)**
- For each user:
  - Uses an agent to generate the user's answer to the question
  - Can run these calls in parallel (e.g., ThreadPoolExecutor via strategy)
- Collects all answers
- Uses a summarization agent to produce a summary of the answers

// ... rest of the document unchanged ... 