# Execution Strategy Pattern for Agent Orchestration

## Overview

This document describes how to use the **Strategy Pattern** with dependency injection to control how agent tasks (such as user questioning or idea-to-questions workflows) are executed: sequentially, in threads, in processes, or asynchronously. This approach enables flexible, testable, and efficient orchestration of LLM or agent calls.

---

## 1. Strategy Pattern: Interface and Implementations

### Base Class Definition

The base class for execution strategies is defined in `autogen_extensions/execution_strategy.py`:

```python
from typing import Callable, List, Any
from abc import ABC, abstractmethod

class ExecutionStrategy(ABC):
    """
    Abstract base class for execution strategies.
    Implementations control how a list of callables (tasks) are executed:
    sequentially, in threads, in processes, or asynchronously.
    """
    @abstractmethod
    def run_tasks(self, callables: List[Callable[[], Any]]) -> List[Any]:
        """
        Execute a list of callables and return their results as a list.
        Implementations may run tasks sequentially, in parallel, or asynchronously.
        """
        pass
```

### Deriving Real Execution Strategies

To implement a real execution strategy, subclass `ExecutionStrategy` and implement the `run_tasks` method. For example:

```python
from autogen_extensions.execution_strategy import ExecutionStrategy
from typing import Callable, List, Any

class SequentialStrategy(ExecutionStrategy):
    def run_tasks(self, callables: List[Callable[[], Any]]) -> List[Any]:
        return [fn() for fn in callables]

from concurrent.futures import ThreadPoolExecutor
class ThreadPoolStrategy(ExecutionStrategy):
    def __init__(self, max_workers=8):
        self.max_workers = max_workers
    def run_tasks(self, callables: List[Callable[[], Any]]) -> List[Any]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(fn) for fn in callables]
            return [f.result() for f in futures]
```

You can similarly implement process-based and async strategies by subclassing and providing the appropriate logic in `run_tasks`.

### Concrete Strategies

- **Sequential:** Runs tasks one after another (useful for debugging, testing, or small jobs)
- **ThreadPool:** Runs tasks in parallel threads (best for I/O-bound workloads, e.g., LLM API calls)
- **ProcessPool:** Runs tasks in parallel processes (best for CPU-bound workloads)
- **Async:** Runs tasks using Python's `asyncio` (best for async-compatible I/O-bound code)

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio

class SequentialStrategy(ExecutionStrategy):
    def run_tasks(self, callables):
        return [fn() for fn in callables]

class ThreadPoolStrategy(ExecutionStrategy):
    def __init__(self, max_workers=8):
        self.max_workers = max_workers
    def run_tasks(self, callables):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(fn) for fn in callables]
            return [f.result() for f in futures]

class ProcessPoolStrategy(ExecutionStrategy):
    def __init__(self, max_workers=4):
        self.max_workers = max_workers
    def run_tasks(self, callables):
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(fn) for fn in callables]
            return [f.result() for f in futures]

class AsyncStrategy(ExecutionStrategy):
    async def run_tasks_async(self, coros):
        return await asyncio.gather(*coros)
    def run_tasks(self, callables):
        # callables must return coroutines
        return asyncio.run(self.run_tasks_async([fn() for fn in callables]))
```

---

## 2. Dependency Injection in Orchestrators

Inject the strategy into your orchestrator or question-asking class:

```python
class UserQuestionOrchestrator:
    def __init__(self, execution_strategy: ExecutionStrategy):
        self.execution_strategy = execution_strategy
    def ask_all(self, user_callables):
        return self.execution_strategy.run_tasks(user_callables)
```

---

## 3. Usage Examples

```python
# Choose strategy at runtime/config
default_strategy = ThreadPoolStrategy(max_workers=8)
orchestrator = UserQuestionOrchestrator(execution_strategy=default_strategy)
results = orchestrator.ask_all([lambda: agent.ask(question) for agent in agents])
```

For async-compatible agents:
```python
async_strategy = AsyncStrategy()
orchestrator = UserQuestionOrchestrator(execution_strategy=async_strategy)
results = orchestrator.ask_all([lambda: agent.ask_async(question) for agent in agents])
```

---

## 4. Error Handling

- **Sequential:** Exceptions propagate immediately.
- **Thread/Process Pools:** Exceptions in tasks are raised when calling `result()` on the future. You can catch and log them individually:

```python
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(fn) for fn in callables]
    results = []
    for f in futures:
        try:
            results.append(f.result())
        except Exception as e:
            # Handle/log error for this task
            results.append(e)
```

- **Async:** Use `return_exceptions=True` in `asyncio.gather` to collect exceptions as results:

```python
async def run_tasks_async(self, coros):
    return await asyncio.gather(*coros, return_exceptions=True)
```

- **Recommendation:**
    - Always log or handle exceptions per task.
    - Optionally, retry failed tasks or aggregate errors for reporting.

---

## 5. Use Cases in User Questioning and Idea-to-Questions

- **User Questioning:**
    - Use the strategy to control how questions are asked to many users (sequentially, in threads, or processes).
    - Enables easy switching between fast parallel runs and deterministic sequential runs for debugging.
- **Idea-to-Questions:**
    - Use the same pattern to parallelize question generation from an idea.
    - For each generated question, recursively use the strategy to ask users.
    - Supports nested parallelism and full flexibility.

---

## 6. Summary Table

| Strategy         | Use Case                        | Pros                        | Cons                  |
|------------------|---------------------------------|-----------------------------|-----------------------|
| Sequential       | Debugging, testing, small jobs  | Simple, deterministic       | Slow for many tasks   |
| ThreadPool       | I/O-bound (API calls)           | Fast, low overhead          | GIL for CPU-bound     |
| ProcessPool      | CPU-bound (heavy computation)   | True parallelism            | More overhead         |
| Async            | Async I/O-bound, async APIs     | Scalable, efficient         | Requires async code   |

---

## 7. Recommendations

- Use **ThreadPoolStrategy** for most LLM API or I/O-bound workloads.
- Use **ProcessPoolStrategy** for CPU-heavy post-processing.
- Use **AsyncStrategy** if your agents and APIs are async-compatible.
- Use **SequentialStrategy** for debugging, deterministic tests, or very small jobs.
- Always inject the strategy for maximum flexibility and testability.
- Handle errors per task and log or aggregate as needed.

---

## 8. Extensibility

- You can add new strategies (e.g., distributed, hybrid, retrying) as needed.
- The same pattern can be used for any batch or recursive agent task (summarization, validation, etc.).

--- 