from typing import Callable, List, Any
from autogen_extensions.execution_strategy import ExecutionStrategy
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio


class SequentialStrategy(ExecutionStrategy):
    """
    Executes tasks sequentially in the current thread.
    """

    def run_tasks(self, callables: List[Callable[[], Any]]) -> List[Any]:
        return [fn() for fn in callables]


class ThreadPoolStrategy(ExecutionStrategy):
    """
    Executes tasks in parallel using a thread pool.
    Best for I/O-bound workloads (e.g., LLM API calls).
    """

    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers

    def run_tasks(self, callables: List[Callable[[], Any]]) -> List[Any]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(fn) for fn in callables]
            return [f.result() for f in futures]


class ProcessPoolStrategy(ExecutionStrategy):
    """
    Executes tasks in parallel using a process pool.
    Best for CPU-bound workloads.
    """

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers

    def run_tasks(self, callables: List[Callable[[], Any]]) -> List[Any]:
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(fn) for fn in callables]
            return [f.result() for f in futures]


class AsyncStrategy(ExecutionStrategy):
    """
    Executes tasks as coroutines using asyncio.
    Callables must return coroutines (not regular functions).
    """

    async def run_tasks_async(self, coros: List[Any]) -> List[Any]:
        return await asyncio.gather(*coros)

    def run_tasks(self, callables: List[Callable[[], Any]]) -> List[Any]:
        # callables must return coroutines
        return asyncio.run(self.run_tasks_async([fn() for fn in callables]))
