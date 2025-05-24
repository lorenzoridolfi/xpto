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
