from abc import ABC, abstractmethod
from typing import Any, List, Callable, Optional
from autogen_extensions.execution_strategy import ExecutionStrategy


class SpawnJoinOrchestrator(ABC):
    """
    Abstract orchestrator for the spawn-join pattern:
    - Spawn: generate a set of tasks (callables)
    - Join: execute them (possibly in parallel) and collect results
    - Summarize: aggregate or post-process the results

    The execution strategy is a required, static dependency for each orchestrator instance.
    This pattern is suitable for workflows like:
    - Asking the same question to many users and summarizing answers
    - Breaking an idea into questions, asking users, and summarizing
    - Any batch or recursive agent task with join/aggregation
    """

    execution_strategy: ExecutionStrategy

    def __init__(
        self, execution_strategy: ExecutionStrategy, llm_cache: Optional[Any] = None
    ):
        self.execution_strategy = execution_strategy
        self.llm_cache = llm_cache

    @abstractmethod
    def generate_tasks(self, *args, **kwargs) -> List[Callable[[], Any]]:
        """
        Produce a list of callables for the tasks to execute (e.g., ask user, generate question).
        Subclasses must implement this.
        """
        pass

    @abstractmethod
    def summarize(self, results: List[Any], *args, **kwargs) -> Any:
        """
        Summarize or aggregate the results of the tasks.
        Subclasses must implement this.
        """
        pass

    def run(self, *args, **kwargs) -> Any:
        """
        Orchestrate the workflow: generate tasks, execute them, and summarize results.
        The execution strategy is fixed for the lifetime of this orchestrator instance.
        """
        tasks = self.generate_tasks(*args, **kwargs)
        results = self.execution_strategy.run_tasks(tasks)
        return self.summarize(results, *args, **kwargs)
