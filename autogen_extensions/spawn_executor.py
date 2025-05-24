from typing import Callable, List, Any
from autogen_extensions.execution_strategy import ExecutionStrategy


class SpawnExecutor:
    """
    Executes a list of zero-argument callables using the provided execution strategy.
    Does not perform any aggregation or join; simply returns the list of results.
    """

    def __init__(self, execution_strategy: ExecutionStrategy):
        self.execution_strategy = execution_strategy

    def run(self, callables: List[Callable[[], Any]]) -> List[Any]:
        """
        Execute the given callables and return their results as a list.
        """
        return self.execution_strategy.run_tasks(callables)
