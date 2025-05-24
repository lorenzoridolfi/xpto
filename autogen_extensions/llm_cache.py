from abc import ABC, abstractmethod
from typing import Optional, Any


class AbstractLLMCache(ABC):
    """
    Abstract base class for LLM cache systems. Can be used in any LLM-driven application.
    """

    @abstractmethod
    def lookup(self, variable_prompt: str) -> Optional[Any]:
        """
        Look up a cached response for the given variable prompt.
        Returns the cached response if found, else None.
        """
        pass

    @abstractmethod
    def store(self, variable_prompt: str, response: Any):
        """
        Store a response for the given variable prompt in the cache.
        """
        pass


class MockLLMCache(AbstractLLMCache):
    """
    Mock cache for testing. Stores and retrieves responses in-memory.
    """

    def __init__(self, initial: Optional[dict[str, Any]] = None):
        self.stored = dict(initial) if initial is not None else {}

    def lookup(self, variable_prompt: str) -> Optional[Any]:
        return self.stored.get(variable_prompt)

    def store(self, variable_prompt: str, response: Any):
        self.stored[variable_prompt] = response


class NoCacheLLMCache(AbstractLLMCache):
    """
    No-cache implementation: always misses the cache.
    """

    def lookup(self, variable_prompt: str) -> Optional[Any]:
        return None

    def store(self, variable_prompt: str, response: Any):
        pass
