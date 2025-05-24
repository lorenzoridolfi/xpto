from autogen_extensions.llm_cache import AbstractLLMCache
from typing import Optional, Any


class BaseAgent:
    """
    Cache-enabled base agent class with LLM cache dependency injection.
    This version is compatible with legacy usage: if llm_cache is not provided, caching is disabled.
    Place this class in a separate module (e.g., base_agent_cache.py) to avoid breaking existing code.
    """

    def __init__(self, llm_cache: Optional[AbstractLLMCache] = None, **kwargs):
        self.llm_cache = llm_cache
        # Store any other agent configuration in kwargs
        self.config = kwargs

    def llm_call(self, prompt: str, *args, **kwargs) -> Any:
        """
        Call the LLM, using the cache if available.
        """
        if self.llm_cache:
            cached = self.llm_cache.lookup(prompt)
            if cached is not None:
                return cached
        # If not cached, call the LLM (subclass should implement this)
        result = self._call_llm(prompt, *args, **kwargs)
        if self.llm_cache:
            self.llm_cache.store(prompt, result)
        return result

    def _call_llm(self, prompt: str, *args, **kwargs) -> Any:
        """
        Subclasses should override this method to implement the actual LLM call.
        """
        raise NotImplementedError("Subclasses must implement _call_llm()")
