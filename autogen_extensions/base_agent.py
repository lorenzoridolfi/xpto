from autogen_extensions.llm_cache import AbstractLLMCache
from typing import Optional, Any

class BaseAgent:
    """
    Cache-enabled base agent class with LLM cache dependency injection.
    Compatible with legacy usage: if llm_cache is not provided, caching is disabled.
    Subclasses should override _call_llm.
    """
    def __init__(self, llm_cache: Optional[AbstractLLMCache] = None, **kwargs):
        self.llm_cache = llm_cache
        self.config = kwargs

    def llm_call(self, prompt: str, *args, **kwargs) -> Any:
        if self.llm_cache:
            cached = self.llm_cache.lookup(prompt)
            if cached is not None:
                return cached
        result = self._call_llm(prompt, *args, **kwargs)
        if self.llm_cache:
            self.llm_cache.store(prompt, result)
        return result

    def _call_llm(self, prompt: str, *args, **kwargs) -> Any:
        raise NotImplementedError("Subclasses must implement _call_llm()") 