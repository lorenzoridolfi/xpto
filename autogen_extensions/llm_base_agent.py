from autogen_extensions.base_agent import BaseAgent
from typing import Callable, Any


class LLMBaseAgent(BaseAgent):
    """
    BaseAgent subclass with dependency injection for the LLM call (e.g., cache, mock, etc).
    """

    def __init__(self, llm_call: Callable[[str], Any], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._llm_call = llm_call

    def _call_llm(self, prompt: str, *args, **kwargs):
        return self._llm_call(prompt, *args, **kwargs)
