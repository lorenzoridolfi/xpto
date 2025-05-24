from autogen_extensions.base_agent import BaseAgent
from typing import Callable, Any, Optional
from autogen_extensions.openai_utils import get_openai_client


class LLMBaseAgent(BaseAgent):
    """
    BaseAgent subclass with dependency injection for the LLM call (e.g., cache, mock, etc).
    If llm_call is not provided, uses OpenAI's chat.completions.create as the default LLM call.
    """

    def __init__(self, llm_call: Optional[Callable[..., Any]] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if llm_call is not None:
            self._llm_call = llm_call
        else:
            client = get_openai_client()

            def default_llm_call(*args, **kwargs):
                # Expect kwargs to include model, messages, etc.
                return client.chat.completions.create(*args, **kwargs)

            self._llm_call = default_llm_call

    def _call_llm(self, *args, **kwargs):
        return self._llm_call(*args, **kwargs)
