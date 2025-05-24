import pytest
from autogen_extensions.base_agent_cache import BaseAgent
from autogen_extensions.llm_cache import MockLLMCache, NoCacheLLMCache


class EchoAgent(BaseAgent):
    """
    Concrete agent for testing: returns the prompt uppercased as the LLM response.
    """

    def _call_llm(self, prompt, *args, **kwargs):
        return prompt.upper()


def test_base_agent_with_mock_cache_hit_and_miss():
    # Pre-populate the mock cache with a response
    cache = MockLLMCache({"hello": "cached response"})
    agent = EchoAgent(llm_cache=cache)
    # Should hit the cache
    assert agent.llm_call("hello") == "cached response"
    # Should miss the cache and call _call_llm
    assert agent.llm_call("world") == "WORLD"
    # The new result should be stored in the cache
    assert cache.stored["world"] == "WORLD"


def test_base_agent_with_no_cache():
    agent = EchoAgent(llm_cache=NoCacheLLMCache())
    # Always calls _call_llm, never caches
    assert agent.llm_call("foo") == "FOO"
    assert agent.llm_call("foo") == "FOO"


def test_base_agent_without_cache():
    agent = EchoAgent()
    # Should behave like no cache
    assert agent.llm_call("bar") == "BAR"
    assert agent.llm_call("bar") == "BAR"
