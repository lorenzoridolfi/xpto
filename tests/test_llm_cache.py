import pytest
from autogen_extensions.llm_cache import AbstractLLMCache, MockLLMCache, NoCacheLLMCache


def test_llm_cache_hit_and_miss(tmp_path):
    cache = MockLLMCache()
    messages = "test"
    response = "result"
    cache.store(messages, response)
    assert cache.lookup(messages) == response
    other_messages = "other"
    assert cache.lookup(other_messages) is None


class DummyLLMCache(AbstractLLMCache):
    def lookup(self, variable_prompt: str):
        return "dummy"

    def store(self, variable_prompt: str, response):
        pass


def test_abstract_llm_cache_cannot_instantiate():
    with pytest.raises(TypeError):
        AbstractLLMCache()
    # But a subclass with implementations can be instantiated
    cache = DummyLLMCache()
    assert cache.lookup("foo") == "dummy"


def test_mock_llm_cache_lookup_and_store():
    cache = MockLLMCache({"q1": "a1", "q2": "a2"})
    assert cache.lookup("q1") == "a1"
    assert cache.lookup("q2") == "a2"
    assert cache.lookup("q3") is None
    cache.store("q3", "a3")
    assert cache.stored["q3"] == "a3"


def test_no_cache_llm_cache_always_miss():
    cache = NoCacheLLMCache()
    assert cache.lookup("anything") is None
    cache.store("anything", "something")  # Should not raise
    # No state to check, but ensure no error
