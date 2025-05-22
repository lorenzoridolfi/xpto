from src.llm_cache import LLMCache


def test_llm_cache_hit_and_miss(tmp_path):
    cache = LLMCache()
    messages = [{"role": "user", "content": "test"}]
    response = "result"
    cache.put(messages, response)
    assert cache.get(messages) == response
    other_messages = [{"role": "user", "content": "other"}]
    assert cache.get(other_messages) is None
