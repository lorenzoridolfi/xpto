import tempfile
from src.llm_cache import LLMCache

def test_llm_cache_hit_and_miss(tmp_path):
    cache = LLMCache(str(tmp_path / "cache.json"))
    prompt = "test"
    response = "result"
    cache.save(prompt, response)
    assert cache.get(prompt) == response
    assert cache.get("other") is None
