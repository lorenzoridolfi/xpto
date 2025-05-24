from autogen_extensions.text_hashing import hash_text, hash_json, hash_dict
import json


def test_hash_text_standardization():
    t1 = "Hello   World"
    t2 = "  hello world  "
    t3 = "HELLO\tWORLD"
    t4 = "hello\nworld"
    # All should hash to the same value
    h = hash_text(t1)
    assert hash_text(t2) == h
    assert hash_text(t3) == h
    assert hash_text(t4) == h
    # Different content should hash differently
    assert hash_text("Hello there") != h


def test_hash_json_equivalence():
    d = {"a": 1, "b": [2, 3], "c": {"d": 4}}
    j1 = json.dumps(d, indent=2)
    j2 = json.dumps(d, separators=(",", ":"))
    j3 = '{"c": {"d": 4}, "b": [2,3], "a": 1}'
    # All should hash to the same value
    h = hash_json(j1)
    assert hash_json(j2) == h
    assert hash_json(j3) == h
    # Different JSON should hash differently
    assert hash_json(json.dumps({"a": 2})) != h


def test_hash_dict_equivalence():
    d1 = {"x": 1, "y": 2}
    d2 = {"y": 2, "x": 1}
    # Key order should not affect hash
    h = hash_dict(d1)
    assert hash_dict(d2) == h
    # Different dict should hash differently
    assert hash_dict({"x": 1, "y": 3}) != h


def test_hash_dict_vs_json():
    d = {"foo": [1, 2, 3], "bar": {"baz": 4}}
    j = json.dumps(d, indent=4)
    # hash_dict and hash_json should match for equivalent content
    assert hash_dict(d) == hash_json(j)
