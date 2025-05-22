import pytest
from src.json_validator import validate_json


def test_validate_json_valid():
    data = {"a": 1}
    schema = {"type": "object", "properties": {"a": {"type": "number"}}}
    assert validate_json(data, schema) is True


def test_validate_json_invalid():
    data = {"a": "bad"}
    schema = {"type": "object", "properties": {"a": {"type": "number"}}}
    with pytest.raises(Exception):
        validate_json(data, schema)
