import pytest
import asyncio
from pydantic import BaseModel, ValidationError
from autogen_extensions.structured_llm_base_agent import StructuredLLMBaseAgent

class TestModel(BaseModel):
    foo: int
    bar: str

class DummyStructuredAgent(StructuredLLMBaseAgent):
    output_model = TestModel

@pytest.fixture
def valid_llm_call():
    async def _call_llm(*args, **kwargs):
        class Msg:
            content = '{"foo": 42, "bar": "baz"}'
        class Choice:
            message = Msg()
        class Response:
            choices = [Choice()]
        return Response()
    return _call_llm

@pytest.fixture
def invalid_llm_call():
    async def _call_llm(*args, **kwargs):
        class Msg:
            content = '{"foo": "not_an_int", "bar": 123}'
        class Choice:
            message = Msg()
        class Response:
            choices = [Choice()]
        return Response()
    return _call_llm

@pytest.mark.asyncio
async def test_structured_llm_success(valid_llm_call):
    agent = DummyStructuredAgent(llm_call=valid_llm_call)
    result = await agent.call_structured_llm(prompt="Give me foo and bar")
    assert isinstance(result, TestModel)
    assert result.foo == 42
    assert result.bar == "baz"

@pytest.mark.asyncio
async def test_structured_llm_validation_error(invalid_llm_call):
    agent = DummyStructuredAgent(llm_call=invalid_llm_call)
    with pytest.raises(RuntimeError) as excinfo:
        await agent.call_structured_llm(prompt="Give me foo and bar")
    assert "Failed to validate structured LLM output" in str(excinfo.value)

@pytest.mark.asyncio
async def test_structured_llm_custom_messages(valid_llm_call):
    agent = DummyStructuredAgent(llm_call=valid_llm_call)
    messages = [
        {"role": "system", "content": "Custom system."},
        {"role": "user", "content": "Custom user."},
    ]
    result = await agent.call_structured_llm(prompt="ignored", messages=messages)
    assert isinstance(result, TestModel)

@pytest.mark.asyncio
async def test_structured_llm_output_model_via_constructor(valid_llm_call):
    class OtherModel(BaseModel):
        x: int
    class Agent(StructuredLLMBaseAgent):
        pass
    agent = Agent(output_model=OtherModel, llm_call=valid_llm_call)
    # Patch the LLM to return valid for OtherModel
    async def patched_llm(*args, **kwargs):
        class Msg:
            content = '{"x": 123}'
        class Choice:
            message = Msg()
        class Response:
            choices = [Choice()]
        return Response()
    agent._call_llm = patched_llm
    result = await agent.call_structured_llm(prompt="Give me x")
    assert isinstance(result, OtherModel)
    assert result.x == 123

@pytest.mark.asyncio
async def test_structured_llm_output_schema(valid_llm_call):
    try:
        from pydantic import create_model_from_schema
    except ImportError:
        pytest.skip("Pydantic v2+ required for schema-based model creation")
    schema = {
        "title": "TestSchema",
        "type": "object",
        "properties": {
            "foo": {"type": "integer"},
            "bar": {"type": "string"}
        },
        "required": ["foo", "bar"]
    }
    agent = StructuredLLMBaseAgent(output_schema=schema, llm_call=valid_llm_call)
    result = await agent.call_structured_llm(prompt="Give me foo and bar")
    assert result.foo == 42
    assert result.bar == "baz"
    # The returned object is a Pydantic model, but not TestModel
    assert hasattr(result, "foo") and hasattr(result, "bar") 