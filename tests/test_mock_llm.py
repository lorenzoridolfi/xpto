"""Tests for the mock LLM implementation."""

import pytest
from datetime import datetime
from .mock_llm import MockLLM, DynamicMockLLM


@pytest.fixture
def basic_response_map():
    """Fixture providing a basic response map for testing."""
    return {
        "hello": "Hello, how can I help you?",
        "error": "Error: Invalid input",
        "success": "Operation completed successfully"
    }


@pytest.fixture
def template_response_map():
    """Fixture providing response templates for dynamic testing."""
    return {
        "count": "This is call number {count}",
        "time": "Current time is {timestamp}",
        "length": "Prompt length is {prompt_length} characters"
    }


@pytest.mark.asyncio
async def test_mock_llm_basic(basic_response_map):
    """Test basic functionality of MockLLM."""
    llm = MockLLM(basic_response_map)
    
    # Test matching response
    response = await llm.generate("hello there")
    assert response == "Hello, how can I help you?"
    
    # Test error response
    response = await llm.generate("this will cause an error")
    assert response == "Error: Invalid input"
    
    # Test default response
    response = await llm.generate("unmatched prompt")
    assert response == "I am a mock LLM response."


@pytest.mark.asyncio
async def test_mock_llm_history(basic_response_map):
    """Test call history tracking in MockLLM."""
    llm = MockLLM(basic_response_map)
    
    # Make some calls
    await llm.generate("hello")
    await llm.generate("error")
    
    # Check history
    history = llm.get_call_history()
    assert len(history) == 2
    assert history[0]["prompt"] == "hello"
    assert history[1]["prompt"] == "error"
    
    # Test history clearing
    llm.clear_history()
    assert len(llm.get_call_history()) == 0


@pytest.mark.asyncio
async def test_dynamic_mock_llm(template_response_map):
    """Test dynamic response generation in DynamicMockLLM."""
    llm = DynamicMockLLM(template_response_map)
    
    # Test count template
    response = await llm.generate("count")
    assert "call number 1" in response
    
    # Test timestamp template
    response = await llm.generate("time")
    assert "Current time is" in response
    
    # Test length template
    response = await llm.generate("length")
    assert "characters" in response


@pytest.mark.asyncio
async def test_dynamic_mock_llm_counts(template_response_map):
    """Test response counting in DynamicMockLLM."""
    llm = DynamicMockLLM(template_response_map)
    
    # Make multiple calls
    await llm.generate("count")
    await llm.generate("count")
    await llm.generate("count")
    
    # Check counts
    counts = llm.get_response_counts()
    assert counts["count"] == 3
    
    # Test count clearing
    llm.clear_counts()
    assert len(llm.get_response_counts()) == 0


@pytest.mark.asyncio
async def test_dynamic_mock_llm_error_handling(template_response_map):
    """Test error handling in DynamicMockLLM."""
    llm = DynamicMockLLM(template_response_map)
    
    # Test with invalid template
    response = await llm.generate("invalid template {nonexistent}")
    assert response == "I am a mock LLM response."


@pytest.mark.asyncio
async def test_mock_llm_kwargs(basic_response_map):
    """Test handling of additional kwargs in MockLLM."""
    llm = MockLLM(basic_response_map)
    
    # Test with additional parameters
    await llm.generate("hello", temperature=0.7, max_tokens=100)
    
    history = llm.get_call_history()
    assert history[0]["kwargs"]["temperature"] == 0.7
    assert history[0]["kwargs"]["max_tokens"] == 100


@pytest.mark.asyncio
async def test_dynamic_mock_llm_inheritance(template_response_map):
    """Test that DynamicMockLLM properly inherits from MockLLM."""
    llm = DynamicMockLLM(template_response_map)
    
    # Test inherited functionality
    await llm.generate("count")
    history = llm.get_call_history()
    assert len(history) == 1
    assert "timestamp" in history[0]
    
    # Test dynamic functionality
    counts = llm.get_response_counts()
    assert counts["count"] == 1 