"""Tests for different types of LLM responses and patterns."""

import pytest
from datetime import datetime
from typing import Dict, Any
from .mock_llm import MockLLM, DynamicMockLLM


class TestBasicResponses:
    """Tests for basic response patterns."""
    
    @pytest.mark.asyncio
    async def test_simple_responses(self, mock_llm):
        """Test simple response patterns."""
        # Test hello response
        response = await mock_llm.generate("hello")
        assert response == "Hello, how can I help you?"
        
        # Test error response
        response = await mock_llm.generate("error")
        assert response == "Error: Invalid input"
        
        # Test success response
        response = await mock_llm.generate("success")
        assert response == "Operation completed successfully"
    
    @pytest.mark.asyncio
    async def test_file_operations(self, mock_llm):
        """Test file operation responses."""
        # Test read file
        response = await mock_llm.generate("read file")
        assert "File content" in response
        
        # Test write file
        response = await mock_llm.generate("write file")
        assert "File written" in response
    
    @pytest.mark.asyncio
    async def test_default_response(self, mock_llm):
        """Test default response behavior."""
        # Test with unknown prompt
        response = await mock_llm.generate("unknown prompt")
        assert response == "I am a mock LLM response."
        
        # Test with empty prompt
        response = await mock_llm.generate("")
        assert response == "I am a mock LLM response."


class TestTemplateResponses:
    """Tests for template-based responses."""
    
    @pytest.mark.asyncio
    async def test_count_template(self, dynamic_mock_llm):
        """Test count-based template responses."""
        # Test first call
        response = await dynamic_mock_llm.generate("count")
        assert "call number 1" in response
        
        # Test subsequent calls
        response = await dynamic_mock_llm.generate("count")
        assert "call number 2" in response
    
    @pytest.mark.asyncio
    async def test_time_template(self, dynamic_mock_llm):
        """Test time-based template responses."""
        response = await dynamic_mock_llm.generate("time")
        assert "Current time is" in response
        assert datetime.fromisoformat(response.split("is ")[1])
    
    @pytest.mark.asyncio
    async def test_length_template(self, dynamic_mock_llm):
        """Test length-based template responses."""
        test_prompt = "test prompt"
        response = await dynamic_mock_llm.generate(test_prompt)
        assert f"characters" in response
        assert str(len(test_prompt)) in response


class TestErrorResponses:
    """Tests for error response patterns."""
    
    @pytest.mark.asyncio
    async def test_timeout_error(self, error_mock_llm):
        """Test timeout error response."""
        response = await error_mock_llm.generate("timeout")
        assert "Operation timed out" in response
    
    @pytest.mark.asyncio
    async def test_validation_error(self, error_mock_llm):
        """Test validation error response."""
        response = await error_mock_llm.generate("validation")
        assert "Validation failed" in response
    
    @pytest.mark.asyncio
    async def test_permission_error(self, error_mock_llm):
        """Test permission error response."""
        response = await error_mock_llm.generate("permission")
        assert "Permission denied" in response


class TestComplexResponses:
    """Tests for complex response patterns."""
    
    @pytest.mark.asyncio
    async def test_multi_step_responses(self, complex_mock_llm):
        """Test multi-step response patterns."""
        # Test step 1
        response = await complex_mock_llm.generate("step1")
        assert "First step completed" in response
        
        # Test step 2
        response = await complex_mock_llm.generate("step2")
        assert "Second step completed" in response
        
        # Test step 3
        response = await complex_mock_llm.generate("step3")
        assert "Final step completed" in response
    
    @pytest.mark.asyncio
    async def test_conditional_responses(self, complex_mock_llm):
        """Test conditional response patterns."""
        # Test success condition
        response = await complex_mock_llm.generate("if_success")
        assert "Operation succeeded" in response
        
        # Test error condition
        response = await complex_mock_llm.generate("if_error")
        assert "Operation failed" in response
        
        # Test timeout condition
        response = await complex_mock_llm.generate("if_timeout")
        assert "Operation timed out" in response


class TestResponseHistory:
    """Tests for response history tracking."""
    
    @pytest.mark.asyncio
    async def test_call_history(self, mock_llm):
        """Test call history tracking."""
        # Make some calls
        await mock_llm.generate("hello", temperature=0.7)
        await mock_llm.generate("error", max_tokens=100)
        
        # Check history
        history = mock_llm.get_call_history()
        assert len(history) == 2
        assert history[0]["prompt"] == "hello"
        assert history[0]["kwargs"]["temperature"] == 0.7
        assert history[1]["prompt"] == "error"
        assert history[1]["kwargs"]["max_tokens"] == 100
    
    @pytest.mark.asyncio
    async def test_history_clearing(self, mock_llm):
        """Test history clearing functionality."""
        # Add some history
        await mock_llm.generate("test")
        assert len(mock_llm.get_call_history()) == 1
        
        # Clear history
        mock_llm.clear_history()
        assert len(mock_llm.get_call_history()) == 0


class TestResponseCounts:
    """Tests for response counting functionality."""
    
    @pytest.mark.asyncio
    async def test_response_counts(self, dynamic_mock_llm):
        """Test response counting."""
        # Make multiple calls
        for _ in range(3):
            await dynamic_mock_llm.generate("count")
        
        # Check counts
        counts = dynamic_mock_llm.get_response_counts()
        assert counts["count"] == 3
    
    @pytest.mark.asyncio
    async def test_count_clearing(self, dynamic_mock_llm):
        """Test count clearing functionality."""
        # Add some counts
        await dynamic_mock_llm.generate("count")
        assert dynamic_mock_llm.get_response_counts()["count"] == 1
        
        # Clear counts
        dynamic_mock_llm.clear_counts()
        assert len(dynamic_mock_llm.get_response_counts()) == 0 