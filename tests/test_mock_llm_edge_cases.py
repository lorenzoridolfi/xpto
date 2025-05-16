"""Tests for edge cases and error handling in the mock LLM implementation."""

import pytest
from datetime import datetime
from typing import Dict, Any
from .mock_llm import MockLLM, DynamicMockLLM


class TestEdgeCases:
    """Tests for edge cases in LLM responses."""
    
    @pytest.mark.asyncio
    async def test_empty_prompt(self, mock_llm):
        """Test handling of empty prompts."""
        response = await mock_llm.generate("")
        assert response == "I am a mock LLM response."
    
    @pytest.mark.asyncio
    async def test_very_long_prompt(self, mock_llm):
        """Test handling of very long prompts."""
        long_prompt = "x" * 10000
        response = await mock_llm.generate(long_prompt)
        assert response == "I am a mock LLM response."
    
    @pytest.mark.asyncio
    async def test_special_characters(self, mock_llm):
        """Test handling of prompts with special characters."""
        special_prompt = "!@#$%^&*()_+{}|:<>?[]\\;',./"
        response = await mock_llm.generate(special_prompt)
        assert response == "I am a mock LLM response."
    
    @pytest.mark.asyncio
    async def test_unicode_characters(self, mock_llm):
        """Test handling of prompts with Unicode characters."""
        unicode_prompt = "你好世界！"
        response = await mock_llm.generate(unicode_prompt)
        assert response == "I am a mock LLM response."


class TestErrorHandling:
    """Tests for error handling in LLM responses."""
    
    @pytest.mark.asyncio
    async def test_invalid_kwargs(self, mock_llm):
        """Test handling of invalid keyword arguments."""
        # Test with invalid temperature
        response = await mock_llm.generate("test", temperature=2.0)
        assert response == "I am a mock LLM response."
        
        # Test with invalid max_tokens
        response = await mock_llm.generate("test", max_tokens=-1)
        assert response == "I am a mock LLM response."
    
    @pytest.mark.asyncio
    async def test_missing_kwargs(self, mock_llm):
        """Test handling of missing keyword arguments."""
        response = await mock_llm.generate("test")
        assert response == "I am a mock LLM response."
    
    @pytest.mark.asyncio
    async def test_invalid_response_map(self, mock_llm):
        """Test handling of invalid response map."""
        # Test with None response map
        llm = MockLLM(None)
        response = await llm.generate("test")
        assert response == "I am a mock LLM response."
        
        # Test with empty response map
        llm = MockLLM({})
        response = await llm.generate("test")
        assert response == "I am a mock LLM response."


class TestConcurrentOperations:
    """Tests for concurrent operations."""
    
    @pytest.mark.asyncio
    async def test_concurrent_generate(self, mock_llm):
        """Test concurrent generate operations."""
        import asyncio
        
        # Create multiple concurrent tasks
        tasks = [
            mock_llm.generate("hello"),
            mock_llm.generate("error"),
            mock_llm.generate("success")
        ]
        
        # Execute tasks concurrently
        responses = await asyncio.gather(*tasks)
        
        # Verify responses
        assert len(responses) == 3
        assert "Hello" in responses[0]
        assert "Error" in responses[1]
        assert "successfully" in responses[2]
    
    @pytest.mark.asyncio
    async def test_concurrent_history_access(self, mock_llm):
        """Test concurrent access to call history."""
        import asyncio
        
        # Create tasks that both generate and access history
        async def generate_and_check():
            await mock_llm.generate("test")
            return len(mock_llm.get_call_history())
        
        # Run multiple tasks
        tasks = [generate_and_check() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        # Verify results
        assert all(r > 0 for r in results)
        assert len(mock_llm.get_call_history()) == 5


class TestResourceManagement:
    """Tests for resource management."""
    
    @pytest.mark.asyncio
    async def test_history_cleanup(self, mock_llm):
        """Test cleanup of call history."""
        # Add some history
        for _ in range(10):
            await mock_llm.generate("test")
        
        # Clear history
        mock_llm.clear_history()
        assert len(mock_llm.get_call_history()) == 0
    
    @pytest.mark.asyncio
    async def test_count_cleanup(self, dynamic_mock_llm):
        """Test cleanup of response counts."""
        # Add some counts
        for _ in range(5):
            await dynamic_mock_llm.generate("count")
        
        # Clear counts
        dynamic_mock_llm.clear_counts()
        assert len(dynamic_mock_llm.get_response_counts()) == 0


class TestIntegration:
    """Tests for integration with other components."""
    
    @pytest.mark.asyncio
    async def test_with_prepopulated_history(self, mock_llm_with_history):
        """Test with pre-populated history."""
        # Verify initial history
        history = mock_llm_with_history.get_call_history()
        assert len(history) == 2
        assert history[0]["prompt"] == "test prompt 1"
        assert history[1]["prompt"] == "test prompt 2"
        
        # Add new history
        await mock_llm_with_history.generate("new prompt")
        assert len(mock_llm_with_history.get_call_history()) == 3
    
    @pytest.mark.asyncio
    async def test_with_prepopulated_counts(self, dynamic_mock_llm_with_counts):
        """Test with pre-populated counts."""
        # Verify initial counts
        counts = dynamic_mock_llm_with_counts.get_response_counts()
        assert counts["count"] == 5
        assert counts["time"] == 3
        assert counts["length"] == 2
        
        # Add new count
        await dynamic_mock_llm_with_counts.generate("count")
        assert dynamic_mock_llm_with_counts.get_response_counts()["count"] == 6 