"""Mock LLM implementation for testing.

This module provides mock implementations of LLM classes for testing purposes.
It allows for deterministic responses and controlled testing scenarios without
making actual API calls to LLM services.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any


class MockLLM:
    """A mock LLM implementation for testing.
    
    This class simulates an LLM by returning predefined responses based on
    patterns in the input prompts. It maintains a history of all calls made
    to it for verification purposes.
    """
    
    def __init__(self, response_map: Optional[Dict[str, str]] = None):
        """Initialize the mock LLM.
        
        Args:
            response_map: A dictionary mapping prompt patterns to responses.
                         If None, an empty dictionary is used.
        """
        self.response_map = response_map or {}
        self.call_history: List[Dict[str, Any]] = []
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response for the given prompt.
        
        Args:
            prompt: The input prompt to generate a response for.
            **kwargs: Additional arguments that would be passed to a real LLM.
        
        Returns:
            A string response based on the prompt pattern matching.
        """
        # Record the call in history
        self.call_history.append({
            "prompt": prompt,
            "kwargs": kwargs,
            "timestamp": datetime.now().isoformat()
        })
        
        # Return predefined response based on prompt
        for pattern, response in self.response_map.items():
            if pattern in prompt:
                return response
        
        # Default response if no pattern matches
        return "I am a mock LLM response."

    def get_call_history(self) -> List[Dict[str, Any]]:
        """Get the history of all calls made to the mock LLM.
        
        Returns:
            A list of dictionaries containing call details.
        """
        return self.call_history
    
    def clear_history(self) -> None:
        """Clear the call history."""
        self.call_history = []


class DynamicMockLLM(MockLLM):
    """A dynamic mock LLM that can generate responses with variable content.
    
    This class extends MockLLM to provide dynamic responses that can include
    call counts, timestamps, and other variable information.
    """
    
    def __init__(self, response_map: Optional[Dict[str, str]] = None):
        """Initialize the dynamic mock LLM.
        
        Args:
            response_map: A dictionary mapping prompt patterns to response templates.
                         The templates can include format placeholders.
        """
        super().__init__(response_map)
        self.response_count: Dict[str, int] = {}
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a dynamic response for the given prompt.
        
        Args:
            prompt: The input prompt to generate a response for.
            **kwargs: Additional arguments that would be passed to a real LLM.
        
        Returns:
            A string response with dynamic content based on the prompt pattern matching.
        """
        # Track response frequency
        for pattern in self.response_map:
            if pattern in prompt:
                self.response_count[pattern] = self.response_count.get(pattern, 0) + 1
        
        # Get base response from parent class
        response = await super().generate(prompt, **kwargs)
        
        # Add dynamic content if response is a template
        try:
            return response.format(
                count=self.response_count.get(prompt, 0),
                timestamp=datetime.now().isoformat(),
                prompt_length=len(prompt)
            )
        except KeyError:
            # If formatting fails, return the response as is
            return response
    
    def get_response_counts(self) -> Dict[str, int]:
        """Get the count of responses generated for each pattern.
        
        Returns:
            A dictionary mapping patterns to their response counts.
        """
        return self.response_count.copy()
    
    def clear_counts(self) -> None:
        """Clear the response counts."""
        self.response_count = {} 