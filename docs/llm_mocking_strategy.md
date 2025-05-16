# LLM Mocking Strategy for Testing

## Overview
This document outlines the strategy for mocking LLM responses in tests to achieve deterministic results, reduce costs, and improve test reliability.

## 1. Mock LLM Implementation

### Basic Mock LLM
```python
class MockLLM:
    def __init__(self, response_map=None):
        self.response_map = response_map or {}
        self.call_history = []
    
    async def generate(self, prompt, **kwargs):
        self.call_history.append({
            "prompt": prompt,
            "kwargs": kwargs
        })
        
        # Return predefined response based on prompt
        for pattern, response in self.response_map.items():
            if pattern in prompt:
                return response
        
        # Default response if no pattern matches
        return "I am a mock LLM response."

    def get_call_history(self):
        return self.call_history
```

### Response Mapping
```python
# Example response mapping
llm_responses = {
    "read file": "Here is the content of the file: test content",
    "update manifest": "Manifest updated successfully to version 1.0.0",
    "validate": "Validation passed with no errors",
    "error": "Error: Invalid input format",
    "default": "I understand your request and will proceed accordingly."
}

# Usage in tests
mock_llm = MockLLM(llm_responses)
```

## 2. Test Scenarios

### Basic Response Testing
```python
def test_basic_llm_interaction():
    """Test basic LLM interaction with mocked responses"""
    mock_llm = MockLLM({
        "read file": "File content: test.txt"
    })
    
    response = await mock_llm.generate("Please read the file")
    assert "File content" in response
    assert len(mock_llm.call_history) == 1
```

### Error Handling Testing
```python
def test_llm_error_handling():
    """Test LLM error handling with mocked responses"""
    mock_llm = MockLLM({
        "error": "Error: Invalid input"
    })
    
    response = await mock_llm.generate("This will cause an error")
    assert "Error" in response
    assert mock_llm.call_history[0]["prompt"] == "This will cause an error"
```

### Complex Interaction Testing
```python
def test_complex_llm_interaction():
    """Test complex LLM interactions with multiple responses"""
    mock_llm = MockLLM({
        "step 1": "First step completed",
        "step 2": "Second step completed",
        "step 3": "Final step completed"
    })
    
    # Test multi-step interaction
    responses = []
    for step in ["step 1", "step 2", "step 3"]:
        response = await mock_llm.generate(f"Execute {step}")
        responses.append(response)
    
    assert len(responses) == 3
    assert "completed" in responses[-1]
```

## 3. Integration with Existing Tests

### BaseSupervisor Tests with Mock LLM
```python
@pytest.fixture
def mock_llm():
    return MockLLM({
        "process task": "Task processed successfully",
        "handle error": "Error handled appropriately"
    })

def test_supervisor_with_mock_llm(mock_llm):
    """Test supervisor with mocked LLM"""
    supervisor = BaseSupervisor({
        "agents": ["agent1"],
        "llm": mock_llm
    })
    
    result = await supervisor.process_task({
        "id": "task1",
        "type": "test"
    })
    
    assert result["status"] == "completed"
    assert len(mock_llm.call_history) > 0
```

### Program-Specific Tests
```python
def test_toy_example_with_mock_llm():
    """Test toy example with mocked LLM responses"""
    mock_llm = MockLLM({
        "read file": "File content: test.txt",
        "write file": "File written successfully",
        "verify": "Verification passed"
    })
    
    supervisor = ToyExampleSupervisor({
        "agents": ["reader", "writer"],
        "llm": mock_llm
    })
    
    result = await supervisor.process_task({
        "type": "file_processing",
        "data": {"file": "test.txt"}
    })
    
    assert result["status"] == "completed"
    assert "processed" in result["data"]
```

## 4. Response Patterns

### Common Response Patterns
```python
RESPONSE_PATTERNS = {
    # Success patterns
    "success": "Operation completed successfully",
    "validation": "Validation passed",
    "update": "Update completed",
    
    # Error patterns
    "error": "Error occurred",
    "timeout": "Operation timed out",
    "invalid": "Invalid input",
    
    # Specific task patterns
    "file_read": "File content: {content}",
    "file_write": "File written: {path}",
    "manifest_update": "Manifest updated to {version}"
}
```

### Dynamic Response Generation
```python
class DynamicMockLLM(MockLLM):
    def __init__(self, response_map=None):
        super().__init__(response_map)
        self.response_count = {}
    
    async def generate(self, prompt, **kwargs):
        # Track response frequency
        for pattern in self.response_map:
            if pattern in prompt:
                self.response_count[pattern] = self.response_count.get(pattern, 0) + 1
        
        # Generate dynamic response
        response = await super().generate(prompt, **kwargs)
        return response.format(
            count=self.response_count.get(prompt, 0),
            timestamp=datetime.now().isoformat()
        )
```

## 5. Best Practices

### Response Management
1. **Organize Responses**
   - Group by functionality
   - Use consistent patterns
   - Document response formats

2. **Response Validation**
   - Validate response format
   - Check required fields
   - Ensure consistency

3. **Error Simulation**
   - Simulate common errors
   - Test error handling
   - Verify recovery

### Test Organization
1. **Response Categories**
   - Success responses
   - Error responses
   - Edge cases
   - Special scenarios

2. **Test Data**
   - Realistic prompts
   - Varied inputs
   - Edge cases
   - Error conditions

3. **Coverage**
   - All response types
   - Error scenarios
   - Edge cases
   - Integration points

## 6. Implementation Tips

### Setting Up Mock LLM
1. **Configuration**
   ```python
   # In conftest.py
   @pytest.fixture
   def mock_llm():
       return MockLLM({
           "pattern1": "response1",
           "pattern2": "response2"
       })
   ```

2. **Usage in Tests**
   ```python
   def test_with_mock_llm(mock_llm):
       # Use mock_llm in tests
       response = await mock_llm.generate("test prompt")
       assert response == "expected response"
   ```

### Response Management
1. **Centralized Responses**
   ```python
   # In test_data.py
   LLM_RESPONSES = {
       "category1": {
           "pattern1": "response1",
           "pattern2": "response2"
       },
       "category2": {
           "pattern3": "response3",
           "pattern4": "response4"
       }
   }
   ```

2. **Response Validation**
   ```python
   def validate_response(response, expected_format):
       assert all(key in response for key in expected_format)
       assert isinstance(response["content"], str)
   ``` 