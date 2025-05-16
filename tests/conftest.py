"""Test configuration and shared fixtures.

This module provides shared fixtures and configuration for all tests.
"""

import pytest
from typing import Dict, Any, Generator
from datetime import datetime
from .mock_llm import MockLLM, DynamicMockLLM


# Basic response patterns
BASIC_RESPONSES = {
    "hello": "Hello, how can I help you?",
    "error": "Error: Invalid input",
    "success": "Operation completed successfully",
    "read file": "File content: test.txt",
    "write file": "File written successfully",
    "validate": "Validation passed",
    "update": "Update completed",
    "delete": "Deletion successful",
    "search": "Search results found",
    "analyze": "Analysis complete",
    "default": "I am a mock LLM response."
}

# Template response patterns
TEMPLATE_RESPONSES = {
    "count": "This is call number {count}",
    "time": "Current time is {timestamp}",
    "length": "Prompt length is {prompt_length} characters",
    "file": "File {filename} processed at {timestamp}",
    "task": "Task {task_id} completed in {count} steps",
    "error": "Error {error_code} occurred at {timestamp}",
    "status": "Status: {status} at {timestamp}",
    "progress": "Progress: {count}/{total} at {timestamp}"
}

# Error response patterns
ERROR_RESPONSES = {
    "timeout": "Error: Operation timed out",
    "invalid": "Error: Invalid input format",
    "permission": "Error: Permission denied",
    "not_found": "Error: Resource not found",
    "conflict": "Error: Resource conflict",
    "server": "Error: Internal server error",
    "network": "Error: Network connection failed",
    "validation": "Error: Validation failed"
}

# Complex response patterns
COMPLEX_RESPONSES = {
    "multi_step": {
        "step1": "First step completed",
        "step2": "Second step completed",
        "step3": "Final step completed"
    },
    "conditional": {
        "if_success": "Operation succeeded",
        "if_error": "Operation failed",
        "if_timeout": "Operation timed out"
    },
    "nested": {
        "outer": {
            "inner1": "Inner operation 1",
            "inner2": "Inner operation 2"
        }
    }
}


@pytest.fixture
def basic_response_map() -> Dict[str, str]:
    """Fixture providing basic response patterns."""
    return BASIC_RESPONSES.copy()


@pytest.fixture
def template_response_map() -> Dict[str, str]:
    """Fixture providing template response patterns."""
    return TEMPLATE_RESPONSES.copy()


@pytest.fixture
def error_response_map() -> Dict[str, str]:
    """Fixture providing error response patterns."""
    return ERROR_RESPONSES.copy()


@pytest.fixture
def complex_response_map() -> Dict[str, Any]:
    """Fixture providing complex response patterns."""
    return COMPLEX_RESPONSES.copy()


@pytest.fixture
def mock_llm(basic_response_map: Dict[str, str]) -> MockLLM:
    """Fixture providing a basic MockLLM instance."""
    return MockLLM(basic_response_map)


@pytest.fixture
def dynamic_mock_llm(template_response_map: Dict[str, str]) -> DynamicMockLLM:
    """Fixture providing a DynamicMockLLM instance."""
    return DynamicMockLLM(template_response_map)


@pytest.fixture
def error_mock_llm(error_response_map: Dict[str, str]) -> MockLLM:
    """Fixture providing a MockLLM instance configured for error testing."""
    return MockLLM(error_response_map)


@pytest.fixture
def complex_mock_llm(complex_response_map: Dict[str, Any]) -> MockLLM:
    """Fixture providing a MockLLM instance with complex response patterns."""
    return MockLLM(complex_response_map)


@pytest.fixture
def mock_llm_with_history(mock_llm: MockLLM) -> Generator[MockLLM, None, None]:
    """Fixture providing a MockLLM instance with pre-populated history."""
    # Add some history
    mock_llm.call_history = [
        {
            "prompt": "test prompt 1",
            "kwargs": {"temperature": 0.7},
            "timestamp": datetime.now().isoformat()
        },
        {
            "prompt": "test prompt 2",
            "kwargs": {"max_tokens": 100},
            "timestamp": datetime.now().isoformat()
        }
    ]
    yield mock_llm
    # Cleanup
    mock_llm.clear_history()


@pytest.fixture
def dynamic_mock_llm_with_counts(dynamic_mock_llm: DynamicMockLLM) -> Generator[DynamicMockLLM, None, None]:
    """Fixture providing a DynamicMockLLM instance with pre-populated counts."""
    # Add some counts
    dynamic_mock_llm.response_count = {
        "count": 5,
        "time": 3,
        "length": 2
    }
    yield dynamic_mock_llm
    # Cleanup
    dynamic_mock_llm.clear_counts()


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "asyncio: mark test as async"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test"
    )


# Test session setup and teardown
@pytest.fixture(scope="session", autouse=True)
def setup_test_session():
    """Setup and teardown for the entire test session."""
    # Setup
    print("\nStarting test session...")
    yield
    # Teardown
    print("\nTest session completed.")


# Test module setup and teardown
@pytest.fixture(scope="module", autouse=True)
def setup_test_module():
    """Setup and teardown for each test module."""
    # Setup
    print("\nStarting test module...")
    yield
    # Teardown
    print("\nTest module completed.")


# Test function setup and teardown
@pytest.fixture(autouse=True)
def setup_test_function():
    """Setup and teardown for each test function."""
    # Setup
    print("\nStarting test function...")
    yield
    # Teardown
    print("\nTest function completed.") 