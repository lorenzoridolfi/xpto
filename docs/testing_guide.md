# Testing Guide

This guide explains how to use and understand the test suite for the mock LLM implementation.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Test Structure](#test-structure)
3. [Running Tests](#running-tests)
4. [Understanding Test Files](#understanding-test-files)
5. [Test Categories](#test-categories)
6. [Best Practices](#best-practices)

## Getting Started

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Installation
1. Install the required packages:
```bash
pip install pytest pytest-asyncio
```

### Basic Concepts
- **pytest**: A testing framework for Python that makes it easy to write simple tests
- **Fixture**: A way to provide test data or test resources
- **Async Test**: A test that uses Python's async/await syntax
- **Mark**: A way to categorize or mark tests with specific attributes

## Test Structure

The test suite is organized into several files:

1. `conftest.py`: Contains shared fixtures and configuration
2. `test_mock_llm_responses.py`: Tests for different types of responses
3. `test_mock_llm_edge_cases.py`: Tests for edge cases and error handling

### Key Components

#### Fixtures (`conftest.py`)
Fixtures are reusable pieces of test data or resources. For example:
```python
@pytest.fixture
def mock_llm(basic_response_map):
    """Fixture providing a basic MockLLM instance."""
    return MockLLM(basic_response_map)
```

This fixture creates a new MockLLM instance for each test that needs it.

#### Response Patterns
The test suite includes several types of response patterns:
- Basic responses (simple text responses)
- Template responses (dynamic responses with variables)
- Error responses (simulated error conditions)
- Complex responses (multi-step or conditional responses)

## Running Tests

### Basic Commands

1. Run all tests:
```bash
pytest tests/
```

2. Run with verbose output:
```bash
pytest tests/ -v
```

3. Run a specific test file:
```bash
pytest tests/test_mock_llm_responses.py
```

4. Run a specific test class:
```bash
pytest tests/test_mock_llm_responses.py::TestBasicResponses
```

5. Run tests with coverage report:
```bash
pytest tests/ --cov=tests
```

### Understanding Test Output

When you run tests, you'll see output like this:
```
============================= test session starts ==============================
platform darwin -- Python 3.9.0, pytest-6.1.2, py-1.9.0, pluggy-0.13.1
rootdir: /path/to/project
collected 25 items

tests/test_mock_llm_responses.py .......                                [ 28%]
tests/test_mock_llm_edge_cases.py .................                    [100%]

============================== 25 passed in 2.34s ==============================
```

- Each dot (.) represents a passed test
- The percentage shows progress
- The final line shows total tests and execution time

## Understanding Test Files

### Test Classes and Methods

Each test file contains classes that group related tests. For example:

```python
class TestBasicResponses:
    @pytest.mark.asyncio
    async def test_simple_responses(self, mock_llm):
        response = await mock_llm.generate("hello")
        assert response == "Hello, how can I help you?"
```

- `@pytest.mark.asyncio`: Marks the test as async
- `async def`: Defines an async test function
- `assert`: Checks if a condition is true

### Common Test Patterns

1. **Basic Response Test**:
```python
async def test_simple_responses(self, mock_llm):
    response = await mock_llm.generate("hello")
    assert response == "Hello, how can I help you?"
```

2. **Error Handling Test**:
```python
async def test_invalid_kwargs(self, mock_llm):
    response = await mock_llm.generate("test", temperature=2.0)
    assert response == "I am a mock LLM response."
```

3. **Concurrent Operation Test**:
```python
async def test_concurrent_generate(self, mock_llm):
    tasks = [
        mock_llm.generate("hello"),
        mock_llm.generate("error")
    ]
    responses = await asyncio.gather(*tasks)
    assert len(responses) == 2
```

## Test Categories

### 1. Basic Response Tests
- Tests simple text responses
- Verifies default behavior
- Checks file operation responses

### 2. Template Response Tests
- Tests dynamic responses
- Verifies count-based responses
- Checks time-based responses

### 3. Error Response Tests
- Tests timeout handling
- Verifies validation errors
- Checks permission errors

### 4. Edge Case Tests
- Tests empty prompts
- Verifies long prompt handling
- Checks special character handling

### 5. Concurrent Operation Tests
- Tests parallel requests
- Verifies history tracking
- Checks resource management

## Best Practices

1. **Test Organization**
   - Group related tests in classes
   - Use descriptive test names
   - Keep tests focused and simple

2. **Fixture Usage**
   - Use fixtures for common setup
   - Clean up resources after tests
   - Share fixtures across test files

3. **Async Testing**
   - Always use `@pytest.mark.asyncio`
   - Use `async/await` syntax
   - Handle concurrent operations properly

4. **Assertions**
   - Use specific assertions
   - Check both success and failure cases
   - Verify edge cases

5. **Test Maintenance**
   - Keep tests up to date
   - Document test purpose
   - Review test coverage regularly

## Troubleshooting

### Common Issues

1. **Test Not Found**
   - Check test file name (must start with `test_`)
   - Verify test function name (must start with `test_`)
   - Ensure test is in the correct directory

2. **Async Test Failures**
   - Verify `@pytest.mark.asyncio` is used
   - Check for proper `async/await` syntax
   - Ensure event loop is properly configured

3. **Fixture Errors**
   - Check fixture name matches
   - Verify fixture is properly defined
   - Ensure fixture is in `conftest.py` if shared

### Getting Help

- Use `pytest --help` for command options
- Check pytest documentation at https://docs.pytest.org/
- Review test output for detailed error messages 