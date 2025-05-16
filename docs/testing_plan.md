# Testing Plan

## Overview
This plan outlines a systematic approach to testing the framework, starting with unit tests and then moving to testing the `update_manifest` application as an integration test.

## MockLLM Test Structure

### MockLLM Test Components
1. **Response Patterns** (defined in `conftest.py`):
   - `BASIC_RESPONSES`: Simple text responses
   - `TEMPLATE_RESPONSES`: Dynamic response templates
   - `ERROR_RESPONSES`: Error handling patterns
   - `COMPLEX_RESPONSES`: Multi-step and conditional responses

2. **Test Fixtures**:
   - `basic_response_map`: Basic response patterns
   - `template_response_map`: Template response patterns
   - `error_response_map`: Error response patterns
   - `complex_response_map`: Complex response patterns
   - `mock_llm`: Basic MockLLM instance
   - `dynamic_mock_llm`: DynamicMockLLM instance
   - `error_mock_llm`: Error-focused MockLLM
   - `complex_mock_llm`: Complex response MockLLM
   - `mock_llm_with_history`: MockLLM with pre-populated history
   - `dynamic_mock_llm_with_counts`: DynamicMockLLM with counts

## Test Order and Dependencies

### Level 1: Basic Tests (No Dependencies)
1. Basic Response Tests (`TestBasicResponses`)
   - Simple text responses
   - File operations
   - Default behavior
   - Uses: `basic_response_map`, `mock_llm` fixtures

2. Template Response Tests (`TestTemplateResponses`)
   - Count-based responses
   - Time-based responses
   - Length-based responses
   - Uses: `template_response_map`, `dynamic_mock_llm` fixtures

3. Error Response Tests (`TestErrorResponses`)
   - Timeout handling
   - Validation errors
   - Permission errors
   - Uses: `error_response_map`, `error_mock_llm` fixtures

4. Edge Case Tests (`TestEdgeCases`)
   - Empty prompts
   - Long prompts
   - Special characters
   - Unicode characters
   - Uses: `mock_llm` fixture

5. Concurrent Operation Tests (`TestConcurrentOperations`)
   - Parallel requests
   - History tracking
   - Uses: `mock_llm_with_history` fixture

### Level 2: Agent Integration Tests (Depends on Basic Tests)
6. Agent Integration Tests (`TestAgentIntegration`)
   - Agent initialization
   - Agent communication
   - Task handling
   - Concurrent operations
   - Uses: `mock_llm`, `complex_mock_llm` fixtures

7. Agent Error Handling Tests (`TestAgentErrorHandling`)
   - Timeout handling
   - Invalid input handling
   - Uses: `error_mock_llm` fixture

8. Agent Dynamic Response Tests (`TestAgentDynamicResponses`)
   - Dynamic counting
   - Timing responses
   - Uses: `dynamic_mock_llm_with_counts` fixture

### Level 3: Advanced Tests (Depends on Agent Integration)
9. Agent Collaboration Tests (`TestAgentCollaboration`)
   - Complete workflows
   - Consensus building
   - Agent coordination
   - Uses: `complex_mock_llm` fixture

10. Agent State Persistence Tests (`TestAgentStatePersistence`)
    - State save/load
    - State updates
    - State clearing
    - Uses: `mock_llm_with_history` fixture

## Running Tests

### Option 1: Run All Tests in Order
```bash
pytest tests/ -v
```
This will run all tests in the correct order, respecting dependencies.

### Option 2: Run Specific Test Levels
```bash
# Run Level 1 tests (Basic MockLLM tests)
pytest tests/test_mock_llm_responses.py::TestBasicResponses tests/test_mock_llm_responses.py::TestTemplateResponses tests/test_mock_llm_responses.py::TestErrorResponses tests/test_mock_llm_edge_cases.py::TestEdgeCases tests/test_mock_llm_edge_cases.py::TestConcurrentOperations -v

# Run Level 2 tests (Agent Integration with MockLLM)
pytest tests/test_mock_llm_autogen_integration.py::TestAgentIntegration tests/test_mock_llm_autogen_integration.py::TestAgentErrorHandling tests/test_mock_llm_autogen_integration.py::TestAgentDynamicResponses -v

# Run Level 3 tests (Advanced MockLLM scenarios)
pytest tests/test_mock_llm_autogen_integration.py::TestAgentCollaboration tests/test_mock_llm_autogen_integration.py::TestAgentStatePersistence -v
```

### Option 3: Run with Dependencies
```bash
# Run a specific test and its dependencies
pytest tests/test_mock_llm_autogen_integration.py::TestAgentCollaboration -v --deps
```

## Success Criteria

### Level 1 Tests (MockLLM Basic)
- All basic response tests pass
- Template responses work correctly
- Error handling functions properly
- Edge cases are handled correctly
- Concurrent operations work reliably
- Response patterns match expected formats
- History tracking works correctly

### Level 2 Tests (MockLLM with Agents)
- Agent initialization works with MockLLM
- Agent communication functions with MockLLM
- Error handling works in agent context
- Dynamic responses work with agents
- Response patterns are correctly applied
- Agent state is maintained

### Level 3 Tests (MockLLM Advanced)
- Agent collaboration works with MockLLM
- State persistence functions correctly
- Complex workflows execute properly
- Response patterns are correctly sequenced
- Agent interactions are properly tracked

## Troubleshooting

### Common Issues

1. **Test Dependencies**
   - Check if required tests have passed
   - Verify test order is correct
   - Check for circular dependencies
   - Verify fixture availability

2. **Test Failures**
   - Review test output for specific failures
   - Check if dependencies are satisfied
   - Verify test data is correct
   - Check response pattern matches

3. **MockLLM Specific Issues**
   - Verify response patterns are defined
   - Check fixture initialization
   - Verify history tracking
   - Check concurrent operation handling

### Getting Help
- Review test output
- Check error messages
- Consult testing guide
- Review test dependencies
- Check MockLLM documentation

## Next Steps

After completing this testing plan:

1. Review test coverage report
2. Document any issues found
3. Create additional tests if needed
4. Plan integration tests for other applications
5. Consider adding performance benchmarks
6. Review and update response patterns

## Phase 2: Update Manifest Application Test

### Step 1: Setup
1. Create a test manifest file:
```bash
echo '{"version": "1.0.0", "dependencies": []}' > test_manifest.json
```

2. Create a test update file:
```bash
echo '{"version": "1.0.1", "dependencies": ["new-package"]}' > test_update.json
```

### Step 2: Run Update Manifest
1. Run the update_manifest application:
```bash
python update_manifest.py test_manifest.json test_update.json
```

### Step 3: Verify Results
1. Check the updated manifest file:
```bash
cat test_manifest.json
```
- Verify version is updated
- Check dependencies are added
- Ensure JSON format is valid

### Step 4: Test Error Cases
1. Test with invalid manifest:
```bash
echo 'invalid json' > invalid_manifest.json
python update_manifest.py invalid_manifest.json test_update.json
```
- Verify proper error handling
- Check error messages

2. Test with missing files:
```bash
python update_manifest.py nonexistent.json test_update.json
```
- Verify file not found handling
- Check error messages

## Success Criteria

### Unit Tests
- All unit tests pass
- No skipped tests
- No warnings
- Test coverage > 80%

### Update Manifest Application
- Successfully updates manifest file
- Handles errors gracefully
- Maintains JSON format
- Preserves existing data

### Autogen Agent Integration
- Agents initialize correctly with mock LLM
- Agent communication works as expected
- Error handling functions properly
- Dynamic responses are generated correctly
- Concurrent operations work reliably

### Agent Collaboration
- Agents work together in workflows
- Consensus building functions properly
- Coordination between agents works
- Specialized roles perform as expected

### Agent State Management
- State save/load operations work
- State updates are maintained
- State clearing functions properly
- State persists between operations

## Troubleshooting

### Common Issues

1. **Unit Test Failures**
   - Check test output for specific failures
   - Verify test data is correct
   - Check for async/await syntax

2. **Update Manifest Issues**
   - Verify file permissions
   - Check JSON format
   - Ensure file paths are correct

### Getting Help
- Review test output
- Check error messages
- Consult testing guide
- Review update_manifest documentation

## Next Steps

After completing this testing plan:

1. Review test coverage report
2. Document any issues found
3. Create additional tests if needed
4. Plan integration tests for other applications 