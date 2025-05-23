# Testing and Extensibility Guide

This document outlines the testing strategy and extensibility features of the synthetic user generation system.

## Testing Overview

The system includes comprehensive tests covering all major components:

### Unit Tests

1. **Agent Tests**
   - `test_user_generator_agent`: Verifies `SyntheticUser` model output
   - `test_validator_agent`: Tests `CriticOutput` model validation
   - `test_reviewer_agent`: Checks improved user generation
   - `test_user_id_sequential`: Ensures sequential ID generation

2. **Schema Validation Tests**
   - `test_segments_schema_validation`: Validates segments.json
   - `test_segments_json_nickname_and_usercount`: Checks segment fields

3. **Orchestrator Tests**
   - `test_orchestrator_respects_num_usuarios`: Verifies user count
   - Tests proper agent interaction and data flow

4. **Tracing Tests**
   - `test_traced_group_chat`: Validates logging functionality
   - Ensures proper activity tracking

### Integration Tests

1. **End-to-End Workflow**
   - Complete user generation pipeline
   - Schema validation
   - Agent interactions
   - Output validation

2. **Data Flow Tests**
   - Pydantic model transformations
   - Inter-agent communication
   - State management

## Extensibility Features

### 1. Pydantic Models

```python
# Add new fields to SyntheticUser
class SyntheticUser(BaseModel):
    new_field: Dict[str, Any]  # Add new attributes

# Extend CriticOutput
class CriticOutput(BaseModel):
    custom_metrics: Dict[str, float]  # Add custom validation
```

### 2. Agent Customization

- Modify agent configurations in `config_agents.json`
- Update system messages in `agents_update.json`
- Add new validation rules
- Customize temperature settings

### 3. Schema Extensions

- Extend segments schema for new attributes
- Add custom validation rules
- Modify output formats

### 4. Logging and Tracing

- Add custom log entries
- Track new metrics
- Extend tracing context

## Adding New Features

1. **New Agent Types**
   - Implement agent class
   - Add configuration
   - Update orchestrator
   - Add tests

2. **Custom Validation**
   - Extend Pydantic models
   - Add validation rules
   - Update agent logic
   - Add test cases

3. **New Output Formats**
   - Define Pydantic models
   - Update agent outputs
   - Add format validation
   - Test transformations

## Best Practices

1. **Testing**
   - Write tests for new features
   - Update existing tests
   - Maintain test coverage
   - Document test cases

2. **Documentation**
   - Update relevant docs
   - Add usage examples
   - Document configurations
   - Explain changes

3. **Code Quality**
   - Follow type hints
   - Use Pydantic validation
   - Handle errors gracefully
   - Maintain consistency

4. **Performance**
   - Consider caching needs
   - Optimize validation
   - Monitor resource usage
   - Profile changes

---
See other docs in `tradeshow/docs/` for architecture, workflow, and configuration details. 