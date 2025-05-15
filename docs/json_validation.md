# JSON Schema Validation

## Overview

The JSON validator (`src/json_validator_tool.py`) ensures that LLM responses conform to expected schemas and formats.

## Key Features

1. **Schema-Based Validation**
   - JSON Schema validation
   - Custom validation rules
   - Schema versioning
   - Error reporting

2. **Custom Rules**
   - Register custom validation functions
   - Combine with schema validation
   - Detailed error messages
   - Rule management

3. **Performance Optimization**
   - Validation result caching
   - Efficient schema processing
   - Batch validation support

## Usage Example

```python
from src.json_validator_tool import JSONValidator

# Initialize validator
validator = JSONValidator()

# Register schema
schema = {
    "type": "object",
    "properties": {
        "response": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
    },
    "required": ["response", "confidence"]
}
validator.register_schema("llm_response", schema)

# Register custom rule
def validate_confidence(value):
    return 0 <= value.get("confidence", 0) <= 1
validator.register_custom_rule("confidence_range", validate_confidence)

# Validate response
result = validator.validate(
    data=llm_response,
    schema_name="llm_response",
    custom_rules=["confidence_range"]
)
```

## Configuration

```python
validator_config = {
    "cache_size": 1000,           # Validation result cache size
    "strict_mode": True,          # Strict validation mode
    "error_limit": 10,            # Maximum validation errors
    "custom_rules": {             # Custom validation rules
        "confidence_range": validate_confidence
    }
}
```

## Best Practices

1. **Schema Validation**
   - Define clear schemas
   - Use custom rules for complex validation
   - Monitor validation errors
   - Update schemas as needed

2. **Error Handling**
   - Validate responses before use
   - Log validation errors
   - Monitor error patterns 