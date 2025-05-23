# Agent Configuration Guide

This document describes the configuration and behavior of the agents in the synthetic user generation system.

## Agent Overview

The system uses three specialized agents, each with specific roles and structured outputs:

### UserGeneratorAgent

- **Description**: Generates synthetic user profiles based on segment data, ensuring adherence to the user schema and segment characteristics.
- **Output**: Returns a `SyntheticUser` Pydantic model
- **Configuration**:
  ```json
  {
    "temperature": 0.7,
    "model": "gpt-4o",
    "description": "Agent responsible for generating synthetic user profiles that match segment characteristics and schema requirements",
    "system_message": "You are an expert in creating realistic synthetic user profiles..."
  }
  ```

### ValidatorAgent

- **Description**: Validates synthetic users for realism, internal consistency, and alignment with their intended segment.
- **Output**: Returns a `CriticOutput` Pydantic model
- **Configuration**:
  ```json
  {
    "temperature": 0.2,
    "model": "gpt-4o",
    "description": "Agent responsible for validating synthetic users for realism and consistency",
    "system_message": "You are an expert validator analyzing synthetic user profiles..."
  }
  ```

### ReviewerAgent

- **Description**: Reviews and improves synthetic users that fail validation, incorporating validator feedback.
- **Output**: Returns a dict with `update_synthetic_user` field containing a `SyntheticUser` model
- **Configuration**:
  ```json
  {
    "temperature": 0.3,
    "model": "gpt-4o",
    "description": "Agent responsible for reviewing and improving synthetic users based on validation feedback",
    "system_message": "You are an expert reviewer improving synthetic user profiles..."
  }
  ```

## Configuration Files

### config_agents.json

Contains the configuration for all agents:
```json
{
  "UserGeneratorAgent": {
    // Agent-specific configuration
  },
  "ValidatorAgent": {
    // Agent-specific configuration
  },
  "ReviewerAgent": {
    // Agent-specific configuration
  },
  "user_id_field": "user_id"
}
```

### agents_update.json

Contains updated agent descriptions and system messages:
```json
{
  "UserGeneratorAgent": {
    "description": "...",
    "system_message": "..."
  },
  "ValidatorAgent": {
    "description": "...",
    "system_message": "..."
  },
  "ReviewerAgent": {
    "description": "...",
    "system_message": "..."
  }
}
```

## Structured Outputs

### SyntheticUser Model

```python
class SyntheticUser(BaseModel):
    user_id: str
    segment_label: Dict[str, str]
    philosophy: Dict[str, str]
    monthly_income: Dict[str, float]
    education_level: Dict[str, str]
    occupation: Dict[str, str]
    uses_traditional_bank: Dict[str, bool]
    uses_digital_bank: Dict[str, bool]
    uses_broker: Dict[str, bool]
    savings_frequency_per_month: Dict[str, float]
    spending_behavior: Dict[str, str]
    investment_behavior: Dict[str, str]
```

### CriticOutput Model

```python
class CriticOutput(BaseModel):
    score: float
    issues: List[str]
    recommendation: str
```

## Agent Interaction Flow

1. **UserGeneratorAgent**:
   - Receives segment data
   - Generates `SyntheticUser` instance
   - Ensures all required fields are present

2. **ValidatorAgent**:
   - Receives `SyntheticUser` instance
   - Returns `CriticOutput` with validation results
   - Recommendation can be "accept" or "reject"

3. **ReviewerAgent**:
   - Receives `SyntheticUser` and `CriticOutput`
   - Returns improved `SyntheticUser` instance
   - Addresses issues identified in validation

## Configuration Best Practices

1. **Temperature Settings**:
   - UserGeneratorAgent: 0.7 (balance creativity and consistency)
   - ValidatorAgent: 0.2 (consistent validation)
   - ReviewerAgent: 0.3 (focused improvements)

2. **Model Selection**:
   - Use GPT-4 for all agents
   - Enable response format validation
   - Disable caching for synthetic user generation

3. **System Messages**:
   - Keep focused on specific agent role
   - Include validation criteria
   - Specify output requirements

4. **Error Handling**:
   - Use Pydantic validation
   - Provide clear error messages
   - Log validation failures

---
See `architecture_overview.md` for how this fits into the overall system. 

---

**Note:**
- The temperature values for each agent are chosen to optimize their specific roles. See `docs/model_temperatures.md` for the rationale behind each value. 