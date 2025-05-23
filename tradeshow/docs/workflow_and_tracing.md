# Workflow and Tracing Guide

This document describes the synthetic user generation workflow and tracing system.

## Workflow Overview

The synthetic user generation process follows these steps:

1. **Initialization**
   - Load configurations and schemas
   - Validate segments.json against schema
   - Initialize tracing system
   - Set up agent instances

2. **Segment Processing**
   - For each segment in segments.json:
     - Validate segment fields
     - Process requested number of users

3. **User Generation Cycle**
   ```mermaid
   sequenceDiagram
       participant O as Orchestrator
       participant G as UserGeneratorAgent
       participant V as ValidatorAgent
       participant R as ReviewerAgent
       
       O->>G: Generate User
       G-->>O: SyntheticUser
       O->>V: Validate User
       V-->>O: CriticOutput
       alt validation failed
           O->>R: Review User
           R-->>O: Updated SyntheticUser
       end
       O->>O: Save User
   ```

## Structured Data Flow

### 1. Input Validation
```json
{
  "segmentos": [
    {
      "nome": "string",
      "descricao": "string",
      "num_usuarios": "number",
      "atributos": []
    }
  ]
}
```

### 2. User Generation
```python
SyntheticUser(
    user_id="string",
    segment_label={"value": "string"},
    monthly_income={"value": float},
    # ... other fields
)
```

### 3. Validation
```python
CriticOutput(
    score=float,
    issues=["string"],
    recommendation="string"
)
```

## Tracing System

The `TracedGroupChat` class provides comprehensive tracing:

### 1. Trace Entry Structure
```json
{
  "message": "string",
  "activity": "string",
  "agent": {
    "name": "string",
    "description": "string",
    "temperature": number
  },
  "data": {},
  "llm_input": [],
  "llm_output": {}
}
```

### 2. Traced Activities

- User Generation
  ```json
  {
    "activity": "generate_user",
    "data": {"SyntheticUser": {}},
    "llm_input": ["messages"],
    "llm_output": {"user": {}}
  }
  ```

- Validation
  ```json
  {
    "activity": "validate_user",
    "data": {
      "user": {},
      "critic_output": {}
    }
  }
  ```

- Review
  ```json
  {
    "activity": "review_user",
    "data": {
      "update_synthetic_user": {}
    }
  }
  ```

### 3. Error Tracing

- Schema Validation Errors
  ```json
  {
    "activity": "schema_validation",
    "message": "Validation error",
    "data": {"error": "details"}
  }
  ```

- Pydantic Validation Errors
  ```json
  {
    "activity": "pydantic_validation",
    "message": "Model validation error",
    "data": {"error": "details"}
  }
  ```

## Monitoring and Debugging

1. **Log File Location**
   - Default: `tradeshow/logs/trace.json`
   - Configurable in `config.json`

2. **Log Analysis**
   - Track agent performance
   - Monitor validation rates
   - Identify common issues

3. **Error Handling**
   - Schema validation errors
   - Pydantic model errors
   - LLM response errors

4. **Performance Metrics**
   - Generation time per user
   - Validation success rate
   - Review frequency

## Best Practices

1. **Tracing**
   - Log all agent activities
   - Include context in messages
   - Track data transformations

2. **Error Handling**
   - Validate early
   - Provide clear messages
   - Log error context

3. **Data Flow**
   - Use Pydantic models
   - Validate at boundaries
   - Maintain type safety

4. **Monitoring**
   - Review logs regularly
   - Track error patterns
   - Monitor performance

## Step-by-Step Workflow
1. **Initialization:**
   - Load main config, agent config, and user_id state
   - Load segments and schema
   - Validate the segments file (`segments.json`) against the schema (`segmets_schema.json`) at runtime before any processing. If validation fails, execution halts with a clear error message.
2. **For each segment:**
   - For the configured number of users:
     - UserGeneratorAgent creates a user, assigns a unique user_id, and outputs a strictly-typed `SyntheticUser` Pydantic model (**Temperature:** 0.7)
     - ValidatorAgent checks the user for realism, consistency, and segment alignment, and outputs a strictly-typed `CriticOutput` Pydantic model (**Temperature:** 0.0)
     - If invalid, ReviewerAgent reviews and suggests corrections, returning a dict with an `update_synthetic_user` field containing a strictly-typed `SyntheticUser` Pydantic model (**Temperature:** 0.2)
     - All actions are logged
3. **Output:**
   - All users are saved in a single JSON file (e.g., `output/synthetic_users.json`)
   - All trace logs are saved in a single JSON file (e.g., `logs/trace_run.json`)
   - The user_id state is updated in `config_agents_state.json`

## Model Temperature Rationale

| Agent     | Temperature | Rationale                                                                 |
|-----------|-------------|--------------------------------------------------------------------------|
| Generator | 0.7         | Balances creativity and coherence, producing varied yet plausible profiles.|
| Critic    | 0.0         | Ensures deterministic, focused analysis with minimal randomness.           |
| Reviewer  | 0.2         | Allows slight natural variation for precise rewriting while maintaining fidelity. |

See `docs/model_temperatures.md` for more details.

## Structured Outputs and Validation
- All agent outputs are now structured using Pydantic models that exactly match the JSON schemas in `tradeshow/schema/`.
- The `UserGeneratorAgent` outputs a `SyntheticUser` model.
- The `ValidatorAgent` outputs a `CriticOutput` model.
- The `ReviewerAgent` returns an `update_synthetic_user` field using the `SyntheticUser` model.
- At runtime, the segments definition file (`input/segments.json`) is validated against the schema in `schema/segmets_schema.json` before any processing begins. If validation fails, the program halts with a clear error message.

## Agent Interaction and Orchestration
- The Orchestrator manages the workflow, instantiates agents, and loops as needed
- All agent actions and messages are routed through the TracedGroupChat for consistent logging

## Tracing and Auditability
- Every step, decision, and correction is logged in the trace file
- The trace includes agent names, actions, data, and outcomes
- The trace file can be reviewed to audit the generation process or debug issues

## Output and Trace Files
- **Synthetic users:** `output/synthetic_users.json`
- **Trace log:** `logs/trace_run.json`
- **User ID state:** `config_agents_state.json`

---
See `architecture_overview.md` and `agent_configuration.md` for more details. 