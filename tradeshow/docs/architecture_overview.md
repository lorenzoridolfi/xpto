# Synthetic User Generation System: Architecture Overview

## Purpose and Goals
This system generates synthetic users for market segments using a multi-agent, traceable workflow. The goals are:
- High-quality, realistic synthetic user data
- Full traceability and auditability of all actions
- Easy extensibility and testability
- Robust configuration and state management

## High-Level Workflow
1. Load configuration, agent settings, and state
2. For each segment, generate the configured number of users (one per call)
3. Validate each user against the schema
4. Review and correct users if validation fails
5. Log all actions and save outputs

## Agent Roles
- **UserGeneratorAgent:** Generates a realistic individual synthetic user profile for a randomly chosen Brazilian financial segment, ensuring internal consistency, plausibility, and clear segment alignment. Returns a strictly-typed `SyntheticUser` Pydantic model.
- **ValidatorAgent:** Evaluates a single synthetic user profile for realism, internal consistency, and fidelity to its stated Brazilian financial segment. Returns a strictly-typed `CriticOutput` Pydantic model.
- **ReviewerAgent:** Reviews and quality-assures synthetic user profiles using the segment definition and critic feedback, ensuring realism, coherence, and alignment. Returns a dict with an `update_synthetic_user` field containing a strictly-typed `SyntheticUser` Pydantic model.
- **TracedGroupChat:** Logs all actions/messages for auditability
- **Orchestrator:** Coordinates the workflow

## Structured Outputs and Validation
- All agent outputs are now structured using Pydantic models that exactly match the JSON schemas in `tradeshow/schema/`.
- The `UserGeneratorAgent` outputs a `SyntheticUser` model.
- The `ValidatorAgent` outputs a `CriticOutput` model.
- The `ReviewerAgent` returns an `update_synthetic_user` field using the `SyntheticUser` model.
- At runtime, the segments definition file (`input/segments.json`) is validated against the schema in `schema/segmets_schema.json` before any processing begins. If validation fails, the program halts with a clear error message.

## Configuration and State Management
- **config.json:** Main workflow configuration (input/output paths, users per segment, etc.)
- **config_agents.json:** Agent-specific configuration (user_id field, temperature, descriptions, system messages)
- **config_agents_state.json:** Tracks the last used user_id for sequential assignment

## Folder Structure
```
tradeshow/
  config.json
  config_agents.json
  config_agents_state.json
  input/
    segments.json
    synthetic_user_schema.json
  output/
    synthetic_users.json
  logs/
    trace_run.json
  docs/
    ... (markdown docs)
  tests/
    test_synthetic_user_generation.py
  src/
    synthetic_user_generator.py
```

## Extensibility and Testability
- Modular agent classes for easy extension
- All configuration and state externalized in JSON
- Tests use static/mock logic for reproducibility
- System can be adapted for new schemas, segments, or agent logic

---
See other docs in `tradeshow/docs/` for details on workflow, agent configuration, and usage. 