# Testing and Extensibility

## Testing
- Tests are located in `tradeshow/tests/`
- Main test file: `test_synthetic_user_generation.py`
- Tests cover:
  - User generation (UserGeneratorAgent, outputs a strictly-typed `SyntheticUser` Pydantic model, **Temperature:** 0.7)
  - Validation (ValidatorAgent, outputs a strictly-typed `CriticOutput` Pydantic model, **Temperature:** 0.0)
  - Review (ReviewerAgent, returns a dict with an `update_synthetic_user` field containing a strictly-typed `SyntheticUser` Pydantic model, **Temperature:** 0.2)
  - Tracing (TracedGroupChat)
- All agent outputs are now structured using Pydantic models that exactly match the JSON schemas in `tradeshow/schema/`.
- At runtime, the segments definition file (`input/segments.json`) is validated against the schema in `schema/segmets_schema.json` before any processing begins. If validation fails, the program halts with a clear error message.
- Tests use static/mock logic for reproducibility
- Use `pytest` to run all tests: `pytest tradeshow/tests/`

## Model Temperature Rationale

| Agent     | Temperature | Rationale                                                                 |
|-----------|-------------|--------------------------------------------------------------------------|
| Generator | 0.7         | Balances creativity and coherence, producing varied yet plausible profiles.|
| Critic    | 0.0         | Ensures deterministic, focused analysis with minimal randomness.           |
| Reviewer  | 0.2         | Allows slight natural variation for precise rewriting while maintaining fidelity. |

See `docs/model_temperatures.md` for more details.

## Adding New Tests
- Add new test functions to `test_synthetic_user_generation.py` or create new test files
- Use pytest conventions for easy integration

## Extending the System
- To add new agent logic, create a new agent class in `src/`
- Update `config_agents.json` for new agent parameters or system messages
- To support new schemas or segments, update the input files in `input/`
- The system is modular and can be adapted for LLM integration, new validation rules, or richer user profiles

---
See other docs in `tradeshow/docs/` for architecture, workflow, and configuration details. 