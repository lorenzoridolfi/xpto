# Testing and Extensibility

## Testing
- Tests are located in `tradeshow/tests/`
- Main test file: `test_synthetic_user_generation.py`
- Tests cover:
  - User generation (UserGeneratorAgent)
  - Validation (ValidatorAgent)
  - Review (ReviewerAgent)
  - Tracing (TracedGroupChat)
- Tests use static/mock logic for reproducibility
- Use `pytest` to run all tests: `pytest tradeshow/tests/`

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