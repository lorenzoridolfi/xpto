# Synthetic User Generation: Architecture & Code Overview

## Overview & Goals
The synthetic user generation system is designed to create, validate, and review synthetic user data for testing, simulation, or research. The architecture emphasizes modularity, extensibility, robust tracing, and testability.

## Main Components

### 1. Orchestrator
- **Role:** Coordinates the entire workflow: loads configuration, iterates over segments, spawns agents, manages output, and handles tracing.
- **Key Responsibilities:**
  - Reads config, agent config, and schema files
  - Instantiates agents and passes dependencies (including tracer)
  - Handles append/overwrite logic for output files
  - Aggregates and saves results

### 2. Agents
- **UserGeneratorAgent:** Generates a synthetic user for a segment using an LLM.
- **ValidatorAgent:** Validates a synthetic user and produces a critique (CriticOutput).
- **ReviewerAgent:** Reviews a user and critique, producing a final reviewed user.
- **Design:**
  - Each agent receives all dependencies via constructor (including tracer)
  - LLM calls are abstracted for easy mocking in tests
  - Agents log all significant actions/events via the tracer

### 3. Tracing & Logging
- **TracingMixin:** A reusable mixin providing event logging and trace file saving. Any class can inherit this to gain tracing capabilities.
- **TracedGroupChat:** Inherits from TracingMixin; used as the main tracer in the workflow. All agents and the orchestrator use this instance to log events.
- **Trace Events:** All major actions (user generation, validation, review, errors) are logged with rich metadata for auditability and debugging.

## Dependency Injection & Modularity
- All major dependencies (tracer, config, state, schema) are injected via constructors.
- No global state or hard-coded dependencies; this enables easy testing, extension, and reuse.
- The tracer is any object with a `.log()` method, not a specific class, allowing for flexible tracing strategies.

## LLM Integration & Mocking
- LLM calls are encapsulated in agent methods and can be mocked in tests to avoid real API calls.
- Tests use patching to simulate LLM responses and error conditions, ensuring agents and tracing logic are exercised without external dependencies.

## Testing Strategy & Coverage
- **Unit tests:** For TracingMixin, event logging, and error handling.
- **Integration tests:** For agent/tracer interaction, orchestrator workflow, and trace file content.
- **Error/edge case tests:** Simulate LLM failures and validate that errors are logged.
- **No unnecessary mocking:** Only external APIs (LLMs) are mocked; all tracing and workflow logic is tested for real.

## Extensibility & Best Practices
- **Add new agents:** Subclass or follow the agent interface, inject dependencies, and use the tracer for logging.
- **Change tracing:** Swap in any tracer with a `.log()` method (e.g., for database, remote logging, etc.).
- **Add new event types:** Extend the event structure in TracingMixin or TracedGroupChat.
- **Maintain test coverage:** Add tests for new agents, workflows, or trace event types.

## Recent Architectural Improvements
- Unified, dependency-injected tracing for all agents and orchestrator.
- Removal of shortcuts and global state; all logic is explicit and testable.
- Comprehensive tests for agent/tracer integration and error handling.
- Modular, reusable tracing logic via TracingMixin.

## Example: Agent Construction & Tracing
```python
tracer = TracedGroupChat(log_path="trace.json")
agent = UserGeneratorAgent(
    segment=segment,
    agent_config=agent_config,
    agent_state=agent_state,
    user_id_field="user_id",
    schema=schema,
    tracer=tracer,
)
user = agent.generate_user()
tracer.save()
```

## Recommendations for Future Development
- Consider adding async support for agent/orchestrator workflows for scalability.
- Add richer trace event schemas for analytics and debugging.
- Integrate with external logging/monitoring systems if needed.
- Continue to enforce dependency injection and modularity for all new code.

---
**This architecture ensures robust, auditable, and extensible synthetic user generation, ready for production and research use.** 