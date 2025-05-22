# Synthetic User Generation: One-User-Per-Call Approach

## Overview
This document describes the architecture and workflow for generating synthetic users for market segments using a multi-agent system with robust tracing. The system is designed to generate **one synthetic user per LLM call**, ensuring high quality, traceability, and auditability. All components, configuration, and outputs are organized under the `tradeshow` directory.

## Rationale: One User Per Call
- **Quality Control:** Each user is validated and reviewed immediately after generation, minimizing hallucinations and schema violations.
- **Traceability:** Every generation step is logged, making the process transparent and auditable.
- **Iterative Improvement:** Errors or feedback can be addressed for each user individually, improving overall data quality.
- **Resource Efficiency:** LLMs are more reliable when focused on a single, well-defined output.

## Agent Roles

### 1. Input Loader Agent
- Loads market segment descriptions and the current JSON Schema for synthetic users from the `tradeshow/input/` folder.

### 2. User Generator Agent
- For each segment, generates **one synthetic user per call** using the segment's context and schema.
- Receives instructions from the Orchestrator and outputs a candidate user.

### 3. Validator Agent
- Validates the generated user against the loaded JSON Schema.
- Returns validation results and error details if the user does not conform.

### 4. Reviewer Agent
- Reviews users that fail validation, providing actionable feedback or corrections.
- Can be rule-based or LLM-based, depending on requirements.

### 5. Trace Collector Agent
- Collects and logs all actions, messages, and decisions from all agents.
- Ensures a complete trace of the workflow for auditability.

### 6. Orchestrator (Traced Group Chat)
- Coordinates the workflow, ensuring each agent performs its role in sequence.
- Loops the user generation process until the desired number of valid users per segment is reached.
- All agent interactions are routed through the Traced Group Chat for consistent logging.

## Tracing and the Trace Agent Chat
- **All agent actions and messages are routed through the Traced Group Chat.**
- The Trace Collector Agent logs:
  - Agent names and roles
  - Descriptions of each action
  - System messages and decisions
  - Validation and review outcomes
- The trace is saved to `tradeshow/logs/` with metadata for later analysis and audit.
- This approach ensures that every step, decision, and correction is fully transparent and reproducible.

## Folder Structure
```
tradeshow/
  config.json                # Configuration (paths, user counts, etc.)
  input/
    segments.json            # Market segment descriptions
    synthetic_user_schema.json # JSON Schema for users
  output/
    synthetic_users_segment_<name>.json # Output users
  logs/
    trace_<timestamp>.json   # Full trace of each run
  docs/
    synthetic_user_generation.md # This documentation
  tests/
    test_synthetic_user_generation.py # Tests
```

## Workflow Summary
1. Orchestrator loads segment and schema data.
2. For each segment, the User Generator Agent is called to create one user.
3. The Validator Agent checks the user against the schema.
4. If validation fails, the Reviewer Agent suggests corrections.
5. The process repeats until the required number of valid users is generated for each segment.
6. All steps are traced and saved for auditability.

## Benefits
- **Minimizes hallucination and schema drift**
- **Maximizes transparency and auditability**
- **Easily extensible and testable**

---
For more details, see the implementation and tests under the `tradeshow/` directory. 