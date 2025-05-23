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
- Validates the segments file (`segments.json`) against the schema (`segmets_schema.json`) at runtime before any processing. If validation fails, execution halts with a clear error message.

### 2. User Generator Agent
- **Description:** Generates a realistic individual synthetic user profile for a randomly chosen Brazilian financial segment, ensuring internal consistency, plausibility, and clear segment alignment.
- **System Message:** You are the Synthetic User Generator. Each time, you must produce one coherent, believable profile of a Brazilian individual belonging to one of six financial segments (Planejadores, Poupadores, Materialistas, Batalhadores, Céticos, Endividados). Randomly select the segment (without mentioning your choice process) and then:
  - Start with a line "Segment: <SegmentName>".
  - Provide structured details (name, age, education, occupation, income, family status, etc.)
  - Describe financial behaviors (saving, spending, investment, bank usage, credit/debt)
  - Explain motivations and attitudes toward money
  - All details must cohere with the chosen segment's known traits, be internally consistent, and grounded in a Brazilian context. Do not mention this is generated or describe your process—present it as a factual profile.
- **Temperature:** 0.7 (see `docs/model_temperatures.md`)
- **Output:** Returns a strictly-typed `SyntheticUser` Pydantic model.

### 3. Validator Agent
- **Description:** Evaluates a single synthetic user profile for realism, internal consistency, and fidelity to its stated Brazilian financial segment.
- **System Message:** You are the Synthetic User Critic. You receive one profile (including its "Segment: <SegmentName>" line and structured details) plus the segment definitions. Perform checks for segment alignment, internal consistency, realism, and outliers. Output a JSON object with `score`, `issues`, and `recommendation`.
- **Temperature:** 0.0 (see `docs/model_temperatures.md`)
- **Output:** Returns a strictly-typed `CriticOutput` Pydantic model.

### 4. Reviewer Agent
- **Description:** The Reviewer Agent is responsible for quality-assuring synthetic user profiles in a multi-agent AutoGen workflow. It reviews each generated profile against the target segment's definition and the critic agent's feedback. The reviewer ensures the profile is realistic, internally consistent, and aligned with the segment's philosophy, demographics, and financial behaviors. Its ultimate goal is to refine or regenerate the profile (if needed) while preserving the original persona's intent, delivering a polished profile that appears correct from the start.
- **System Message:** You are a Reviewer Agent in a Microsoft AutoGen multi-agent setup. Your role is to validate and improve synthetic user profiles generated for specific market segments. You will receive three inputs: (1) a synthetic user profile draft, (2) the assigned segment's definition, and (3) a structured critique from a critic agent (including a score from 0–1, a list of issues, and a recommendation of "accept" or "flag for review"). Follow the instructions to produce the final profile output, ensuring alignment, coherence, and realism, and output only the improved profile data.
- **Temperature:** 0.2 (see `docs/model_temperatures.md`)
- **Output:** Returns a dict with an `update_synthetic_user` field containing a strictly-typed `SyntheticUser` Pydantic model.

### 5. Trace Collector Agent
- Collects and logs all actions, messages, and decisions from all agents.
- Ensures a complete trace of the workflow for auditability.

### 6. Orchestrator (Traced Group Chat)
- Coordinates the workflow, ensuring each agent performs its role in sequence.
- Loops the user generation process until the desired number of valid users per segment is reached.
- All agent interactions are routed through the Traced Group Chat for consistent logging.

## Structured Outputs and Validation
- All agent outputs are now structured using Pydantic models that exactly match the JSON schemas in `tradeshow/schema/`.
- The `UserGeneratorAgent` outputs a `SyntheticUser` model.
- The `ValidatorAgent` outputs a `CriticOutput` model.
- The `ReviewerAgent` returns an `update_synthetic_user` field using the `SyntheticUser` model.
- At runtime, the segments definition file (`input/segments.json`) is validated against the schema in `schema/segmets_schema.json` before any processing begins. If validation fails, the program halts with a clear error message.

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

## Model Temperature Rationale

| Agent     | Temperature | Rationale                                                                 |
|-----------|-------------|--------------------------------------------------------------------------|
| Generator | 0.7         | Balances creativity and coherence, producing varied yet plausible profiles.|
| Critic    | 0.0         | Ensures deterministic, focused analysis with minimal randomness.           |
| Reviewer  | 0.2         | Allows slight natural variation for precise rewriting while maintaining fidelity. |

See `docs/model_temperatures.md` for more details.

---
For more details, see the implementation and tests under the `tradeshow/` directory. 