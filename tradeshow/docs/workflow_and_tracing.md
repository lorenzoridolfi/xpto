# Workflow and Tracing in Synthetic User Generation

## Step-by-Step Workflow
1. **Initialization:**
   - Load main config, agent config, and user_id state
   - Load segments and schema
   - Validate the segments file (`segments.json`) against the schema (`segmets_schema.json`) at runtime before any processing. If validation fails, execution halts with a clear error message.
2. **For each segment:**
   - For the configured number of users:
     - UserGeneratorAgent creates a user, assigns a unique user_id, and outputs a strictly-typed `SyntheticUser` Pydantic model
     - ValidatorAgent checks the user for realism, consistency, and segment alignment, and outputs a strictly-typed `CriticOutput` Pydantic model
     - If invalid, ReviewerAgent reviews and suggests corrections, returning a dict with an `update_synthetic_user` field containing a strictly-typed `SyntheticUser` Pydantic model
     - All actions are logged
3. **Output:**
   - All users are saved in a single JSON file (e.g., `output/synthetic_users.json`)
   - All trace logs are saved in a single JSON file (e.g., `logs/trace_run.json`)
   - The user_id state is updated in `config_agents_state.json`

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