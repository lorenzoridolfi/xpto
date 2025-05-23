# Workflow and Tracing in Synthetic User Generation

## Step-by-Step Workflow
1. **Initialization:**
   - Load main config, agent config, and user_id state
   - Load segments and schema
2. **For each segment:**
   - For the configured number of users:
     - UserGeneratorAgent creates a user, assigns a unique user_id
     - ValidatorAgent checks the user against the schema
     - If invalid, ReviewerAgent reviews and suggests corrections
     - All actions are logged
3. **Output:**
   - All users are saved in a single JSON file (e.g., `output/synthetic_users.json`)
   - All trace logs are saved in a single JSON file (e.g., `logs/trace_run.json`)
   - The user_id state is updated in `config_agents_state.json`

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