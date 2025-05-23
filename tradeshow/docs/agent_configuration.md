# Agent Configuration for Synthetic User Generation

## Purpose
Agent configuration allows you to control the behavior, parameters, and metadata for each agent in the synthetic user generation workflow. This enables easy tuning, documentation, and extension of agent logic.

## Structure of `config_agents.json`
- `user_id_field`: The field name in the synthetic user JSON that holds the unique sequential user ID
- `UserGeneratorAgent`, `ValidatorAgent`, `ReviewerAgent`: Each has
  - `temperature`: LLM temperature (for future use or LLM integration)
  - `description`: Human-readable description of the agent's role
  - `system_message`: System prompt/message for the agent (for LLM or documentation)

## Example
```json
{
  "user_id_field": "user_id",
  "UserGeneratorAgent": {
    "temperature": 0.7,
    "description": "Generates a synthetic user for a given market segment, using segment attributes and context to create realistic, diverse user profiles. Each user is assigned a unique sequential user_id.",
    "system_message": "You are a Synthetic User Generator Agent. Your task is to create a realistic synthetic user profile for the given market segment, using the provided segment description and attributes. Ensure the user is plausible, diverse, and fits the segment context. Assign a unique user_id to each user you generate."
  },
  "ValidatorAgent": {
    "temperature": 0.3,
    "description": "Validates the structure and content of a synthetic user profile against the provided JSON Schema.",
    "system_message": "You are a Validator Agent. Your job is to check if the synthetic user profile conforms to the provided JSON Schema. If the user is invalid, provide a clear error message."
  },
  "ReviewerAgent": {
    "temperature": 0.3,
    "description": "Reviews and critiques synthetic user profiles that fail validation, suggesting corrections or improvements.",
    "system_message": "You are a Reviewer Agent. When a synthetic user profile fails validation, analyze the error and suggest corrections or improvements."
  }
}
```

## Updating and Using the Configuration
- Edit `config_agents.json` to change agent parameters, descriptions, or system messages
- The field specified by `user_id_field` will be used for sequential user IDs
- The current value of the user ID is tracked in `config_agents_state.json` and updated automatically

---
See `architecture_overview.md` for how this fits into the overall system. 