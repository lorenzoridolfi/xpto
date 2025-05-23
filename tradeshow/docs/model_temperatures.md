# Model Temperatures for Synthetic User Generation Agents

Each agent in the synthetic user generation workflow uses a distinct temperature value for the LLM model, chosen to optimize its specific role:

| Agent               | Temperature | Rationale                                                                 |
|---------------------|-------------|--------------------------------------------------------------------------|
| UserGeneratorAgent  | 0.7         | Balances creativity and coherence, producing varied yet plausible profiles.|
| ValidatorAgent      | 0.0         | Ensures deterministic, focused analysis with minimal randomness.           |
| ReviewerAgent       | 0.2         | Allows slight natural variation for precise rewriting while maintaining fidelity. |

## Rationale

- **UserGeneratorAgent (0.7):**
  - A higher temperature encourages the LLM to generate more diverse and creative outputs, which is important for producing a wide range of plausible synthetic user profiles. However, the value is not so high as to risk incoherence or implausibility.

- **ValidatorAgent (0.0):**
  - A temperature of 0.0 makes the LLM deterministic, ensuring that its analysis and validation of user profiles are consistent, focused, and free from randomness. This is crucial for reliable, repeatable validation.

- **ReviewerAgent (0.2):**
  - A low but nonzero temperature allows the LLM to make precise, nuanced corrections or improvements to user profiles, introducing just enough variation to avoid robotic or overly rigid rewriting, while still maintaining fidelity to the original intent and structure.

These values are set in `config_agents.json` and referenced throughout the documentation and codebase. 