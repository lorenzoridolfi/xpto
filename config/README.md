# Configuration Structure

This directory contains all configuration files for the project, organized into three main categories:

## Shared Configuration (`config/shared/`)

Configuration files used by both programs:

- `global_settings.json`: Global system settings and configurations
- `base_config.json`: Basic system configuration
- `agent_settings.json`: Shared agent configurations
- `agent_validation_schema.json`: Validation schemas for agent configurations
- `logging_settings.json`: Logging system configuration

## Toy Example Configuration (`config/toy_example/`)

Configuration files specific to the toy example program:

- `program_config.json`: Main configuration for the toy example program
  - Contains task description, agent hierarchy, and configuration for text processing
  - Includes specific agent configurations for text analysis and human feedback

## Update Manifest Configuration (`config/update_manifest/`)

Configuration files specific to the update manifest program:

- `program_config.json`: Main configuration for the update manifest program
  - Contains task description, agent hierarchy, and configuration for manifest updates
  - Includes specific configurations for manifest and logging system updates
- `manifest_validation_schema.json`: Validation schema for manifest files

## Usage

Each program should load its specific configuration from its respective directory, along with the shared configurations from the `shared` directory.

### Example Usage

```python
# Loading shared configurations
with open('config/shared/global_settings.json') as f:
    global_config = json.load(f)
with open('config/shared/base_config.json') as f:
    base_config = json.load(f)

# Loading program-specific configuration
with open('config/toy_example/program_config.json') as f:
    program_config = json.load(f)
```

## Configuration Dependencies

1. Both programs depend on:
   - `global_settings.json`
   - `base_config.json`
   - `agent_settings.json`
   - `agent_validation_schema.json`
   - `logging_settings.json`

2. Toy Example additionally uses:
   - `toy_example/program_config.json`

3. Update Manifest additionally uses:
   - `update_manifest/program_config.json`
   - `update_manifest/manifest_validation_schema.json` 