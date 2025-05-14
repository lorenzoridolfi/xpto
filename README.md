# Update Manifest System

A three-agent system for creating, validating, and managing text content. The system uses three specialized agents: one for creating the manifest, another for criticizing it, and a manager to orchestrate the process.

## Features

- Three-agent architecture for manifest creation, validation, and management
- Human-in-the-loop feedback system
- Quality control and validation
- Comprehensive logging system
- Error handling and recovery
- Performance tracking

## System Components

### Agents

- **Creator Agent**: Creates and generates the manifest content
  - Analyzes input files
  - Generates structured content
  - Ensures completeness and accuracy
  - Maintains consistent formatting
  - Logs all creation steps in DEBUG level

- **Critic Agent**: Reviews and improves the manifest
  - Validates content accuracy
  - Suggests improvements
  - Ensures quality standards
  - Provides detailed feedback
  - Logs all validation steps in DEBUG level

- **Manager Agent**: Orchestrates the entire process
  - Coordinates between Creator and Critic agents
  - Manages workflow and timing
  - Handles error recovery
  - Ensures process completion
  - Logs all coordination steps in DEBUG level

### Logging System

The system uses a simple and efficient logging system that:
- Logs to both console and file
- Uses DEBUG level for detailed troubleshooting
- Includes timestamps and log levels
- Captures all system events and user interactions
- Provides structured logging for analysis

#### DEBUG Level Logging Details

The system logs all agent operations in DEBUG level, including:

**Creator Agent Logs:**
- File analysis steps
- Content generation process
- Structure validation
- Formatting decisions
- Error handling details

**Critic Agent Logs:**
- Content validation steps
- Improvement suggestions
- Quality check results
- Feedback generation
- Error detection details

**Manager Agent Logs:**
- Workflow coordination
- Agent communication
- Process state changes
- Error recovery attempts
- Performance metrics

## Configuration

The system is configured through `update_manifest_config.json`, which includes:

- Agent configurations and settings
- File manifest settings
- Logging configuration
- Metadata requirements
- Output format specifications

## Usage

1. Configure the system in `update_manifest_config.json`
2. Run the main script:
   ```bash
   python update_manifest.py
   ```
3. The Manager Agent coordinates the process:
   - Initiates the Creator Agent to process input files
   - Triggers the Critic Agent for review
   - Manages the feedback loop
4. Review the results and provide additional feedback if needed

## Logging

The system logs all operations to `update_manifest.log` with the following information:
- Timestamps for all events
- Agent operations and interactions
- Content creation and validation steps
- User inputs and feedback
- System events and errors
- Detailed DEBUG level information for all agent activities

## Error Handling

The system includes comprehensive error handling:
- Graceful error recovery
- Detailed error logging
- User-friendly error messages
- Automatic retry mechanisms
- Error analysis and reporting

## Development

To contribute to the project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
