# Text Processing System with Human Feedback

A multi-agent system for processing text files, removing markdown formatting, and generating comprehensive metadata through collaborative agent interactions.

## Features

- Multi-agent system with specialized roles
- Comprehensive text processing and analysis
- Human feedback integration
- Detailed logging system
- Backup management
- File manifest generation and validation

## Components

### Agents

The system employs three specialized agents that work together to process and analyze text files:

1. **Creator Agent**
   - Primary content generator
   - Analyzes text files and creates initial manifest
   - Performs deep content analysis
   - Extracts key information
   - Structures data according to specified format
   - Ensures metadata accuracy and completeness

2. **Critic Agent**
   - Quality control specialist
   - Reviews and validates Creator's output
   - Performs detailed analysis of metadata
   - Identifies potential issues
   - Suggests improvements
   - Ensures content meets quality standards

3. **Manager Agent**
   - Process orchestrator
   - Coordinates workflow between Creator and Critic
   - Ensures timely completion of tasks
   - Handles error recovery
   - Resolves conflicts
   - Makes final decisions on content quality

### Agent Interaction Flow

The agents interact in a structured workflow:

1. **Initialization**
   - Manager initiates the process
   - Creator receives file list and requirements
   - Critic prepares validation criteria

2. **Content Generation**
   - Creator analyzes each text file
   - Generates initial metadata including:
     - Summaries (max 200 words)
     - Keywords (3-10 terms)
     - Topics (1-5 main topics)
     - Entities (2-8 relevant entities)
   - Structures data in JSON format

3. **Validation Phase**
   - Critic reviews Creator's output
   - Validates against requirements:
     - Summary accuracy and completeness
     - Keyword relevance and format
     - Topic hierarchy and coverage
     - Entity accuracy and categorization
   - Provides detailed feedback

4. **Improvement Cycle**
   - Creator receives Critic's feedback
   - Makes necessary improvements
   - Resubmits for validation
   - Process repeats until quality standards are met

5. **Final Approval**
   - Manager reviews final version
   - Ensures all requirements are met
   - Approves for manifest generation
   - Coordinates final output

### Configuration

The system is configured through `update_manifest_config.json`:

```json
{
  "llm_config": {
    "creator": { ... },
    "validator": { ... },
    "manager": { ... }
  },
  "agents": {
    "creator": { ... },
    "validator": { ... },
    "manager": { ... }
  },
  "metadata_requirements": { ... },
  "output_format": { ... }
}
```

### Logging System

The system implements comprehensive logging:

- Log level: DEBUG
- Output: Both file and console
- Format: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`
- Log file: `update_manifest.log`

### Backup System

Automatic backup management:

- Enabled by default
- Directory: `manifest_backups`
- Maximum backups: 5
- Timestamp format: `%Y%m%d_%H%M%S`
- Automatic cleanup of old backups

## Usage

1. Ensure all required dependencies are installed
2. Set up the OpenAI API key in environment variables
3. Configure the system in `update_manifest_config.json`
4. Run the script:
   ```bash
   python update_manifest.py
   ```

## Error Handling

The system includes comprehensive error handling:

- File access and permission errors
- JSON parsing and validation errors
- API communication errors
- Backup and restore errors
- Agent interaction errors

## Development

### Adding New Features

1. Update the configuration file
2. Modify agent behaviors as needed
3. Add new validation rules
4. Update the manifest schema

### Testing

1. Run unit tests
2. Verify agent interactions
3. Check error handling
4. Validate output format

## License

MIT License
