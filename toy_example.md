# Toy Example: Multi-Agent Text Processing System

## Overview
This is a simplified demonstration of a multi-agent system that processes text files and incorporates human feedback. The system uses multiple specialized agents to handle different aspects of text processing, analysis, and improvement.

## Features
- Multi-agent architecture with specialized roles
- Human feedback integration
- Content analysis and verification
- Structured logging system
- Error handling and recovery
- Performance tracking and optimization
- Post-feedback analysis and reporting

## Components

### Agents
1. **FileReaderAgent**
   - Reads and processes text files
   - Removes markdown formatting
   - Extracts relevant content
   - Logs operations

2. **WriterAgent**
   - Generates content based on input
   - Analyzes input content
   - Maintains context
   - Ensures quality

3. **InformationVerifierAgent**
   - Verifies information accuracy
   - Validates content
   - Identifies inconsistencies
   - Provides verification results

4. **TextQualityAgent**
   - Evaluates text quality
   - Checks grammar and style
   - Assesses readability
   - Provides quality scores

5. **RootCauseAnalyzerAgent**
   - Analyzes system behavior
   - Generates improvement reports
   - Identifies bottlenecks
   - Provides recommendations

## Agent Interaction Flow

### 1. Initialization
- Load configuration
- Initialize agents
- Setup logging
- Configure cache

### 2. Content Processing
- Read input document
- Generate content
- Verify information
- Check quality

### 3. Human Feedback Loop
- Request user feedback
- Process feedback
- Implement improvements
- Verify changes

### 4. Post-Feedback Analysis
- Analyze interaction flow
- Identify bottlenecks
- Generate improvement report
- Save recommendations

### 5. System Improvement Report
The system generates a comprehensive report after processing feedback, including:

#### Data Analysis
- Input quality metrics
- Processing efficiency
- Transformation accuracy
- Validation results

#### Agent Performance
- Response accuracy
- Processing speed
- Error rates
- Communication efficiency

#### System Optimization
- Resource utilization
- Processing bottlenecks
- Cache effectiveness
- API efficiency
- Memory usage patterns

#### Recommendations
- Agent behavior improvements
- Data processing enhancements
- System configuration updates
- Performance optimizations
- Error handling improvements

## Configuration
The system is configured through `toy_example.json`:

```json
{
    "agents": {
        "file_reader": { ... },
        "writer": { ... },
        "verifier": { ... },
        "quality_checker": { ... },
        "analyzer": { ... }
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": "toy_example.log"
    },
    "cache": {
        "enabled": true,
        "ttl": 3600,
        "max_size": 1000,
        "storage": {
            "type": "memory",
            "persistent": false
        }
    },
    "api": {
        "timeout": 30,
        "max_retries": 3,
        "retry_delay": 1
    },
    "performance": {
        "response_time_threshold": 5.0,
        "error_rate_threshold": 0.1,
        "cache_hit_ratio_threshold": 0.5,
        "memory_usage_threshold": 80
    }
}
```

## Logging System
The system uses a structured logging approach:
- Log level: INFO
- Format: Timestamp - Agent - Level - Message
- Output: Console and file (toy_example.log)
- Events: Operation start/complete, errors, metrics

## Usage

### Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

3. Configure system:
   - Edit `toy_example.json`
   - Adjust agent settings
   - Set performance thresholds

### Running
```bash
python toy_example.py
```

### Output
- Processed content
- Quality scores
- Verification results
- System improvement report (system_improvement_report.json)

## Error Handling
The system handles various types of errors:
- File access errors
- API communication errors
- Content processing errors
- Validation errors
- System resource errors

## Development

### Adding New Features
1. Create new agent class
2. Update configuration
3. Add to main workflow
4. Update documentation

### Testing
1. Unit tests for agents
2. Integration tests for workflow
3. Performance benchmarks
4. Error handling tests

## License
MIT License 