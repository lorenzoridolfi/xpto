# Human Feedback Agents System

This document describes the multi-agent system for processing and analyzing human feedback, implemented in `src/agents_human_feedback.py`.

## Overview

The Human Feedback Agents System provides a framework for:
- Processing user feedback
- Analyzing system behavior
- Generating improvement recommendations
- Managing configuration versions
- Handling file operations asynchronously

## Key Components

### Configuration Management

```python
class ConfigVersion:
    """Represents a versioned configuration with timestamp."""
```

Features:
- Semantic versioning
- Timestamp tracking
- Configuration backup
- Update handling

### File Operations

```python
class FileReaderAgent:
    """Handles asynchronous file reading operations with retry logic and error handling."""
```

Features:
- Asynchronous file reading
- Retry logic
- Error handling
- Progress tracking

### Root Cause Analysis

```python
class RootCauseAnalyzerAgent(AnalyticsAssistantAgent):
    """Analyzes system behavior and user feedback to identify root causes and recommendations."""
```

Features:
- Interaction flow analysis
- Improvement report generation
- Confidence scoring
- Issue categorization

### Error Handling

```python
@dataclass
class FileReadError:
    """Represents an error that occurred during file reading."""

@dataclass
class AgentError:
    """Represents an error that occurred during agent operation."""
```

Features:
- Structured error representation
- Severity levels
- Context preservation
- Timestamp tracking

## Usage Example

```python
async def main():
    # Load configuration
    config, version = load_config_with_versioning(
        "config.json",
        default_config={
            "file_manifest": ["file1.txt", "file2.txt"],
            "max_retries": 3,
            "retry_delay": 1
        }
    )
    
    # Initialize agents
    file_reader = FileReaderAgent(config)
    analyzer = RootCauseAnalyzerAgent(config)
    
    # Process files
    file_contents = await file_reader.process_files()
    
    # Analyze feedback
    analysis_input = RootCauseInput(
        config=config,
        user_feedback="User feedback text",
        action_log=[],
        event_log=[]
    )
    
    results = await analyzer.analyze(analysis_input)
```

## Configuration Structure

```json
{
    "file_manifest": [
        "string"
    ],
    "max_retries": "number",
    "retry_delay": "number",
    "analysis": {
        "confidence_threshold": "number",
        "max_iterations": "number"
    },
    "logging": {
        "level": "string",
        "file": "string",
        "format": "string"
    }
}
```

## Best Practices

1. **Configuration Management**
   - Use semantic versioning
   - Create backups before updates
   - Validate configuration changes
   - Document version history

2. **File Operations**
   - Use asynchronous reading
   - Implement retry logic
   - Handle errors gracefully
   - Track operation progress

3. **Analysis**
   - Validate input data
   - Calculate confidence scores
   - Categorize issues
   - Generate actionable recommendations

4. **Error Handling**
   - Use structured error types
   - Include context information
   - Track error severity
   - Log error details

## Integration

To integrate the Human Feedback Agents System:

1. **Setup**
   - Import required components
   - Configure file manifest
   - Set up logging
   - Initialize agents

2. **Implementation**
   - Process user feedback
   - Analyze system behavior
   - Generate reports
   - Handle errors

3. **Configuration**
   - Define file manifest
   - Set retry parameters
   - Configure analysis settings
   - Set up logging

4. **Testing**
   - Test file operations
   - Test analysis logic
   - Test error handling
   - Test configuration management

## Maintenance

Regular maintenance tasks:

1. **Configuration**
   - Review version history
   - Update default settings
   - Clean up old backups
   - Document changes

2. **File Operations**
   - Monitor read performance
   - Review error patterns
   - Update retry logic
   - Clean up old files

3. **Analysis**
   - Review confidence scores
   - Update analysis rules
   - Improve recommendations
   - Track effectiveness

4. **Error Handling**
   - Review error patterns
   - Update error types
   - Improve error messages
   - Monitor error rates 