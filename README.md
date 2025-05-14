# Multi-Agent Human Feedback System

A multi-agent system for processing and analyzing text content with human feedback, featuring advanced caching, manifest management, and quality control.

## Features

- **Multi-Agent Architecture**: Specialized agents for reading, writing, verification, and quality control
- **Human-in-the-Loop**: Interactive feedback and improvement cycles
- **Advanced Caching**: LLM response caching with configurable similarity thresholds
- **Manifest Management**: Structured file tracking with metadata and validation
- **Quality Control**: Multiple layers of content verification and improvement
- **Analytics**: Detailed performance metrics and root cause analysis
- **Error Handling**: Comprehensive error handling and recovery
- **Logging**: Detailed logging of all operations and decisions

## System Components

### Agents

1. **CoordinatorAgent**: Orchestrates the workflow between agents
2. **FileReaderAgent**: Reads and processes input files
3. **WriterAgent**: Generates content based on input
4. **InformationVerifierAgent**: Validates information accuracy
5. **TextQualityAgent**: Ensures content quality
6. **RootCauseAnalyzerAgent**: Analyzes feedback and system behavior

### Cache System

The system includes an advanced LLM cache with the following features:
- **Similarity-based Caching**: Matches similar queries using configurable thresholds
- **GPU Acceleration**: Optional GPU support for faster processing
- **Configurable Parameters**:
  - `similarity_threshold`: 0.99 (for safer caching)
  - `max_size`: Maximum number of cached items
  - `expiration_hours`: Cache entry lifetime
  - `language`: Language-specific processing (e.g., "portuguese")

### Manifest System

The system uses a structured manifest to track files and their metadata:

#### Schema Features

- **Version Control**: Semantic versioning for manifest compatibility
- **File Tracking**:
  - Basic info (filename, path, status)
  - Metadata (summary, keywords, topics, entities)
  - Technical details (hash, size, type, encoding)
  - Dependencies and categories
- **Global Metadata**: Cross-file topics and entities
- **Statistics**: File counts, sizes, and update tracking

#### File Selection

The manifest supports flexible file selection using glob patterns:
- **Basic Patterns**:
  - `*.txt` - All text files
  - `data/*.csv` - CSV files in data directory
  - `[0-9]*.log` - Log files starting with numbers
- **Recursive Patterns**:
  - `**/*.json` - JSON files in any subdirectory
  - `docs/**/*.md` - Markdown files in docs and subdirectories
- **Multiple Patterns**: Combine different patterns in the manifest

### Configuration

The system is configured through JSON files:

1. **Main Configuration** (`toy_example.json`):
   - Task description
   - Agent hierarchy
   - LLM settings
   - Cache parameters
   - File manifest

2. **Manifest Configuration** (`update_manifest_config.json`):
   - File patterns
   - Metadata requirements
   - Output format
   - Processing rules

3. **Schema Definition** (`manifest_schema.json`):
   - File structure
   - Required fields
   - Data types
   - Validation rules

## Usage

1. **Setup Configuration**:
   ```json
   {
     "file_manifest": [
       {
         "filename": "*.txt",
         "description": "Text files"
       },
       {
         "filename": "data/**/*.json",
         "description": "JSON data files"
       }
     ],
     "cache_config": {
       "similarity_threshold": 0.99,
       "max_size": 1000,
       "expiration_hours": 24,
       "language": "portuguese"
     }
   }
   ```

2. **Generate Manifest**:
   ```bash
   python update_manifest.py
   ```

3. **Run System**:
   ```bash
   python toy_example.py
   ```

## Error Handling

The system includes comprehensive error handling:
- **Manifest Validation**: Schema-based validation
- **File Processing**: Graceful handling of missing or invalid files
- **Cache Management**: Automatic cleanup and recovery
- **Backup System**: Automatic manifest backups with versioning

## Logging

The system provides detailed logging:
- **File Operations**: Track all file reads and writes
- **Agent Actions**: Record agent decisions and interactions
- **Cache Statistics**: Monitor cache performance
- **Error Tracking**: Detailed error logging with context

## Performance

The system includes several performance optimizations:
- **Caching**: Configurable LLM response caching
- **Parallel Processing**: Support for concurrent file processing
- **Incremental Updates**: Efficient manifest updates
- **Resource Management**: Automatic cleanup of old data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
