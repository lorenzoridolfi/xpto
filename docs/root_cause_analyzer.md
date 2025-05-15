# Root Cause Analyzer Agent

## Overview

The `RootCauseAnalyzerAgent` is a specialized agent that inherits from `AnalyticsAssistantAgent` to provide advanced analytics and root cause analysis capabilities. It combines configuration data, user feedback, and system logs to provide insights into system behavior and potential improvements.

## Features

### Analytics Capabilities
- Inherits all analytics features from `AnalyticsAssistantAgent`
- Advanced metrics collection and analysis
- Performance monitoring and optimization
- Tool usage analytics

### Root Cause Analysis
- Structured analysis of system issues
- Detailed recommendations
- Confidence scoring
- Performance optimization suggestions

### Interaction Analysis
- Agent interaction pattern analysis
- Communication efficiency metrics
- Workflow optimization suggestions
- Collaboration effectiveness assessment

### System Improvement
- Comprehensive improvement reports
- Resource utilization analysis
- Processing bottleneck identification
- Cache effectiveness evaluation
- API efficiency metrics
- Memory usage optimization

## Methods

### analyze_interaction_flow
```python
async def analyze_interaction_flow(self, metrics: Dict[str, Any]) -> Dict[str, Any]
```
Analyzes the interaction patterns between agents and provides insights for optimization.

#### Parameters
- `metrics`: Dictionary containing metrics from all agents

#### Returns
Dictionary containing:
- Interaction patterns
- Communication efficiency
- Workflow optimization
- Agent collaboration metrics

### generate_improvement_report
```python
async def generate_improvement_report(self, data: Dict[str, Any]) -> Dict[str, Any]
```
Generates a comprehensive system improvement report.

#### Parameters
- `data`: Dictionary containing system metrics and analysis data

#### Returns
Dictionary containing:
- Data analysis
- Agent performance metrics
- System optimization suggestions
- Detailed recommendations

### analyze
```python
async def analyze(self, input_data: RootCauseInput) -> Dict[str, Any]
```
Main analysis method that combines all features.

#### Parameters
- `input_data`: RootCauseInput instance containing analysis data

#### Returns
Dictionary containing:
- Root causes
- Recommendations
- Confidence score

### _calculate_confidence_score
```python
def _calculate_confidence_score(self, report: Dict[str, Any]) -> float
```
Calculates confidence score based on report data.

#### Parameters
- `report`: Dictionary containing analysis report

#### Returns
Float representing confidence score (0.0 to 1.0)

## Usage Example

```python
# Initialize the agent
root_cause_analyzer = RootCauseAnalyzerAgent(config)

# Prepare input data
input_data = RootCauseInput(
    config=config,
    user_feedback="System performance issues",
    action_log=[...],
    event_log=[...]
)

# Perform analysis
analysis = await root_cause_analyzer.analyze(input_data)

# Access results
root_causes = analysis["root_causes"]
recommendations = analysis["recommendations"]
confidence = analysis["confidence_score"]
```

## Output Format

### Root Causes
```json
{
    "root_causes": [
        {
            "description": "string",
            "severity": "high|medium|low",
            "evidence": ["string"],
            "affected_components": ["string"]
        }
    ]
}
```

### Recommendations
```json
{
    "recommendations": [
        {
            "description": "string",
            "priority": "high|medium|low",
            "implementation_steps": ["string"]
        }
    ]
}
```

### Improvement Report
```json
{
    "data_analysis": {
        "input_quality": {
            "score": "number",
            "issues": ["string"],
            "recommendations": ["string"]
        },
        "processing_efficiency": {
            "score": "number",
            "bottlenecks": ["string"],
            "optimizations": ["string"]
        }
    },
    "agent_performance": {
        "response_accuracy": {
            "overall_score": "number",
            "agent_scores": {
                "agent_name": "number"
            }
        }
    },
    "system_optimization": {
        "resource_utilization": {
            "score": "number",
            "issues": ["string"],
            "recommendations": ["string"]
        }
    }
}
```

## Dependencies

- AnalyticsAssistantAgent
- OpenAIChatCompletionClient
- LLMCache
- ToolAnalytics
- ToolUsageMetrics

## Configuration

The agent requires the following configuration:

```json
{
    "cache": {
        "enabled": true
    },
    "analytics": {
        "enabled": true,
        "metrics": ["performance", "usage", "errors"]
    }
}
``` 