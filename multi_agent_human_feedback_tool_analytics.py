"""
Tool Analytics and Optimization System

This module provides advanced analytics and optimization for agent tool usage,
including detailed reflection criteria, usage optimization, and performance tracking.
"""

import json
import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
from abc import ABC, abstractmethod

@dataclass
class BaseMetrics:
    """Base class for all metrics tracking."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    average_response_time: float = 0.0
    success_rate: float = 0.0

@dataclass
class ToolUsageMetrics(BaseMetrics):
    """Metrics for tracking tool usage effectiveness."""
    impact_score: float = 0.0
    context_relevance: float = 0.0

class BaseAnalytics(ABC):
    """Base class for all analytics systems."""
    
    def __init__(self):
        self.metrics = defaultdict(BaseMetrics)
        self.usage_patterns = defaultdict(list)
        self.performance_history = []
        
    @abstractmethod
    def update_metrics(self, *args, **kwargs):
        """Update metrics for an interaction."""
        pass
        
    @abstractmethod
    def analyze_usage(self, *args, **kwargs) -> Dict[str, Any]:
        """Analyze usage patterns and effectiveness."""
        pass
        
    @abstractmethod
    def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """Get optimization suggestions based on usage patterns."""
        pass
        
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Generate a performance dashboard."""
        return {
            "overall_metrics": {
                "total_items": len(self.metrics),
                "total_calls": sum(m.total_calls for m in self.metrics.values()),
                "average_success_rate": sum(m.success_rate for m in self.metrics.values()) / len(self.metrics) if self.metrics else 0
            },
            "item_metrics": {
                name: {
                    "total_calls": metrics.total_calls,
                    "success_rate": metrics.success_rate,
                    "average_response_time": metrics.average_response_time
                }
                for name, metrics in self.metrics.items()
            },
            "usage_patterns": self.usage_patterns
        }

class ToolAnalytics(BaseAnalytics):
    """Analytics system for tracking and optimizing tool usage."""
    
    def __init__(self):
        super().__init__()
        self.metrics = defaultdict(ToolUsageMetrics)
        self.optimization_suggestions = []
        
    def update_metrics(self, tool_name: str, success: bool, response_time: float,
                      impact_score: float, context_relevance: float):
        """Update metrics for a tool usage."""
        metrics = self.metrics[tool_name]
        metrics.total_calls += 1
        if success:
            metrics.successful_calls += 1
        else:
            metrics.failed_calls += 1
            
        # Update averages
        metrics.average_response_time = (
            (metrics.average_response_time * (metrics.total_calls - 1) + response_time)
            / metrics.total_calls
        )
        metrics.success_rate = metrics.successful_calls / metrics.total_calls
        metrics.impact_score = impact_score
        metrics.context_relevance = context_relevance
        
    def analyze_usage(self, tool_calls: List[Dict]) -> Dict[str, Any]:
        """Analyze tool usage patterns and effectiveness."""
        analysis = {
            "tool_usage_patterns": [],
            "effectiveness_metrics": {},
            "optimization_suggestions": [],
            "context_analysis": {}
        }
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            metrics = self.metrics[tool_name]
            
            # Analyze usage pattern
            pattern = {
                "tool": tool_name,
                "success_rate": metrics.success_rate,
                "average_response_time": metrics.average_response_time,
                "impact_score": metrics.impact_score,
                "context_relevance": metrics.context_relevance
            }
            analysis["tool_usage_patterns"].append(pattern)
            
            # Generate optimization suggestions
            if metrics.success_rate < 0.7:
                analysis["optimization_suggestions"].append({
                    "tool": tool_name,
                    "suggestion": "Consider alternative approaches or tool combinations",
                    "reason": f"Low success rate: {metrics.success_rate:.2%}"
                })
                
        return analysis
        
    def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """Get optimization suggestions based on usage patterns."""
        suggestions = []
        for tool_name, metrics in self.metrics.items():
            if metrics.success_rate < 0.7:
                suggestions.append({
                    "tool": tool_name,
                    "type": "success_rate",
                    "suggestion": "Consider using alternative tools or approaches",
                    "metrics": {
                        "success_rate": metrics.success_rate,
                        "average_response_time": metrics.average_response_time
                    }
                })
            if metrics.context_relevance < 0.6:
                suggestions.append({
                    "tool": tool_name,
                    "type": "context_relevance",
                    "suggestion": "Improve context awareness when using this tool",
                    "metrics": {
                        "context_relevance": metrics.context_relevance
                    }
                })
        return suggestions
        
    def generate_reflection_prompt(self, tool_calls: List[Dict]) -> str:
        """Generate a detailed reflection prompt for tool usage."""
        analysis = self.analyze_usage(tool_calls)
        
        prompt = f"""
        Analyze the following tool usage and provide comprehensive insights:
        
        Tool Calls:
        {json.dumps(tool_calls, indent=2)}
        
        Usage Analysis:
        {json.dumps(analysis, indent=2)}
        
        Consider the following aspects:
        
        1. Effectiveness:
           - Success rate of each tool
           - Impact on the final output
           - Response time and efficiency
        
        2. Context Awareness:
           - Was each tool used at the optimal time?
           - Did the tool usage align with the task context?
           - Were there better alternatives available?
        
        3. Error Analysis:
           - What went wrong in failed tool calls?
           - How could errors have been prevented?
           - What patterns emerge in successful vs failed calls?
        
        4. Learning Points:
           - What worked well and should be repeated?
           - What patterns should be avoided?
           - How can tool usage be optimized?
        
        5. Optimization Opportunities:
           - Could the task have been completed with fewer tool calls?
           - Are there more efficient tool combinations?
           - How can tool usage be streamlined?
        
        Provide a detailed reflection focusing on actionable improvements and specific recommendations.
        """
        
        return prompt
        
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Generate a performance dashboard for tool usage."""
        dashboard = super().get_performance_dashboard()
        dashboard.update({
            "tool_metrics": {
                name: {
                    "total_calls": metrics.total_calls,
                    "success_rate": metrics.success_rate,
                    "average_response_time": metrics.average_response_time,
                    "impact_score": metrics.impact_score,
                    "context_relevance": metrics.context_relevance
                }
                for name, metrics in self.metrics.items()
            },
            "optimization_suggestions": self.get_optimization_suggestions()
        })
        return dashboard 