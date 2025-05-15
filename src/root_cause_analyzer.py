#!/usr/bin/env python3

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from .agent_tracer import AgentTracer, AgentEvent
from .analytics_assistant_agent import AnalyticsAssistantAgent

@dataclass
class RootCauseAnalysis:
    """Represents the result of a root cause analysis."""
    summary: str
    issues: List[Dict[str, Any]]
    recommendations: List[str]
    metadata: Optional[Dict[str, Any]] = None

class RootCauseAnalyzer:
    """Analyzes agent traces to identify root causes and provide recommendations."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RootCauseAnalyzer with configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary containing:
                - logging: Logging configuration
                - analysis: Analysis configuration
                - metrics: Metrics to track
        """
        self.config = config
        self.logger = logging.getLogger("root_cause_analyzer")
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Configure logging based on the configuration."""
        log_config = self.config.get("logging", {})
        self.logger.setLevel(getattr(logging, log_config.get("level", "INFO")))
        
        # Add file handler if specified
        if log_config.get("file"):
            file_handler = logging.FileHandler(log_config["file"])
            file_handler.setFormatter(logging.Formatter(log_config.get("format")))
            self.logger.addHandler(file_handler)
        
        # Add console handler if enabled
        if log_config.get("console", True):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(log_config.get("format")))
            self.logger.addHandler(console_handler)
    
    def analyze(self, tracer: AgentTracer, user_feedback: Optional[str] = None) -> RootCauseAnalysis:
        """
        Analyze the agent trace to identify root causes.
        
        Args:
            tracer (AgentTracer): The agent tracer containing the events to analyze
            user_feedback (Optional[str]): Optional user feedback to consider
            
        Returns:
            RootCauseAnalysis: Analysis results
        """
        self.logger.info("Starting root cause analysis")
        
        # Get events from tracer
        events = tracer.get_trace()
        
        # Analyze events
        issues = self._analyze_events(events)
        
        # Add user feedback analysis if provided
        if user_feedback:
            feedback_issues = self._analyze_user_feedback(user_feedback, events)
            issues.extend(feedback_issues)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues)
        
        # Create analysis summary
        summary = self._generate_summary(issues, recommendations)
        
        # Create analysis result
        analysis = RootCauseAnalysis(
            summary=summary,
            issues=issues,
            recommendations=recommendations,
            metadata={
                "total_events": len(events),
                "total_issues": len(issues),
                "total_recommendations": len(recommendations)
            }
        )
        
        self.logger.info("Root cause analysis completed")
        return analysis
    
    def _analyze_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze events to identify issues.
        
        Args:
            events (List[Dict[str, Any]]): List of agent events
            
        Returns:
            List[Dict[str, Any]]: List of identified issues
        """
        issues = []
        
        # Analyze processing times
        for event in events:
            if event["metadata"] and event["metadata"].get("processing_time", 0) > self.config.get("performance", {}).get("response_time_threshold", 5.0):
                issues.append({
                    "type": "performance",
                    "severity": "warning",
                    "description": f"Agent {event['agent_name']} took {event['metadata']['processing_time']:.2f}s to process messages",
                    "recommendation": "Consider optimizing agent processing or increasing timeout threshold"
                })
        
        # Analyze message patterns
        message_counts = {}
        for event in events:
            agent = event["agent_name"]
            message_counts[agent] = message_counts.get(agent, 0) + len(event["inputs"])
        
        for agent, count in message_counts.items():
            if count > self.config.get("analysis", {}).get("max_messages_per_agent", 10):
                issues.append({
                    "type": "communication",
                    "severity": "info",
                    "description": f"Agent {agent} processed {count} messages",
                    "recommendation": "Consider optimizing communication flow or increasing max_messages threshold"
                })
        
        # Analyze error patterns
        for event in events:
            for output in event["outputs"]:
                if isinstance(output["content"], str) and "error" in output["content"].lower():
                    issues.append({
                        "type": "error",
                        "severity": "error",
                        "description": f"Error detected in {event['agent_name']} output",
                        "details": output["content"],
                        "recommendation": "Review agent error handling and input validation"
                    })
        
        return issues
    
    def _analyze_user_feedback(self, feedback: str, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze user feedback in context of events.
        
        Args:
            feedback (str): User feedback
            events (List[Dict[str, Any]]): List of agent events
            
        Returns:
            List[Dict[str, Any]]: List of issues identified from feedback
        """
        issues = []
        
        # Add feedback as an issue
        issues.append({
            "type": "feedback",
            "severity": "info",
            "description": "User feedback received",
            "details": feedback,
            "recommendation": "Review feedback and consider implementing suggested improvements"
        })
        
        # Analyze feedback against events
        if "slow" in feedback.lower():
            issues.append({
                "type": "performance",
                "severity": "warning",
                "description": "User reported performance issues",
                "recommendation": "Review agent processing times and optimize where possible"
            })
        
        if "error" in feedback.lower() or "failed" in feedback.lower():
            issues.append({
                "type": "reliability",
                "severity": "error",
                "description": "User reported errors or failures",
                "recommendation": "Review error handling and add more robust validation"
            })
        
        return issues
    
    def _generate_recommendations(self, issues: List[Dict[str, Any]]) -> List[str]:
        """
        Generate recommendations based on identified issues.
        
        Args:
            issues (List[Dict[str, Any]]): List of identified issues
            
        Returns:
            List[str]: List of recommendations
        """
        recommendations = set()
        
        # Add recommendations from issues
        for issue in issues:
            if "recommendation" in issue:
                recommendations.add(issue["recommendation"])
        
        # Add general recommendations based on issue types
        issue_types = {issue["type"] for issue in issues}
        
        if "performance" in issue_types:
            recommendations.add("Consider implementing caching for frequently used data")
            recommendations.add("Review and optimize database queries if applicable")
        
        if "error" in issue_types:
            recommendations.add("Implement more comprehensive error handling")
            recommendations.add("Add input validation at all entry points")
        
        if "communication" in issue_types:
            recommendations.add("Review and optimize agent communication patterns")
            recommendations.add("Consider implementing message batching where appropriate")
        
        return list(recommendations)
    
    def _generate_summary(self, issues: List[Dict[str, Any]], recommendations: List[str]) -> str:
        """
        Generate a summary of the analysis.
        
        Args:
            issues (List[Dict[str, Any]]): List of identified issues
            recommendations (List[str]): List of recommendations
            
        Returns:
            str: Analysis summary
        """
        total_issues = len(issues)
        issue_types = {issue["type"] for issue in issues}
        severity_counts = {}
        
        for issue in issues:
            severity = issue["severity"]
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        summary = f"Analysis Summary:\n"
        summary += f"- Total Issues: {total_issues}\n"
        summary += f"- Issue Types: {', '.join(issue_types)}\n"
        summary += f"- Severity Breakdown: {', '.join(f'{k}: {v}' for k, v in severity_counts.items())}\n"
        summary += f"- Recommendations: {len(recommendations)}\n"
        
        return summary
    
    def save_analysis(self, analysis: RootCauseAnalysis, file_path: str) -> None:
        """
        Save the analysis results to a file.
        
        Args:
            analysis (RootCauseAnalysis): Analysis results
            file_path (str): Path to save the analysis
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(analysis), f, indent=2)
            self.logger.info(f"Analysis saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving analysis to {file_path}: {e}")
            raise 