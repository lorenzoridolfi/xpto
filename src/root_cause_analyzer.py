"""
Root Cause Analyzer Module

This module provides functionality for analyzing agent traces to identify root causes
of issues and generate recommendations. It includes:
- Event analysis
- User feedback analysis
- Recommendation generation
- Analysis reporting

The analyzer helps identify performance issues, communication patterns, and error
conditions in the agent system.
"""

import json
import logging
from typing import Dict, List, Any, Optional, TypedDict, Set
from dataclasses import dataclass, asdict
from pathlib import Path
from .agent_tracer import AgentTracer, AgentEvent
from .analytics_assistant_agent import AnalyticsAssistantAgent


# Custom Exceptions
class AnalysisError(Exception):
    """Base exception for analysis-related errors."""

    pass


class AnalysisConfigurationError(AnalysisError):
    """Raised when analysis configuration is invalid."""

    pass


class AnalysisProcessingError(AnalysisError):
    """Raised when analysis processing fails."""

    pass


class AnalysisSaveError(AnalysisError):
    """Raised when saving analysis results fails."""

    pass


# Type Definitions
class AnalysisConfig(TypedDict):
    """Type definition for analysis configuration."""

    logging: Dict[str, Any]
    analysis: Dict[str, Any]
    metrics: Dict[str, Any]
    performance: Dict[str, float]


class Issue(TypedDict):
    """Type definition for analysis issue."""

    type: str
    severity: str
    description: str
    details: Optional[str]
    recommendation: Optional[str]


class AnalysisMetadata(TypedDict):
    """Type definition for analysis metadata."""

    total_events: int
    total_issues: int
    total_recommendations: int


@dataclass
class RootCauseAnalysis:
    """Represents the result of a root cause analysis.

    This class encapsulates the results of a root cause analysis, including
    identified issues, recommendations, and associated metadata.

    Attributes:
        summary: Summary of the analysis
        issues: List of identified issues
        recommendations: List of recommendations
        metadata: Optional metadata about the analysis
    """

    summary: str
    issues: List[Issue]
    recommendations: List[str]
    metadata: Optional[AnalysisMetadata] = None


class RootCauseAnalyzer:
    """Analyzes agent traces to identify root causes and provide recommendations.

    This class provides functionality for analyzing agent traces to identify
    performance issues, communication patterns, and error conditions. It generates
    recommendations based on the analysis and can save results to a file.
    """

    def __init__(self, config: AnalysisConfig) -> None:
        """Initialize the RootCauseAnalyzer with configuration.

        Args:
            config: Configuration dictionary containing logging, analysis,
                   metrics, and performance settings

        Raises:
            AnalysisConfigurationError: If configuration is invalid
        """
        try:
            self.config = config
            self.logger = logging.getLogger("root_cause_analyzer")
            self._setup_logging()
        except Exception as e:
            raise AnalysisConfigurationError(f"Failed to initialize analyzer: {str(e)}")

    def _setup_logging(self) -> None:
        """Configure logging based on the configuration.

        Raises:
            AnalysisConfigurationError: If logging setup fails
        """
        try:
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
                console_handler.setFormatter(
                    logging.Formatter(log_config.get("format"))
                )
                self.logger.addHandler(console_handler)
        except Exception as e:
            raise AnalysisConfigurationError(f"Failed to setup logging: {str(e)}")

    def analyze(
        self, tracer: AgentTracer, user_feedback: Optional[str] = None
    ) -> RootCauseAnalysis:
        """Analyze the agent trace to identify root causes.

        Args:
            tracer: The agent tracer containing the events to analyze
            user_feedback: Optional user feedback to consider

        Returns:
            Analysis results

        Raises:
            AnalysisProcessingError: If analysis fails
        """
        try:
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
                    "total_recommendations": len(recommendations),
                },
            )

            self.logger.info("Root cause analysis completed")
            return analysis
        except Exception as e:
            raise AnalysisProcessingError(f"Analysis failed: {str(e)}")

    def _analyze_events(self, events: List[Dict[str, Any]]) -> List[Issue]:
        """Analyze events to identify issues.

        Args:
            events: List of agent events

        Returns:
            List of identified issues

        Raises:
            AnalysisProcessingError: If event analysis fails
        """
        try:
            issues: List[Issue] = []

            # Analyze processing times
            for event in events:
                if event["metadata"] and event["metadata"].get(
                    "processing_time", 0
                ) > self.config.get("performance", {}).get(
                    "response_time_threshold", 5.0
                ):
                    issues.append(
                        {
                            "type": "performance",
                            "severity": "warning",
                            "description": f"Agent {event['agent_name']} took {event['metadata']['processing_time']:.2f}s to process messages",
                            "details": None,
                            "recommendation": "Consider optimizing agent processing or increasing timeout threshold",
                        }
                    )

            # Analyze message patterns
            message_counts: Dict[str, int] = {}
            for event in events:
                agent = event["agent_name"]
                message_counts[agent] = message_counts.get(agent, 0) + len(
                    event["inputs"]
                )

            for agent, count in message_counts.items():
                if count > self.config.get("analysis", {}).get(
                    "max_messages_per_agent", 10
                ):
                    issues.append(
                        {
                            "type": "communication",
                            "severity": "info",
                            "description": f"Agent {agent} processed {count} messages",
                            "details": None,
                            "recommendation": "Consider optimizing communication flow or increasing max_messages threshold",
                        }
                    )

            # Analyze error patterns
            for event in events:
                for output in event["outputs"]:
                    if (
                        isinstance(output["content"], str)
                        and "error" in output["content"].lower()
                    ):
                        issues.append(
                            {
                                "type": "error",
                                "severity": "error",
                                "description": f"Error detected in {event['agent_name']} output",
                                "details": output["content"],
                                "recommendation": "Review agent error handling and input validation",
                            }
                        )

            return issues
        except Exception as e:
            raise AnalysisProcessingError(f"Event analysis failed: {str(e)}")

    def _analyze_user_feedback(
        self, feedback: str, events: List[Dict[str, Any]]
    ) -> List[Issue]:
        """Analyze user feedback in context of events.

        Args:
            feedback: User feedback
            events: List of agent events

        Returns:
            List of issues identified from feedback

        Raises:
            AnalysisProcessingError: If feedback analysis fails
        """
        try:
            issues: List[Issue] = []

            # Add feedback as an issue
            issues.append(
                {
                    "type": "feedback",
                    "severity": "info",
                    "description": "User feedback received",
                    "details": feedback,
                    "recommendation": "Review feedback and consider implementing suggested improvements",
                }
            )

            # Analyze feedback against events
            if "slow" in feedback.lower():
                issues.append(
                    {
                        "type": "performance",
                        "severity": "warning",
                        "description": "User reported performance issues",
                        "details": None,
                        "recommendation": "Review agent processing times and optimize where possible",
                    }
                )

            if "error" in feedback.lower() or "failed" in feedback.lower():
                issues.append(
                    {
                        "type": "reliability",
                        "severity": "error",
                        "description": "User reported errors or failures",
                        "details": None,
                        "recommendation": "Review error handling and add more robust validation",
                    }
                )

            return issues
        except Exception as e:
            raise AnalysisProcessingError(f"Feedback analysis failed: {str(e)}")

    def _generate_recommendations(self, issues: List[Issue]) -> List[str]:
        """Generate recommendations based on identified issues.

        Args:
            issues: List of identified issues

        Returns:
            List of recommendations

        Raises:
            AnalysisProcessingError: If recommendation generation fails
        """
        try:
            recommendations: Set[str] = set()

            # Add recommendations from issues
            for issue in issues:
                if issue.get("recommendation"):
                    recommendations.add(issue["recommendation"])

            # Add general recommendations based on issue types
            issue_types = {issue["type"] for issue in issues}

            if "performance" in issue_types:
                recommendations.add(
                    "Consider implementing caching for frequently used data"
                )
                recommendations.add(
                    "Review and optimize database queries if applicable"
                )

            if "error" in issue_types:
                recommendations.add("Implement more comprehensive error handling")
                recommendations.add("Add input validation at all entry points")

            if "communication" in issue_types:
                recommendations.add("Review and optimize agent communication patterns")
                recommendations.add(
                    "Consider implementing message batching where appropriate"
                )

            return list(recommendations)
        except Exception as e:
            raise AnalysisProcessingError(f"Recommendation generation failed: {str(e)}")

    def _generate_summary(self, issues: List[Issue], recommendations: List[str]) -> str:
        """Generate a summary of the analysis.

        Args:
            issues: List of identified issues
            recommendations: List of recommendations

        Returns:
            Analysis summary

        Raises:
            AnalysisProcessingError: If summary generation fails
        """
        try:
            total_issues = len(issues)
            issue_types = {issue["type"] for issue in issues}
            severity_counts: Dict[str, int] = {}

            for issue in issues:
                severity = issue["severity"]
                severity_counts[severity] = severity_counts.get(severity, 0) + 1

            summary = "Analysis Summary:\n"
            summary += f"- Total Issues: {total_issues}\n"
            summary += f"- Issue Types: {', '.join(issue_types)}\n"
            summary += f"- Severity Breakdown: {', '.join(f'{k}: {v}' for k, v in severity_counts.items())}\n"
            summary += f"- Recommendations: {len(recommendations)}\n"

            return summary
        except Exception as e:
            raise AnalysisProcessingError(f"Summary generation failed: {str(e)}")

    def save_analysis(self, analysis: RootCauseAnalysis, file_path: str) -> None:
        """Save analysis results to a file.

        Args:
            analysis: Analysis results to save
            file_path: Path where to save the results

        Raises:
            AnalysisSaveError: If saving fails
        """
        try:
            # Create directory if it doesn't exist
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

            # Convert analysis to dictionary
            analysis_dict = asdict(analysis)

            # Save to file
            with open(file_path, "w") as f:
                json.dump(analysis_dict, f, indent=2)

            self.logger.info(f"Analysis saved to {file_path}")
        except Exception as e:
            raise AnalysisSaveError(f"Failed to save analysis: {str(e)}")
