"""
Root Cause Analysis System

This module provides a comprehensive system for analyzing and identifying root causes
of issues in agent interactions. It includes:

- Event correlation and analysis
- Pattern recognition
- Dependency tracking
- Impact assessment
- Solution recommendations
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import json
import networkx as nx
from collections import defaultdict

@dataclass
class Event:
    """
    Represents an event in the system.
    
    Attributes:
        timestamp (datetime): When the event occurred
        event_type (str): Type of event
        source (str): Source of the event
        details (Dict[str, Any]): Event details
        severity (int): Event severity (1-5)
        dependencies (List[str]): IDs of dependent events
    """
    timestamp: datetime
    event_type: str
    source: str
    details: Dict[str, Any]
    severity: int
    dependencies: List[str] = None

@dataclass
class RootCause:
    """
    Represents a root cause analysis result.
    
    Attributes:
        cause (str): Description of the root cause
        confidence (float): Confidence score (0-1)
        evidence (List[Dict[str, Any]]): Supporting evidence
        impact (Dict[str, Any]): Impact assessment
        recommendations (List[str]): Recommended solutions
    """
    cause: str
    confidence: float
    evidence: List[Dict[str, Any]]
    impact: Dict[str, Any]
    recommendations: List[str]

class RootCauseAnalyzer:
    """
    Analyzes events to identify root causes of issues.
    
    This class provides:
    - Event correlation and analysis
    - Pattern recognition
    - Dependency tracking
    - Impact assessment
    - Solution recommendations
    
    Attributes:
        events (List[Event]): List of events to analyze
        event_graph (nx.DiGraph): Graph of event dependencies
        patterns (Dict[str, List[Dict[str, Any]]]): Known event patterns
        solutions (Dict[str, List[str]]): Known solutions for issues
    """
    
    def __init__(self):
        """Initialize the root cause analyzer."""
        self.events: List[Event] = []
        self.event_graph = nx.DiGraph()
        self.patterns = {}
        self.solutions = {}

    def add_event(self, event: Event) -> None:
        """
        Add an event to the analysis.

        Args:
            event (Event): Event to add

        Raises:
            ValueError: If event is invalid
        """
        if not isinstance(event, Event):
            raise ValueError("Invalid event object")
            
        if event.severity < 1 or event.severity > 5:
            raise ValueError("Event severity must be between 1 and 5")
            
        self.events.append(event)
        
        # Add to graph
        self.event_graph.add_node(
            id(event),
            timestamp=event.timestamp,
            type=event.event_type,
            source=event.source,
            details=event.details,
            severity=event.severity
        )
        
        # Add dependencies
        if event.dependencies:
            for dep_id in event.dependencies:
                self.event_graph.add_edge(dep_id, id(event))

    def add_pattern(self,
                   pattern_name: str,
                   pattern: List[Dict[str, Any]]) -> None:
        """
        Add a known event pattern.

        Args:
            pattern_name (str): Name of the pattern
            pattern (List[Dict[str, Any]]): Pattern definition

        Raises:
            ValueError: If pattern is invalid
        """
        if not isinstance(pattern, list):
            raise ValueError("Pattern must be a list")
            
        if not all(isinstance(p, dict) for p in pattern):
            raise ValueError("Pattern elements must be dictionaries")
            
        self.patterns[pattern_name] = pattern

    def add_solution(self,
                    issue_type: str,
                    solutions: List[str]) -> None:
        """
        Add known solutions for an issue type.

        Args:
            issue_type (str): Type of issue
            solutions (List[str]): List of solutions

        Raises:
            ValueError: If solutions are invalid
        """
        if not isinstance(solutions, list):
            raise ValueError("Solutions must be a list")
            
        if not all(isinstance(s, str) for s in solutions):
            raise ValueError("Solutions must be strings")
            
        self.solutions[issue_type] = solutions

    def analyze(self,
               start_time: Optional[datetime] = None,
               end_time: Optional[datetime] = None,
               min_severity: int = 1) -> List[RootCause]:
        """
        Analyze events to identify root causes.

        Args:
            start_time (Optional[datetime]): Start time for analysis
            end_time (Optional[datetime]): End time for analysis
            min_severity (int): Minimum event severity to consider

        Returns:
            List[RootCause]: List of identified root causes

        Raises:
            ValueError: If time range is invalid
        """
        if start_time and end_time and start_time > end_time:
            raise ValueError("Start time must be before end time")
            
        # Filter events
        filtered_events = [
            event for event in self.events
            if (not start_time or event.timestamp >= start_time) and
               (not end_time or event.timestamp <= end_time) and
               event.severity >= min_severity
        ]
        
        if not filtered_events:
            return []
            
        # Analyze patterns
        root_causes = []
        for pattern_name, pattern in self.patterns.items():
            matches = self._find_pattern_matches(filtered_events, pattern)
            if matches:
                root_causes.extend(self._analyze_pattern(pattern_name, matches))
                
        # Analyze dependencies
        dependency_causes = self._analyze_dependencies(filtered_events)
        root_causes.extend(dependency_causes)
        
        # Sort by confidence
        root_causes.sort(key=lambda x: x.confidence, reverse=True)
        
        return root_causes

    def _find_pattern_matches(self,
                            events: List[Event],
                            pattern: List[Dict[str, Any]]) -> List[List[Event]]:
        """
        Find events matching a pattern.

        Args:
            events (List[Event]): Events to search
            pattern (List[Dict[str, Any]]): Pattern to match

        Returns:
            List[List[Event]]: List of matching event sequences
        """
        matches = []
        
        for i in range(len(events) - len(pattern) + 1):
            sequence = events[i:i + len(pattern)]
            if self._matches_pattern(sequence, pattern):
                matches.append(sequence)
                
        return matches

    def _matches_pattern(self,
                        sequence: List[Event],
                        pattern: List[Dict[str, Any]]) -> bool:
        """
        Check if a sequence of events matches a pattern.

        Args:
            sequence (List[Event]): Sequence of events
            pattern (List[Dict[str, Any]]): Pattern to match

        Returns:
            bool: True if sequence matches pattern
        """
        if len(sequence) != len(pattern):
            return False
            
        for event, pattern_item in zip(sequence, pattern):
            # Check event type
            if pattern_item.get("type") and event.event_type != pattern_item["type"]:
                return False
                
            # Check source
            if pattern_item.get("source") and event.source != pattern_item["source"]:
                return False
                
            # Check severity
            if pattern_item.get("min_severity") and event.severity < pattern_item["min_severity"]:
                return False
                
            # Check details
            if pattern_item.get("details"):
                for key, value in pattern_item["details"].items():
                    if key not in event.details or event.details[key] != value:
                        return False
                        
        return True

    def _analyze_pattern(self,
                        pattern_name: str,
                        matches: List[List[Event]]) -> List[RootCause]:
        """
        Analyze pattern matches to identify root causes.

        Args:
            pattern_name (str): Name of the pattern
            matches (List[List[Event]]): Pattern matches

        Returns:
            List[RootCause]: Identified root causes
        """
        root_causes = []
        
        for match in matches:
            # Calculate confidence based on match quality
            confidence = self._calculate_pattern_confidence(match)
            
            # Get impact
            impact = self._assess_impact(match)
            
            # Get recommendations
            recommendations = self.solutions.get(pattern_name, [])
            
            # Create root cause
            root_cause = RootCause(
                cause=f"Pattern match: {pattern_name}",
                confidence=confidence,
                evidence=[{
                    "type": "pattern_match",
                    "pattern": pattern_name,
                    "events": [
                        {
                            "timestamp": event.timestamp.isoformat(),
                            "type": event.event_type,
                            "source": event.source,
                            "details": event.details,
                            "severity": event.severity
                        }
                        for event in match
                    ]
                }],
                impact=impact,
                recommendations=recommendations
            )
            
            root_causes.append(root_cause)
            
        return root_causes

    def _analyze_dependencies(self,
                            events: List[Event]) -> List[RootCause]:
        """
        Analyze event dependencies to identify root causes.

        Args:
            events (List[Event]): Events to analyze

        Returns:
            List[RootCause]: Identified root causes
        """
        root_causes = []
        
        # Get event IDs
        event_ids = {id(event): event for event in events}
        
        # Find root nodes in dependency graph
        root_nodes = [
            node for node in self.event_graph.nodes()
            if self.event_graph.in_degree(node) == 0 and node in event_ids
        ]
        
        for root in root_nodes:
            # Get dependent events
            descendants = nx.descendants(self.event_graph, root)
            dependent_events = [
                event_ids[node] for node in descendants
                if node in event_ids
            ]
            
            if dependent_events:
                # Calculate confidence
                confidence = self._calculate_dependency_confidence(
                    event_ids[root],
                    dependent_events
                )
                
                # Get impact
                impact = self._assess_impact([event_ids[root]] + dependent_events)
                
                # Get recommendations
                recommendations = self.solutions.get(
                    event_ids[root].event_type,
                    []
                )
                
                # Create root cause
                root_cause = RootCause(
                    cause=f"Dependency root: {event_ids[root].event_type}",
                    confidence=confidence,
                    evidence=[{
                        "type": "dependency",
                        "root_event": {
                            "timestamp": event_ids[root].timestamp.isoformat(),
                            "type": event_ids[root].event_type,
                            "source": event_ids[root].source,
                            "details": event_ids[root].details,
                            "severity": event_ids[root].severity
                        },
                        "dependent_events": [
                            {
                                "timestamp": event.timestamp.isoformat(),
                                "type": event.event_type,
                                "source": event.source,
                                "details": event.details,
                                "severity": event.severity
                            }
                            for event in dependent_events
                        ]
                    }],
                    impact=impact,
                    recommendations=recommendations
                )
                
                root_causes.append(root_cause)
                
        return root_causes

    def _calculate_pattern_confidence(self,
                                    match: List[Event]) -> float:
        """
        Calculate confidence score for a pattern match.

        Args:
            match (List[Event]): Pattern match

        Returns:
            float: Confidence score (0-1)
        """
        # Base confidence on severity and timing
        severity_score = sum(event.severity for event in match) / (5 * len(match))
        
        # Check timing between events
        timing_score = 1.0
        for i in range(len(match) - 1):
            time_diff = (match[i + 1].timestamp - match[i].timestamp).total_seconds()
            if time_diff > 3600:  # More than 1 hour
                timing_score *= 0.5
                
        return (severity_score + timing_score) / 2

    def _calculate_dependency_confidence(self,
                                       root: Event,
                                       dependents: List[Event]) -> float:
        """
        Calculate confidence score for a dependency.

        Args:
            root (Event): Root event
            dependents (List[Event]): Dependent events

        Returns:
            float: Confidence score (0-1)
        """
        # Base confidence on severity
        severity_score = root.severity / 5
        
        # Check timing between root and dependents
        timing_score = 1.0
        for dependent in dependents:
            time_diff = (dependent.timestamp - root.timestamp).total_seconds()
            if time_diff > 3600:  # More than 1 hour
                timing_score *= 0.5
                
        # Check number of dependents
        dependency_score = min(len(dependents) / 10, 1.0)
        
        return (severity_score + timing_score + dependency_score) / 3

    def _assess_impact(self, events: List[Event]) -> Dict[str, Any]:
        """
        Assess the impact of events.

        Args:
            events (List[Event]): Events to assess

        Returns:
            Dict[str, Any]: Impact assessment including:
                - severity: Overall severity
                - affected_sources: List of affected sources
                - duration: Duration of impact
                - event_count: Number of events
        """
        if not events:
            return {
                "severity": 0,
                "affected_sources": [],
                "duration": 0,
                "event_count": 0
            }
            
        # Calculate overall severity
        severity = max(event.severity for event in events)
        
        # Get affected sources
        affected_sources = list(set(event.source for event in events))
        
        # Calculate duration
        duration = (
            max(event.timestamp for event in events) -
            min(event.timestamp for event in events)
        ).total_seconds()
        
        return {
            "severity": severity,
            "affected_sources": affected_sources,
            "duration": duration,
            "event_count": len(events)
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get analysis statistics.

        Returns:
            Dict[str, Any]: Statistics including:
                - total_events: Total number of events
                - pattern_count: Number of known patterns
                - solution_count: Number of known solutions
                - avg_severity: Average event severity
                - dependency_count: Number of event dependencies
        """
        if not self.events:
            return {
                "total_events": 0,
                "pattern_count": len(self.patterns),
                "solution_count": len(self.solutions),
                "avg_severity": 0,
                "dependency_count": 0
            }
            
        return {
            "total_events": len(self.events),
            "pattern_count": len(self.patterns),
            "solution_count": len(self.solutions),
            "avg_severity": sum(event.severity for event in self.events) / len(self.events),
            "dependency_count": self.event_graph.number_of_edges()
        } 