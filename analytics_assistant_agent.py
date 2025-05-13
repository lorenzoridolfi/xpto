"""
Analytics Assistant Agent

This module implements an enhanced AssistantAgent with built-in analytics capabilities.
The agent tracks and analyzes its performance, tool usage, and interaction patterns.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
from collections import defaultdict
import json
import statistics
from dataclasses import dataclass, asdict

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseChatMessage
from tool_analytics import ToolAnalytics, ToolUsageMetrics

logger = logging.getLogger("AnalyticsAssistantAgent")

@dataclass
class InteractionMetrics:
    """Metrics for a single interaction."""
    start_time: datetime
    end_time: datetime
    response_time: float
    message_count: int
    tool_calls: int
    success: bool
    error_message: Optional[str] = None

@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics."""
    total_interactions: int
    average_response_time: float
    median_response_time: float
    response_time_std_dev: float
    success_rate: float
    tool_usage_rate: float
    error_rate: float

class AnalyticsAssistantAgent(AssistantAgent):
    """
    An enhanced AssistantAgent with built-in analytics capabilities.
    
    This agent extends the base AssistantAgent to add comprehensive analytics
    tracking, including performance metrics, tool usage patterns, and interaction
    analysis.
    """

    def __init__(
        self,
        name: str,
        model_client,
        system_message: str,
        tools: Optional[List[Dict]] = None,
        reflect_on_tool_use: bool = True
    ):
        """
        Initialize the AnalyticsAssistantAgent.

        Args:
            name (str): Name of the agent
            model_client: The model client to use
            system_message (str): System message for the agent
            tools (Optional[List[Dict]]): List of tools available to the agent
            reflect_on_tool_use (bool): Whether to reflect on tool use
        """
        super().__init__(name, model_client=model_client, system_message=system_message, tools=tools)
        self.tool_analytics = ToolAnalytics()
        self.reflect_on_tool_use = reflect_on_tool_use
        self.interaction_history = []
        self.performance_metrics = defaultdict(list)
        self.interaction_metrics: List[InteractionMetrics] = []
        self.error_history: List[Dict[str, Any]] = []
        
    async def on_messages(self, messages: List[BaseChatMessage], cancellation_token) -> Response:
        """
        Process incoming messages with analytics tracking.

        Args:
            messages (List[BaseChatMessage]): List of messages to process
            cancellation_token: Token for cancellation support

        Returns:
            Response: The agent's response to the messages
        """
        start_time = datetime.now()
        interaction_metrics = InteractionMetrics(
            start_time=start_time,
            end_time=start_time,
            response_time=0.0,
            message_count=len(messages),
            tool_calls=0,
            success=True
        )
        
        try:
            # Track interaction start
            self._track_interaction_start(messages)
            
            # Process messages
            response = await super().on_messages(messages, cancellation_token)
            
            # Calculate response time
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            # Update interaction metrics
            interaction_metrics.end_time = end_time
            interaction_metrics.response_time = response_time
            
            # Track interaction completion
            self._track_interaction_completion(response, response_time)
            
            # Update analytics if tools were used
            if hasattr(response, 'tool_calls') and response.tool_calls:
                interaction_metrics.tool_calls = len(response.tool_calls)
                self._update_tool_analytics(response.tool_calls, response_time)
                
                # Generate reflection if enabled
                if self.reflect_on_tool_use:
                    await self._reflect_on_tool_usage(response.tool_calls)
            
            # Store interaction metrics
            self.interaction_metrics.append(interaction_metrics)
            
            return response
            
        except Exception as e:
            # Track error
            interaction_metrics.success = False
            interaction_metrics.error_message = str(e)
            self.error_history.append({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "error": str(e),
                "context": {
                    "message_count": len(messages),
                    "tool_calls": interaction_metrics.tool_calls
                }
            })
            raise
    
    def _track_interaction_start(self, messages: List[BaseChatMessage]) -> None:
        """Track the start of an interaction."""
        self.interaction_history.append({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": "start",
            "messages": [{"source": m.source, "content": m.content} for m in messages]
        })
    
    def _track_interaction_completion(self, response: Response, response_time: float) -> None:
        """Track the completion of an interaction."""
        self.interaction_history.append({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": "completion",
            "response": {
                "source": response.chat_message.source,
                "content": response.chat_message.content
            },
            "response_time": response_time
        })
        
        # Update performance metrics
        self.performance_metrics["response_times"].append(response_time)
    
    def _update_tool_analytics(self, tool_calls: List[Dict], response_time: float) -> None:
        """Update tool usage analytics."""
        for tool_call in tool_calls:
            success = True  # You might want to determine this based on the response
            impact_score = 0.8  # You might want to calculate this based on the response
            context_relevance = 0.7  # You might want to calculate this based on the context
            
            self.tool_analytics.update_metrics(
                tool_call.get("name"),
                success,
                response_time,
                impact_score,
                context_relevance
            )
    
    async def _reflect_on_tool_usage(self, tool_calls: List[Dict]) -> None:
        """Reflect on tool usage and generate insights."""
        reflection_prompt = self.tool_analytics.generate_reflection_prompt(tool_calls)
        
        try:
            reflection_response = await self.model_client.create(
                messages=[{"role": "system", "content": reflection_prompt}]
            )
            
            reflection = reflection_response.choices[0].message.content
            
            # Log reflection insights
            logger.info(f"Tool usage reflection for {self.name}:")
            logger.info(reflection)
            
            # Store reflection in analytics
            self.tool_analytics.add_reflection(tool_calls, reflection)
            
        except Exception as e:
            logger.error(f"Error generating tool reflection: {e}")
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.

        Returns:
            PerformanceMetrics: Aggregated performance metrics
        """
        response_times = [m.response_time for m in self.interaction_metrics]
        total_interactions = len(self.interaction_metrics)
        
        if not total_interactions:
            return PerformanceMetrics(
                total_interactions=0,
                average_response_time=0.0,
                median_response_time=0.0,
                response_time_std_dev=0.0,
                success_rate=0.0,
                tool_usage_rate=0.0,
                error_rate=0.0
            )
        
        successful_interactions = sum(1 for m in self.interaction_metrics if m.success)
        tool_using_interactions = sum(1 for m in self.interaction_metrics if m.tool_calls > 0)
        error_count = len(self.error_history)
        
        return PerformanceMetrics(
            total_interactions=total_interactions,
            average_response_time=statistics.mean(response_times),
            median_response_time=statistics.median(response_times),
            response_time_std_dev=statistics.stdev(response_times) if len(response_times) > 1 else 0.0,
            success_rate=successful_interactions / total_interactions,
            tool_usage_rate=tool_using_interactions / total_interactions,
            error_rate=error_count / total_interactions
        )
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the agent's analytics.

        Returns:
            Dict[str, Any]: Analytics summary including performance metrics,
                           tool usage patterns, and interaction history
        """
        performance_metrics = self.get_performance_metrics()
        
        return {
            "performance_metrics": asdict(performance_metrics),
            "tool_usage": self.tool_analytics.get_performance_dashboard(),
            "interaction_history": self.interaction_history,
            "error_history": self.error_history,
            "optimization_suggestions": self.tool_analytics.get_optimization_suggestions()
        }
    
    def get_error_analysis(self) -> Dict[str, Any]:
        """
        Analyze error patterns and provide insights.

        Returns:
            Dict[str, Any]: Error analysis including patterns and recommendations
        """
        if not self.error_history:
            return {"error_count": 0, "patterns": [], "recommendations": []}
        
        # Group errors by type
        error_types = defaultdict(list)
        for error in self.error_history:
            error_types[error["error"]].append(error)
        
        # Analyze patterns
        patterns = []
        for error_type, occurrences in error_types.items():
            patterns.append({
                "error_type": error_type,
                "count": len(occurrences),
                "first_occurrence": min(e["timestamp"] for e in occurrences),
                "last_occurrence": max(e["timestamp"] for e in occurrences),
                "context": {
                    "avg_message_count": statistics.mean(e["context"]["message_count"] for e in occurrences),
                    "avg_tool_calls": statistics.mean(e["context"]["tool_calls"] for e in occurrences)
                }
            })
        
        # Generate recommendations
        recommendations = []
        for pattern in patterns:
            if pattern["count"] > 1:
                recommendations.append({
                    "error_type": pattern["error_type"],
                    "suggestion": f"Implement retry mechanism for {pattern['error_type']}",
                    "priority": "high" if pattern["count"] > 3 else "medium"
                })
        
        return {
            "error_count": len(self.error_history),
            "patterns": patterns,
            "recommendations": recommendations
        } 