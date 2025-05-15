"""
Tool Analytics System

This module provides comprehensive analytics for tool usage and performance.
It includes:

- Usage tracking and metrics
- Performance monitoring
- Error analysis
- Cost tracking
- Trend analysis
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import time
from collections import defaultdict

@dataclass
class ToolUsage:
    """
    Represents a tool usage event.
    
    Attributes:
        timestamp (datetime): When the tool was used
        tool_name (str): Name of the tool
        parameters (Dict[str, Any]): Tool parameters
        duration (float): Execution duration in seconds
        success (bool): Whether the tool call succeeded
        error (Optional[str]): Error message if failed
        cost (Optional[float]): Cost of the tool call
    """
    timestamp: datetime
    tool_name: str
    parameters: Dict[str, Any]
    duration: float
    success: bool
    error: Optional[str] = None
    cost: Optional[float] = None

@dataclass
class ToolMetrics:
    """
    Represents aggregated tool metrics.
    
    Attributes:
        total_calls (int): Total number of calls
        success_rate (float): Rate of successful calls
        avg_duration (float): Average call duration
        total_cost (float): Total cost of calls
        error_count (int): Number of errors
        error_types (Dict[str, int]): Count of error types
    """
    total_calls: int
    success_rate: float
    avg_duration: float
    total_cost: float
    error_count: int
    error_types: Dict[str, int]

class ToolAnalytics:
    """
    Analyzes tool usage and performance.
    
    This class provides:
    - Usage tracking and metrics
    - Performance monitoring
    - Error analysis
    - Cost tracking
    - Trend analysis
    
    Attributes:
        usages (List[ToolUsage]): List of tool usage events
        metrics (Dict[str, ToolMetrics]): Aggregated metrics per tool
        error_patterns (Dict[str, List[str]]): Known error patterns
        cost_rates (Dict[str, float]): Cost rates per tool
    """
    
    def __init__(self):
        """Initialize the tool analytics system."""
        self.usages: List[ToolUsage] = []
        self.metrics: Dict[str, ToolMetrics] = {}
        self.error_patterns = {}
        self.cost_rates = {}

    def record_usage(self, usage: ToolUsage) -> None:
        """
        Record a tool usage event.

        Args:
            usage (ToolUsage): Tool usage event to record

        Raises:
            ValueError: If usage is invalid
        """
        if not isinstance(usage, ToolUsage):
            raise ValueError("Invalid usage object")
            
        if usage.duration < 0:
            raise ValueError("Duration must be non-negative")
            
        if usage.cost is not None and usage.cost < 0:
            raise ValueError("Cost must be non-negative")
            
        self.usages.append(usage)
        self._update_metrics(usage)

    def _update_metrics(self, usage: ToolUsage) -> None:
        """
        Update metrics for a tool.

        Args:
            usage (ToolUsage): Tool usage event
        """
        if usage.tool_name not in self.metrics:
            self.metrics[usage.tool_name] = ToolMetrics(
                total_calls=0,
                success_rate=0.0,
                avg_duration=0.0,
                total_cost=0.0,
                error_count=0,
                error_types={}
            )
            
        metrics = self.metrics[usage.tool_name]
        
        # Update counts
        metrics.total_calls += 1
        if not usage.success:
            metrics.error_count += 1
            if usage.error:
                metrics.error_types[usage.error] = metrics.error_types.get(usage.error, 0) + 1
                
        # Update rates
        metrics.success_rate = (
            (metrics.total_calls - metrics.error_count) /
            metrics.total_calls
        )
        
        # Update duration
        metrics.avg_duration = (
            (metrics.avg_duration * (metrics.total_calls - 1) + usage.duration) /
            metrics.total_calls
        )
        
        # Update cost
        if usage.cost is not None:
            metrics.total_cost += usage.cost

    def get_tool_metrics(self,
                        tool_name: str,
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None) -> Optional[ToolMetrics]:
        """
        Get metrics for a specific tool.

        Args:
            tool_name (str): Name of the tool
            start_time (Optional[datetime]): Start time for metrics
            end_time (Optional[datetime]): End time for metrics

        Returns:
            Optional[ToolMetrics]: Tool metrics if found

        Raises:
            ValueError: If time range is invalid
        """
        if start_time and end_time and start_time > end_time:
            raise ValueError("Start time must be before end time")
            
        if tool_name not in self.metrics:
            return None
            
        # Filter usages
        filtered_usages = [
            usage for usage in self.usages
            if usage.tool_name == tool_name and
               (not start_time or usage.timestamp >= start_time) and
               (not end_time or usage.timestamp <= end_time)
        ]
        
        if not filtered_usages:
            return None
            
        # Calculate metrics
        total_calls = len(filtered_usages)
        success_count = sum(1 for u in filtered_usages if u.success)
        error_count = total_calls - success_count
        
        error_types = defaultdict(int)
        for usage in filtered_usages:
            if not usage.success and usage.error:
                error_types[usage.error] += 1
                
        return ToolMetrics(
            total_calls=total_calls,
            success_rate=success_count / total_calls if total_calls > 0 else 0.0,
            avg_duration=sum(u.duration for u in filtered_usages) / total_calls if total_calls > 0 else 0.0,
            total_cost=sum(u.cost or 0.0 for u in filtered_usages),
            error_count=error_count,
            error_types=dict(error_types)
        )

    def get_all_metrics(self,
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> Dict[str, ToolMetrics]:
        """
        Get metrics for all tools.

        Args:
            start_time (Optional[datetime]): Start time for metrics
            end_time (Optional[datetime]): End time for metrics

        Returns:
            Dict[str, ToolMetrics]: Dictionary of tool metrics

        Raises:
            ValueError: If time range is invalid
        """
        if start_time and end_time and start_time > end_time:
            raise ValueError("Start time must be before end time")
            
        return {
            tool_name: self.get_tool_metrics(tool_name, start_time, end_time)
            for tool_name in self.metrics.keys()
        }

    def analyze_errors(self,
                      tool_name: Optional[str] = None,
                      start_time: Optional[datetime] = None,
                      end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Analyze tool errors.

        Args:
            tool_name (Optional[str]): Name of the tool to analyze
            start_time (Optional[datetime]): Start time for analysis
            end_time (Optional[datetime]): End time for analysis

        Returns:
            Dict[str, Any]: Error analysis including:
                - total_errors: Total number of errors
                - error_types: Count of error types
                - error_trend: Error trend over time
                - common_errors: Most common errors
                - error_patterns: Identified error patterns

        Raises:
            ValueError: If time range is invalid
        """
        if start_time and end_time and start_time > end_time:
            raise ValueError("Start time must be before end time")
            
        # Filter usages
        filtered_usages = [
            usage for usage in self.usages
            if (not tool_name or usage.tool_name == tool_name) and
               (not start_time or usage.timestamp >= start_time) and
               (not end_time or usage.timestamp <= end_time) and
               not usage.success
        ]
        
        if not filtered_usages:
            return {
                "total_errors": 0,
                "error_types": {},
                "error_trend": [],
                "common_errors": [],
                "error_patterns": []
            }
            
        # Count error types
        error_types = defaultdict(int)
        for usage in filtered_usages:
            if usage.error:
                error_types[usage.error] += 1
                
        # Calculate error trend
        if start_time and end_time:
            time_range = end_time - start_time
            interval = time_range / 10  # 10 intervals
            
            error_trend = []
            for i in range(10):
                interval_start = start_time + (interval * i)
                interval_end = interval_start + interval
                
                interval_errors = sum(
                    1 for u in filtered_usages
                    if interval_start <= u.timestamp < interval_end
                )
                
                error_trend.append({
                    "timestamp": interval_start.isoformat(),
                    "count": interval_errors
                })
        else:
            error_trend = []
            
        # Find common errors
        common_errors = sorted(
            error_types.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Identify error patterns
        error_patterns = []
        for pattern_name, patterns in self.error_patterns.items():
            matches = sum(
                1 for usage in filtered_usages
                if usage.error and any(p in usage.error for p in patterns)
            )
            if matches > 0:
                error_patterns.append({
                    "pattern": pattern_name,
                    "matches": matches
                })
                
        return {
            "total_errors": len(filtered_usages),
            "error_types": dict(error_types),
            "error_trend": error_trend,
            "common_errors": common_errors,
            "error_patterns": error_patterns
        }

    def analyze_performance(self,
                          tool_name: Optional[str] = None,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Analyze tool performance.

        Args:
            tool_name (Optional[str]): Name of the tool to analyze
            start_time (Optional[datetime]): Start time for analysis
            end_time (Optional[datetime]): End time for analysis

        Returns:
            Dict[str, Any]: Performance analysis including:
                - avg_duration: Average call duration
                - duration_trend: Duration trend over time
                - slow_calls: Slowest calls
                - throughput: Calls per time unit
                - cost_efficiency: Cost per successful call

        Raises:
            ValueError: If time range is invalid
        """
        if start_time and end_time and start_time > end_time:
            raise ValueError("Start time must be before end time")
            
        # Filter usages
        filtered_usages = [
            usage for usage in self.usages
            if (not tool_name or usage.tool_name == tool_name) and
               (not start_time or usage.timestamp >= start_time) and
               (not end_time or usage.timestamp <= end_time)
        ]
        
        if not filtered_usages:
            return {
                "avg_duration": 0.0,
                "duration_trend": [],
                "slow_calls": [],
                "throughput": 0.0,
                "cost_efficiency": 0.0
            }
            
        # Calculate average duration
        avg_duration = sum(u.duration for u in filtered_usages) / len(filtered_usages)
        
        # Calculate duration trend
        if start_time and end_time:
            time_range = end_time - start_time
            interval = time_range / 10  # 10 intervals
            
            duration_trend = []
            for i in range(10):
                interval_start = start_time + (interval * i)
                interval_end = interval_start + interval
                
                interval_usages = [
                    u for u in filtered_usages
                    if interval_start <= u.timestamp < interval_end
                ]
                
                if interval_usages:
                    avg_interval_duration = sum(u.duration for u in interval_usages) / len(interval_usages)
                    duration_trend.append({
                        "timestamp": interval_start.isoformat(),
                        "avg_duration": avg_interval_duration
                    })
        else:
            duration_trend = []
            
        # Find slowest calls
        slow_calls = sorted(
            filtered_usages,
            key=lambda x: x.duration,
            reverse=True
        )[:5]
        
        # Calculate throughput
        if start_time and end_time:
            time_range = (end_time - start_time).total_seconds()
            throughput = len(filtered_usages) / time_range if time_range > 0 else 0
        else:
            throughput = 0
            
        # Calculate cost efficiency
        successful_calls = [u for u in filtered_usages if u.success]
        total_cost = sum(u.cost or 0.0 for u in successful_calls)
        cost_efficiency = total_cost / len(successful_calls) if successful_calls else 0
        
        return {
            "avg_duration": avg_duration,
            "duration_trend": duration_trend,
            "slow_calls": [
                {
                    "timestamp": u.timestamp.isoformat(),
                    "tool": u.tool_name,
                    "duration": u.duration,
                    "parameters": u.parameters
                }
                for u in slow_calls
            ],
            "throughput": throughput,
            "cost_efficiency": cost_efficiency
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics.

        Returns:
            Dict[str, Any]: Statistics including:
                - total_calls: Total number of tool calls
                - tool_count: Number of unique tools
                - success_rate: Overall success rate
                - total_cost: Total cost of all calls
                - avg_duration: Average call duration
        """
        if not self.usages:
            return {
                "total_calls": 0,
                "tool_count": 0,
                "success_rate": 0.0,
                "total_cost": 0.0,
                "avg_duration": 0.0
            }
            
        total_calls = len(self.usages)
        success_count = sum(1 for u in self.usages if u.success)
        total_cost = sum(u.cost or 0.0 for u in self.usages)
        
        return {
            "total_calls": total_calls,
            "tool_count": len(self.metrics),
            "success_rate": success_count / total_calls if total_calls > 0 else 0.0,
            "total_cost": total_cost,
            "avg_duration": sum(u.duration for u in self.usages) / total_calls if total_calls > 0 else 0.0
        } 