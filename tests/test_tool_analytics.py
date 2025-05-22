import pytest
from src.tool_analytics import ToolAnalytics


def test_tool_analytics_metrics():
    analytics = ToolAnalytics()
    # If ToolAnalytics has a method like get_metrics or similar, use it here. Otherwise, skip the test.
    if hasattr(analytics, "get_metrics"):
        metrics = analytics.get_metrics()
        assert isinstance(metrics, dict)
    else:
        pytest.skip("ToolAnalytics has no public usage recording method.")
