from src.tool_analytics import ToolAnalytics

def test_tool_analytics_metrics():
    analytics = ToolAnalytics()
    analytics.record_usage("tool1")
    analytics.record_usage("tool1")
    analytics.record_usage("tool2")
    metrics = analytics.get_metrics()
    assert metrics["tool1"] == 2
    assert metrics["tool2"] == 1
