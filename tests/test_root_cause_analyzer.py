from src.root_cause_analyzer import RootCauseAnalyzer, AnalysisConfig
from src.agent_tracer import AgentTracer

def test_root_cause_analyzer_analyze():
    config = AnalysisConfig()
    analyzer = RootCauseAnalyzer(config)
    tracer = AgentTracer({"logging": {"level": "WARNING"}})
    result = analyzer.analyze(tracer)
    assert hasattr(result, "summary")
    assert hasattr(result, "issues")
    assert hasattr(result, "recommendations")
