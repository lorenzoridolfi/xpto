from src.root_cause_analyzer import RootCauseAnalyzer, AnalysisConfig

def test_root_cause_analyzer_analyze():
    config = AnalysisConfig()
    analyzer = RootCauseAnalyzer(config)
    result = analyzer.analyze(None)
    assert hasattr(result, "summary")
