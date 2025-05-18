import pytest
from src.feedback_manager import QueryFeedbackManager

def test_feedback_manager_process_query_and_feedback():
    mgr = QueryFeedbackManager()
    entry_id = pytest.run(mgr.process_query("query"))
    assert entry_id is not None
    pytest.run(mgr.process_feedback(entry_id, "feedback"))
