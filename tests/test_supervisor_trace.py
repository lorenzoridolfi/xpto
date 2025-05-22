from src.supervisor_trace import SupervisorTrace


def test_supervisor_trace_lifecycle():
    trace = SupervisorTrace({})
    trace.start_trace({"task": "t"})
    trace.end_trace({"result": "r"})
    trace.trace_agent_creation(None, {})
    trace.trace_agent_interaction(None, None, "msg")
    trace.trace_decision("type", {})
    trace.trace_error(Exception("err"), {})
    assert isinstance(trace.get_all_traces(), list)
    trace.clear_traces()
    assert trace.get_all_traces() == []
