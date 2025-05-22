from src.agent_tracer import AgentTracer


def test_agent_tracer_trace():
    tracer = AgentTracer({})
    tracer.on_messages_invoke("agent", [{"content": "msg"}], None)
    tracer.on_messages_complete("agent", [{"output": "out"}], None)
    trace = tracer.get_trace()
    assert isinstance(trace, list)
    tracer.clear_trace()
    assert tracer.get_trace() == []
