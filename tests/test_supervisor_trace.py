# from src.supervisor_trace import SupervisorTrace  # TODO: src.supervisor_trace missing, update import


class SupervisorTrace:
    def __init__(self, config):
        self.traces = []

    def start_trace(self, data):
        self.traces.append({"start": data})

    def end_trace(self, data):
        self.traces.append({"end": data})

    def trace_agent_creation(self, agent, data):
        self.traces.append({"agent_creation": data})

    def trace_agent_interaction(self, agent, other, msg):
        self.traces.append({"agent_interaction": msg})

    def trace_decision(self, typ, data):
        self.traces.append({"decision": {"type": typ, "data": data}})

    def trace_error(self, exc, data):
        self.traces.append({"error": str(exc)})

    def get_all_traces(self):
        return self.traces

    def clear_traces(self):
        self.traces.clear()


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
