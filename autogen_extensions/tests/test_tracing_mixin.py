import os
import json
import tempfile
import pytest
from autogen_extensions.tracing import TracingMixin

class DummyTracer(TracingMixin):
    def __init__(self, trace_path=None):
        super().__init__(trace_path=trace_path)


def test_log_event_and_save_trace():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        path = tmp.name
    tracer = DummyTracer(trace_path=path)
    tracer.log_event("test_event", {"foo": "bar", "num": 42})
    tracer.log_event("another_event", {"baz": True})
    tracer.save_trace()
    with open(path) as f:
        data = json.load(f)
    assert len(data) == 2
    assert data[0]["event_type"] == "test_event"
    assert data[0]["foo"] == "bar"
    assert data[1]["event_type"] == "another_event"
    assert data[1]["baz"] is True
    os.remove(path)


def test_save_trace_without_path_raises():
    tracer = DummyTracer(trace_path=None)
    tracer.log_event("event", {})
    with pytest.raises(ValueError):
        tracer.save_trace() 