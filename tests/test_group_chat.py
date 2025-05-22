from autogen_extensions.traced_group_chat import TracedGroupChat


class TestTracedGroupChat:
    def test_traced_group_chat_instantiation(self):
        group = TracedGroupChat(agents=[], trace_path="/tmp/trace.json", messages=[])
        assert group.trace_path == "/tmp/trace.json"
        assert group.action_trace == []

    def test_traced_group_chat_log_action(self):
        group = TracedGroupChat(agents=[], trace_path="/tmp/trace.json", messages=[])
        group._log_action("test_action", {"foo": "bar"}, "TestAgent")
        assert len(group.action_trace) == 1
        assert group.action_trace[0]["action_type"] == "test_action"
        assert group.action_trace[0]["agent"] == "TestAgent"

    def test_traced_group_chat_save_trace(self, tmp_path):
        trace_file = tmp_path / "trace.json"
        group = TracedGroupChat(agents=[], trace_path=str(trace_file), messages=[])
        group._log_action("test_action", {"foo": "bar"}, "TestAgent")
        group._save_trace()
        import json

        with open(trace_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "actions" in data
        assert data["actions"][0]["action_type"] == "test_action"
