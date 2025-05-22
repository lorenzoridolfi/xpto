import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from autogen_extensions.auto_tracing_group_chat import AutoTracingGroupChat
import pytest

class DummyAgent:
    def __init__(self, name, description, system_message):
        self.name = name
        self.description = description
        self.system_message = system_message

    def read(self, file_path):
        return "dummy content"

    def summarize(self, file_path, content):
        return {"description": "desc", "summary": "summary"}

    def validate(self, manifest, schema):
        return True

def build_simple_manifest_with_tracing(text_dir, trace_path):
    file_reader = DummyAgent("FileReaderAgent", "Reads file content.", "Reads file content.")
    summarizer = DummyAgent("SummarizerAgent", "Summarizes file content.", "Summarizes file content.")
    validator = DummyAgent("ValidatorAgent", "Validates manifest.", "Validates manifest.")
    agents = [file_reader, summarizer, validator]
    group_description = "Test group for manifest workflow."
    manifest = {"version": "1.0.0", "files": [], "metadata": {"statistics": {}}}
    with AutoTracingGroupChat(agents=agents, trace_path=trace_path, description=group_description) as group:
        for file_path in Path(text_dir).iterdir():
            if file_path.name.startswith('.') or not file_path.is_file():
                continue
            content = file_reader.read(file_path)
            group.agent_action("file_read", {"file": str(file_path)}, file_reader.name)
            summary_data = summarizer.summarize(file_path, content)
            group.agent_action("file_summarized", {"file": str(file_path), "summary": summary_data["summary"]}, summarizer.name)
            file_entry = {
                "filename": file_path.name,
                "path": str(file_path),
                "description": summary_data["description"],
                "status": "okay",
                "metadata": {"summary": summary_data["summary"]},
                "sha256": "dummysha256",
                "modified_date": "2025-01-01T00:00:00Z",
                "file_type": "text",
                "encoding": "utf-8",
                "size": 42,
                "dependencies": [],
                "category": "input",
                "read_order": 0,
            }
            manifest["files"].append(file_entry)
            group.agent_action("file_added_to_manifest", {"file": str(file_path)}, "System")
        valid = validator.validate(manifest, {})
        group.agent_action("manifest_validated", {"result": valid}, validator.name)
    return manifest

@pytest.mark.asyncio
async def test_auto_tracing_group_chat_workflow(tmp_path):
    # Setup: create a sample file
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("hello world", encoding="utf-8")
    trace_path = tmp_path / "trace.json"
    # Run the workflow
    manifest = build_simple_manifest_with_tracing(tmp_path, trace_path)
    # Validate manifest
    assert len(manifest["files"]) == 1
    assert manifest["files"][0]["filename"] == "sample.txt"
    # Validate trace file
    with open(trace_path, "r", encoding="utf-8") as f:
        trace = json.load(f)
    assert trace["group_description"] == "Test group for manifest workflow."
    assert "FileReaderAgent" in trace["agents"]
    assert "SummarizerAgent" in trace["agents"]
    assert "ValidatorAgent" in trace["agents"]
    action_types = [a["action_type"] for a in trace["actions"]]
    assert "file_read" in action_types
    assert "file_summarized" in action_types
    assert "file_added_to_manifest" in action_types
    assert "manifest_validated" in action_types
    # Clean up
    sample_file.unlink()
    trace_path.unlink() 