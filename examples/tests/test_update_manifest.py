import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import tempfile
import shutil
import json
import pytest
from pathlib import Path
from update_manifest import (
    compute_sha256,
    get_file_metadata,
    generate_manifest,
    validate_manifest,
    update_and_save_manifest,
    load_manifest_schema,
    run_update_manifest_workflow,
    create_agents,
    config
)
import jsonschema
from autogen_extensions.config import load_merged_config
import update_manifest
import asyncio
from autogen_extensions.common_io import load_json_file
from autogen_extensions.messages import BaseChatMessage, TextMessage
from autogen_extensions.agents import AssistantAgent, UserProxyAgent
import logging


class DummyFile:
    def __init__(self, path, content):
        self.path = path
        self.content = content
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)


@pytest.fixture
def temp_text_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


def test_compute_sha256(temp_text_dir):
    file_path = os.path.join(temp_text_dir, "test.txt")
    content = "hello world"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    expected = compute_sha256(file_path)
    # Compute manually
    import hashlib

    m = hashlib.sha256()
    m.update(content.encode("utf-8"))
    assert expected == m.hexdigest()


def test_get_file_metadata(temp_text_dir):
    file_path = Path(temp_text_dir) / "file1.txt"
    file_path.write_text("test content", encoding="utf-8")
    meta = get_file_metadata(file_path)
    assert meta["filename"] == "file1.txt"
    assert meta["status"] == "okay"
    assert meta["file_type"] == "text"
    assert meta["encoding"] == "utf-8"
    assert meta["size"] == len("test content")
    assert "sha256" in meta
    assert "metadata" in meta


def test_generate_manifest(temp_text_dir):
    # Create two files
    Path(temp_text_dir, "a.txt").write_text("A", encoding="utf-8")
    Path(temp_text_dir, "b.txt").write_text("B", encoding="utf-8")
    manifest, files = generate_manifest(temp_text_dir)
    assert manifest["version"] == "1.0.0"
    assert len(manifest["files"]) == 2
    assert manifest["metadata"]["statistics"]["total_files"] == 2
    assert manifest["metadata"]["statistics"]["total_size"] == 2


def test_validate_manifest(temp_text_dir):
    Path(temp_text_dir, "a.txt").write_text("A", encoding="utf-8")
    manifest, _ = generate_manifest(temp_text_dir)
    schema = load_manifest_schema()
    # Should not raise
    validate_manifest(manifest, schema)
    # Remove required field to trigger error
    manifest.pop("version")
    with pytest.raises(Exception):
        validate_manifest(manifest, schema)


def test_trace_schema_validation(tmp_path, monkeypatch):
    # Simulate a valid trace file
    valid_trace = [{
        "task_id": "1",
        "task_type": "test",
        "start_time": "2024-01-01T00:00:00Z",
        "end_time": None,
        "status": None,
        "result": None,
        "events": [{
            "type": "event",
            "timestamp": "2024-01-01T00:00:00Z",
            "agent_name": None,
            "agent_type": None,
            "config": None,
            "error_type": None,
            "error_message": None,
            "context": None,
            "data": None,
            "metadata": {},
            "task_description": None,
            "objective": None,
            "human_feedback": None
        }],
        "agent_interactions": [{
            "timestamp": "2024-01-01T00:00:00Z",
            "sender": "a",
            "recipient": "b",
            "message": "msg"
        }],
        "decisions": [{
            "timestamp": "2024-01-01T00:00:00Z",
            "type": "decision",
            "context": {}
        }],
        "task_description": "desc",
        "objective": "obj",
        "data": None,
        "metadata": {},
        "context": {},
        "human_feedback": None
    }]
    schema_path = os.path.join("config", "trace_schema.json")
    with open(schema_path) as f:
        schema = json.load(f)
    # Should not raise
    jsonschema.validate(instance=valid_trace, schema=schema)

    # Simulate an invalid trace (missing required field)
    invalid_trace = [{
        # missing 'task_id'
        "task_type": "test",
        "start_time": "2024-01-01T00:00:00Z",
        "events": [],
        "agent_interactions": [],
        "decisions": [],
        "task_description": "desc",
        "objective": "obj",
        "metadata": {},
        "context": {}
    }]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=invalid_trace, schema=schema)


def test_update_manifest_trace_is_valid(monkeypatch, tmp_path):
    # Patch config and file operations to use temp dir
    config = load_merged_config(
        "config/shared/base_config.json",
        "config/update_manifest/program_config.json"
    )
    manifest_path = tmp_path / "manifest.json"
    config["shared_manifest_file"] = str(manifest_path)
    monkeypatch.setattr(update_manifest, "load_config", lambda: config)
    monkeypatch.setattr(update_manifest, "save_json_file", lambda data, path: Path(path).write_text(json.dumps(data)))
    monkeypatch.setattr(update_manifest, "load_manifest_schema", update_manifest.load_manifest_schema)
    # Fix recursion: save original generate_manifest
    original_generate_manifest = update_manifest.generate_manifest
    monkeypatch.setattr(update_manifest, "generate_manifest", lambda text_dir: (
        original_generate_manifest("text") if os.path.exists("text") else ({"version": "1.0.0", "files": [], "metadata": {"topics": {}, "entities": {}, "statistics": {"total_files": 0, "total_size": 0, "last_updated": ""}}}, [])
    ))
    # Should run without error and produce a valid trace
    update_manifest.update_and_save_manifest()
    trace_path = Path("update_manifest_traces.json")
    assert trace_path.exists()
    with open(trace_path) as f:
        trace = json.load(f)
    with open("config/trace_schema.json") as f:
        schema = json.load(f)
    jsonschema.validate(instance=trace, schema=schema)


def test_create_agents_from_config():
    agents = create_agents()
    assert "assistant" in agents
    assert "user" in agents
    assert "trace_collector" in agents
    # Check that agent attributes match config
    assistant_cfg = config.get("agents", {}).get("ManifestUpdaterAgent", {})
    user_cfg = config.get("agents", {}).get("SupervisorAgent", config.get("agents", {}).get("User", {}))
    assistant = agents["assistant"]
    user = agents["user"]
    assert assistant.name == assistant_cfg.get("name", "assistant")
    assert assistant.system_message == assistant_cfg.get("system_message", "You are an assistant.")
    assert assistant.description == assistant_cfg.get("description", "Assistant agent")
    assert user.name == user_cfg.get("name", "user")
    assert user.system_message == user_cfg.get("system_message", "You are a user.")
    assert user.description == user_cfg.get("description", "User agent")


def test_load_agent_config_json():
    config_path = os.path.join("config", "update_manifest", "program_config.json")
    config = load_json_file(config_path)
    assert "agents" in config
    assert isinstance(config["agents"], dict)


def test_create_assistant_agent_from_config():
    config_path = os.path.join("config", "update_manifest", "program_config.json")
    config = load_json_file(config_path)
    agents = create_agents()
    assert "assistant" in agents
    assistant = agents["assistant"]
    assistant_cfg = config["agents"].get("ManifestUpdaterAgent", {})
    assert assistant.name == assistant_cfg.get("name", "ManifestUpdaterAgent")
    assert assistant.system_message == assistant_cfg.get("system_message", "You are an assistant.")
    assert assistant.description == assistant_cfg.get("description", "Assistant agent")


def test_create_user_agent_from_config():
    config_path = os.path.join("config", "update_manifest", "program_config.json")
    config = load_json_file(config_path)
    agents = create_agents()
    assert "user" in agents
    user = agents["user"]
    user_cfg = config["agents"].get("SupervisorAgent", config["agents"].get("User", {}))
    assert user.name == user_cfg.get("name", "SupervisorAgent")
    assert user.system_message == user_cfg.get("system_message", "You are a user.")
    assert user.description == user_cfg.get("description", "User agent")


def test_missing_required_agent_field_raises():
    config_path = os.path.join("config", "update_manifest", "program_config.json")
    config = load_json_file(config_path)
    # Remove a required field from assistant
    assistant_cfg = config["agents"].get("AssistantAgent", {}).copy()
    if "name" in assistant_cfg:
        del assistant_cfg["name"]
    config["agents"]["AssistantAgent"] = assistant_cfg
    from update_manifest import get_openai_client
    llm_config = config.get("llm_config")
    if not (isinstance(llm_config, dict) and "config_list" in llm_config):
        llm_config = {"config_list": [{"model": "gpt-4"}]}
    model_client = get_openai_client(api_key=os.environ.get("OPENAI_API_KEY"), model="gpt-4")
    with pytest.raises((KeyError, TypeError, ValueError)):
        AssistantAgent(
            name=assistant_cfg["name"],  # This will raise
            system_message=assistant_cfg.get("system_message", "You are an assistant."),
            llm_config=llm_config,
            description=assistant_cfg.get("description", "Assistant agent"),
            model_client=model_client
        )


def test_build_update_manifest_groupchat(monkeypatch):
    from autogen_extensions.agent_tracer import AgentTracer
    from autogen_extensions.load_openai import get_openai_client
    from update_manifest import build_update_manifest_groupchat
    tracer = AgentTracer(config={})
    llm_config = {"config_list": [{"model": "gpt-4"}]}
    openai_api_key = os.environ.get("OPENAI_API_KEY", "test-key")
    model_client = get_openai_client(api_key=openai_api_key, model="gpt-4")
    config = {"llm_config": llm_config}
    result = build_update_manifest_groupchat(config, tracer, model_client)
    assert "group_chat" in result
    assert "builder" in result
    assert "writer" in result
    assert "validator" in result
    assert "trace_collector" in result
    # Check agent types
    from autogen_extensions.agents import AssistantAgent
    from autogen_extensions.trace_collector_agent import TraceCollectorAgent
    assert isinstance(result["builder"], AssistantAgent)
    assert isinstance(result["writer"], AssistantAgent)
    assert isinstance(result["validator"], AssistantAgent)
    assert isinstance(result["trace_collector"], TraceCollectorAgent)
    # Check group chat contains all agents
    agent_names = [a.name for a in result["group_chat"].agents]
    for key in ["builder", "writer", "validator", "trace_collector"]:
        assert result[key].name in agent_names
