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
from unittest.mock import patch, MagicMock
from examples.common import JsonSchemaValidationError


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


@pytest.mark.asyncio
def test_update_manifest_workflow_trace():
    # Create a dummy manifest file
    dummy_manifest = {
        "version": "1.0.0",
        "files": [],
        "metadata": {"topics": {}, "entities": {}, "statistics": {"total_files": 0, "total_size": 0, "last_updated": ""}}
    }
    manifest_path = "test_update_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(dummy_manifest, f)

    try:
        # Use a stub requirements and config
        update_requirements = {
            "type": "dependency_update",
            "target_dependencies": ["test_dep"],
            "version_constraints": {"test_dep": ">=1.0.0"},
        }
        test_config = {
            "cache_config": {"max_size": 10, "similarity_threshold": 0.8, "expiration_hours": 1},
            "logging": {},
            "agent_settings": {},
            "shared_manifest_file": manifest_path
        }
        output = asyncio.get_event_loop().run_until_complete(run_update_manifest_workflow(update_requirements, config_override=test_config))
        trace = output["trace"]
        agents = output["agents"]
        assert len(trace) > 0
        agent_names = [a.name for a in agents]
        sources = {msg["source"] for msg in trace if "source" in msg}
        for name in agent_names:
            assert name in sources
        for msg in trace:
            assert "agent_name" in msg
            assert "role" in msg
            assert "content" in msg
            assert "source" in msg
        assert output["result"] is not None
    finally:
        if os.path.exists(manifest_path):
            os.remove(manifest_path)


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
    from update_manifest import AgentTracer, get_openai_client
    tracer = AgentTracer(config)
    llm_config = config.get("llm_config")
    if not (isinstance(llm_config, dict) and "config_list" in llm_config):
        llm_config = {"config_list": [{"model": "gpt-4"}]}
    model_client = get_openai_client(api_key=os.environ.get("OPENAI_API_KEY"), model="gpt-4")
    from autogen import AssistantAgent
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


@pytest.fixture
def mock_config():
    """Mock agent configuration."""
    return {
        "agents": {
            "ManifestBuilder": {
                "system_message": "Build manifest",
            },
            "ManifestValidator": {
                "system_message": "Validate manifest",
            },
            "ManifestWriter": {
                "system_message": "Write manifest",
            },
        }
    }

@pytest.fixture
def mock_manifest():
    """Mock manifest data."""
    return {
        "files": [
            {
                "path": "test.py",
                "metadata": {"type": "python"}
            }
        ]
    }

@pytest.fixture
def mock_schema():
    """Mock manifest schema."""
    return {
        "type": "object",
        "properties": {
            "files": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "metadata": {"type": "object"}
                    },
                    "required": ["path", "metadata"]
                }
            }
        },
        "required": ["files"]
    }

@pytest.mark.asyncio
async def test_create_agents(tmp_path, mock_config):
    """Test agent creation with logging."""
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(mock_config, f)
    
    with patch("logging.getLogger") as mock_logger:
        agents = await create_agents(str(config_path))
        
        # Verify logging calls
        mock_logger.return_value.info.assert_any_call(f"Loading agent configuration from {config_path}")
        mock_logger.return_value.info.assert_any_call("Creating manifest workflow agents")
        
        # Verify agents created
        assert "manifest_builder" in agents
        assert "manifest_validator" in agents
        assert "manifest_writer" in agents
        assert "trace_collector" in agents
        
        # Verify agent configuration
        assert agents["manifest_builder"].name == "ManifestBuilder"
        assert agents["manifest_validator"].name == "ManifestValidator"
        assert agents["manifest_writer"].name == "ManifestWriter"
        assert agents["trace_collector"].name == "TraceCollector"

@pytest.mark.asyncio
async def test_run_manifest_update(tmp_path, mock_manifest, mock_schema):
    """Test manifest update workflow with logging."""
    # Create test files
    manifest_path = tmp_path / "manifest.json"
    schema_path = tmp_path / "schema.json"
    with open(manifest_path, "w") as f:
        json.dump(mock_manifest, f)
    with open(schema_path, "w") as f:
        json.dump(mock_schema, f)
    
    # Create mock agents
    mock_agents = {
        "manifest_builder": MagicMock(),
        "manifest_validator": MagicMock(),
        "manifest_writer": MagicMock(),
        "trace_collector": MagicMock(get_trace=MagicMock(return_value={"events": []})),
    }
    
    with patch("logging.getLogger") as mock_logger:
        await run_manifest_update(
            mock_agents,
            manifest_path=str(manifest_path),
            schema_path=str(schema_path)
        )
        
        # Verify logging calls
        mock_logger.return_value.info.assert_any_call("Initializing GroupChat for manifest update workflow")
        mock_logger.return_value.info.assert_any_call("Starting manifest update workflow")
        mock_logger.return_value.info.assert_any_call("Completed GroupChat workflow execution")
        
        # Verify trace was saved
        assert (Path.cwd() / "update_manifest_trace.json").exists()

@pytest.mark.asyncio
async def test_manifest_validation_error(tmp_path, mock_manifest):
    """Test handling of manifest validation errors."""
    manifest_path = tmp_path / "manifest.json"
    schema_path = tmp_path / "schema.json"
    
    # Create invalid manifest
    invalid_manifest = {"wrong_key": []}
    with open(manifest_path, "w") as f:
        json.dump(invalid_manifest, f)
    with open(schema_path, "w") as f:
        json.dump({"type": "object", "required": ["files"]}, f)
    
    mock_agents = {
        "manifest_builder": MagicMock(),
        "manifest_validator": MagicMock(),
        "manifest_writer": MagicMock(),
        "trace_collector": MagicMock(),
    }
    
    with patch("logging.getLogger") as mock_logger:
        await run_manifest_update(
            mock_agents,
            manifest_path=str(manifest_path),
            schema_path=str(schema_path)
        )
        
        # Verify error logging
        mock_logger.return_value.error.assert_called()

@pytest.mark.asyncio
async def test_main_workflow(tmp_path, mock_config, mock_manifest, mock_schema):
    """Test the complete workflow."""
    # Setup test files
    config_path = tmp_path / "config.json"
    manifest_path = tmp_path / "manifest.json"
    schema_path = tmp_path / "schema.json"
    
    with open(config_path, "w") as f:
        json.dump(mock_config, f)
    with open(manifest_path, "w") as f:
        json.dump(mock_manifest, f)
    with open(schema_path, "w") as f:
        json.dump(mock_schema, f)
    
    with patch("examples.update_manifest.DEFAULT_CONFIG_PATH", str(config_path)), \
         patch("examples.update_manifest.DEFAULT_MANIFEST_PATH", str(manifest_path)), \
         patch("examples.update_manifest.DEFAULT_SCHEMA_PATH", str(schema_path)), \
         patch("logging.getLogger") as mock_logger:
        
        await main()
        
        # Verify workflow completed
        mock_logger.return_value.info.assert_any_call("Starting manifest update workflow")
        mock_logger.return_value.info.assert_any_call("Manifest update workflow completed successfully")
        
        # Verify trace file was created
        assert (Path.cwd() / "update_manifest_trace.json").exists()
