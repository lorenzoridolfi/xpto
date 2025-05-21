import os
import shutil
import tempfile
import pytest
import asyncio
from pathlib import Path

from src.file_reader_agent import FileReaderAgent
from src.writer_agent import WriterAgent
from src.verifier_agent import VerifierAgent
from src.quality_agent import QualityAgent
from src.toy_example_supervisor import ToyExampleSupervisor
from src.update_manifest_supervisor import UpdateManifestSupervisor
from src.base_supervisor import AgentError

# ---------------------- FileReaderAgent Tests ----------------------

def test_file_reader_reads_existing_file():
    temp_dir = tempfile.mkdtemp()
    file_path = Path(temp_dir) / "test.txt"
    file_path.write_text("hello world", encoding="utf-8")
    agent = FileReaderAgent({})
    agent.text_dir = Path(temp_dir)  # override for test
    result = agent.read_files(["test.txt"])
    assert result["test.txt"] == "hello world"
    shutil.rmtree(temp_dir)

def test_file_reader_handles_missing_file():
    agent = FileReaderAgent({})
    agent.text_dir = Path(tempfile.mkdtemp())  # override for test
    result = agent.read_files(["does_not_exist.txt"])
    assert "ERROR" in result["does_not_exist.txt"]

# ---------------------- WriterAgent Tests ----------------------

def test_writer_agent_writes_file():
    temp_dir = tempfile.mkdtemp()
    agent = WriterAgent({"output_dir": temp_dir})
    msg = agent.write("data", "out.txt")
    assert "Successfully wrote" in msg
    assert (Path(temp_dir) / "out.txt").read_text(encoding="utf-8") == "data"
    shutil.rmtree(temp_dir)

def test_writer_agent_handles_write_error():
    agent = WriterAgent({"output_dir": "/nonexistent_dir"})
    msg = agent.write("data", "out.txt")
    assert "ERROR" in msg

# ---------------------- VerifierAgent Tests ----------------------

def test_verifier_agent_valid_content():
    agent = VerifierAgent({})
    assert agent.verify("something") is True

def test_verifier_agent_invalid_content():
    agent = VerifierAgent({})
    assert agent.verify("") is False
    assert agent.verify("   ") is False

# ---------------------- QualityAgent Tests ----------------------

def test_quality_agent_good():
    agent = QualityAgent({})
    assert agent.assess_quality("this is a long enough string") == "good"

def test_quality_agent_poor():
    agent = QualityAgent({})
    assert agent.assess_quality("short") == "poor"

# ---------------------- Supervisor Tests ----------------------

def test_toy_example_supervisor_creates_known_agents():
    sup = ToyExampleSupervisor({})
    for name in ["file_reader", "writer", "verifier", "quality"]:
        agent = asyncio.run(sup._create_agent(name, {}))
        assert agent is not None

def test_toy_example_supervisor_handles_unknown_agent():
    sup = ToyExampleSupervisor({})
    with pytest.raises(AgentError):
        asyncio.run(sup._create_agent("unknown", {}))

def test_update_manifest_supervisor_creates_known_agents():
    sup = UpdateManifestSupervisor({})
    for name in ["file_reader"]:
        agent = asyncio.run(sup._create_agent(name, {}))
        assert agent is not None

def test_update_manifest_supervisor_handles_unknown_agent():
    sup = UpdateManifestSupervisor({})
    with pytest.raises(Exception):
        asyncio.run(sup._create_agent("unknown", {}))
