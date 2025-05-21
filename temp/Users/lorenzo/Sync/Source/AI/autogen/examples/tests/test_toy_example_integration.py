import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import pytest
import asyncio
from toy_example import run_toy_example_workflow
from autogen_extensions.messages import BaseChatMessage, TextMessage

@pytest.mark.asyncio
async def test_toy_example_workflow_runs():
    document_path = "test_document.txt"
    test_config = {
        "cache_config": {"max_size": 10, "similarity_threshold": 0.8, "expiration_hours": 1},
        "logging": {},
        "agent_settings": {}
    }
    output = await run_toy_example_workflow(document_path, config_override=test_config)
    assert "trace" in output
    assert "agents" in output
    assert output["result"] is not None
    assert len(output["trace"]) > 0 