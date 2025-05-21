import pytest
import asyncio
from update_manifest import run_update_manifest_workflow
import json
import os

@pytest.mark.asyncio
async def test_update_manifest_workflow_runs():
    manifest_path = os.path.join(os.path.dirname(__file__), "test_update_manifest.json")
    dummy_manifest = {
        "version": "1.0.0",
        "files": [],
        "metadata": {"topics": {}, "entities": {}, "statistics": {"total_files": 0, "total_size": 0, "last_updated": ""}}
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(dummy_manifest, f)

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
    try:
        output = await run_update_manifest_workflow(update_requirements, config_override=test_config)
        assert "trace" in output
        assert "agents" in output
        assert len(output["trace"]) > 0
    finally:
        if os.path.exists(manifest_path):
            os.remove(manifest_path) 