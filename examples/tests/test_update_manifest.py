import os
import tempfile
import pytest
from pathlib import Path
from examples.update_manifest import (
    compute_sha256,
    build_manifest_with_agents,
    ValidatorAgent,
    llm_config,
    MINIMAL_MANIFEST_SCHEMA,
)

def temp_text_dir():
    d = tempfile.mkdtemp()
    yield d
    # Clean up
    for f in os.listdir(d):
        os.remove(os.path.join(d, f))
    os.rmdir(d)

@pytest.fixture
def temp_text_dir_fixture():
    yield from temp_text_dir()

def test_compute_sha256(temp_text_dir_fixture):
    file_path = os.path.join(temp_text_dir_fixture, "test.txt")
    content = "hello world"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    expected = compute_sha256(file_path)
    import hashlib
    m = hashlib.sha256()
    m.update(content.encode("utf-8"))
    assert expected == m.hexdigest()

def test_build_manifest_with_agents(temp_text_dir_fixture):
    Path(temp_text_dir_fixture, "a.txt").write_text("A", encoding="utf-8")
    Path(temp_text_dir_fixture, "b.txt").write_text("B", encoding="utf-8")
    manifest = build_manifest_with_agents(temp_text_dir_fixture)
    assert manifest["version"] == "1.0.0"
    assert len(manifest["files"]) == 2
    assert manifest["metadata"]["statistics"]["total_files"] == 2

def test_validator_agent_validates_manifest(temp_text_dir_fixture):
    Path(temp_text_dir_fixture, "a.txt").write_text("A", encoding="utf-8")
    manifest = build_manifest_with_agents(temp_text_dir_fixture)
    validator = ValidatorAgent("ValidatorAgent", llm_config, [], "Validates the manifest against the schema.")
    assert validator.validate(manifest, MINIMAL_MANIFEST_SCHEMA)
    # Remove required field to trigger error
    manifest.pop("version")
    assert not validator.validate(manifest, MINIMAL_MANIFEST_SCHEMA)
