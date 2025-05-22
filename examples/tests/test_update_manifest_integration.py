import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from examples.update_manifest import main as update_manifest_main

@pytest.mark.asyncio
async def test_update_manifest_workflow_creates_manifest_and_trace(tmp_path):
    # Setup: create a text directory with sample files in the expected location
    project_root = Path(__file__).parent.parent
    text_dir = project_root / "text"
    text_dir.mkdir(exist_ok=True)
    sample_file = text_dir / "sample.txt"
    sample_file.write_text("This is a test file for manifest integration.", encoding="utf-8")

    # Remove manifest and trace if they exist
    manifest_path = project_root / "manifest.json"
    trace_path = project_root / "update_manifest_trace.json"
    if manifest_path.exists():
        manifest_path.unlink()
    if trace_path.exists():
        trace_path.unlink()

    # Patch OpenAI LLM call to return a dummy summary instantly
    with patch("openai.chat.completions.create") as mock_create:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "DESCRIÇÃO: Arquivo de teste\nRESUMO: Resumo de teste"
        )
        mock_create.return_value = mock_response

        # Run the workflow with reduced rounds
        await update_manifest_main(max_round=1)

    # Check manifest file
    assert manifest_path.exists(), "Manifest file was not created."
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert "version" in manifest
    assert "files" in manifest
    assert any(f["filename"] == "sample.txt" for f in manifest["files"])

    # Check trace file
    assert trace_path.exists(), "Trace file was not created."
    with open(trace_path, "r", encoding="utf-8") as f:
        trace = json.load(f)
    assert "actions" in trace
    # Check for at least one group chat action
    assert any(a["action_type"] in ("message_sent", "message_received") for a in trace["actions"])

    # Cleanup
    sample_file.unlink()
    if manifest_path.exists():
        manifest_path.unlink()
    if trace_path.exists():
        trace_path.unlink()
    if text_dir.exists():
        text_dir.rmdir() 