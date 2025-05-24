# from src.config import load_config, save_config, Config
# TODO: Refactor this test to use config logic from autogen_extensions or another valid location.

from autogen_extensions.config_utils import load_config, save_config
import tempfile
import os


def test_load_and_save_config():
    config = {"section": {"key": "value"}}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        file_path = tmp.name
    try:
        save_config(config, file_path)
        loaded = load_config(file_path)
        assert loaded == config
    finally:
        os.remove(file_path)
