from src.config import load_config, save_config, Config


def test_load_and_save_config(tmp_path):
    config = {"section": {"key": "value"}}
    file_path = tmp_path / "config.json"
    save_config(config, str(file_path))
    loaded = load_config(str(file_path))
    assert loaded == config


def test_config_singleton():
    c1 = Config()
    c2 = Config()
    assert c1 is c2
