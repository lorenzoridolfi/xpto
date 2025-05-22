from src.base_agent_system import (
    setup_logging,
    log_event,
    load_json_file,
    save_json_file,
)


def test_setup_logging_creates_logger():
    logger = setup_logging(
        {
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "test.log",
            }
        }
    )
    assert logger is not None
    logger.info("Test log message")


def test_log_event_writes_log(tmp_path):
    logger = setup_logging(
        {
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": str(tmp_path / "test.log"),
            }
        }
    )
    log_event(logger, "agent", "event", [], "output")
    # No exception means pass (actual log file not checked here)


def test_load_and_save_json_file(tmp_path):
    data = {"a": 1}
    file_path = tmp_path / "test.json"
    save_json_file(data, str(file_path))
    loaded = load_json_file(str(file_path))
    assert loaded == data
