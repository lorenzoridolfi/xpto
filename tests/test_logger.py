from src.logger import get_logger, LoggerMixin


def test_get_logger_returns_logger():
    logger = get_logger("test")
    assert logger is not None
    logger.info("Logger works")


class Dummy(LoggerMixin):
    pass


def test_logger_mixin_methods():
    d = Dummy()
    d.log_info("info")
    d.log_warning("warn")
    d.log_error("error", exc_info=False)
    d.log_debug("debug")
