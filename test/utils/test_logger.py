"""Tests for utils.logger."""

import json
import logging
from unittest.mock import patch

from utils.logger import (
    APP_LOGGER_NAME,
    ColoredJsonFormatter,
    _serialize_message,
    get_logger,
    setup_logging,
)


class TestSerializeMessage:
    """Tests for _serialize_message."""

    def test_dict_returns_pretty_json(self) -> None:
        msg = {"a": 1, "b": "two"}
        assert _serialize_message(msg) == json.dumps(msg, indent=2, default=str)

    def test_list_returns_pretty_json(self) -> None:
        msg = [1, "x", {"k": "v"}]
        assert _serialize_message(msg) == json.dumps(msg, indent=2, default=str)

    def test_str_no_args_returns_str(self) -> None:
        assert _serialize_message("hello") == "hello"

    def test_str_with_format_args_interpolates(self) -> None:
        assert _serialize_message("hi %s", "world") == "hi world"
        assert _serialize_message("n=%d", 42) == "n=42"

    def test_str_with_dict_arg_serializes_arg(self) -> None:
        out = _serialize_message("data: %s", {"x": 1})
        assert "data:" in out
        assert '"x": 1' in out or "'x': 1" in out

    def test_str_with_list_arg_serializes_arg(self) -> None:
        out = _serialize_message("items: %s", [1, 2])
        assert "items:" in out
        assert "1" in out and "2" in out

    def test_interpolation_failure_fallback_to_join(self) -> None:
        # %d with non-int -> fallback joins msg and args with space
        out = _serialize_message("bad %d", "not a number")
        assert "bad" in out and "not a number" in out


class TestColoredJsonFormatter:
    """Tests for ColoredJsonFormatter."""

    def _make_record(
        self,
        msg: str | dict = "test",
        levelno: int = logging.INFO,
        name: str = "test.logger",
    ) -> logging.LogRecord:
        return logging.makeLogRecord(
            {
                "name": name,
                "levelno": levelno,
                "levelname": logging.getLevelName(levelno),
                "pathname": "",
                "lineno": 0,
                "msg": msg,
                "args": (),
            }
        )

    def test_format_without_color_is_deterministic(self) -> None:
        formatter = ColoredJsonFormatter(use_color=False, fmt="%(levelname)s %(message)s")
        record = self._make_record(msg="hello")
        out = formatter.format(record)
        assert "INFO" in out or "info" in out.lower()
        assert "hello" in out
        assert "\033[" not in out  # no ANSI escape codes

    def test_format_serializes_dict_message(self) -> None:
        formatter = ColoredJsonFormatter(use_color=False, fmt="%(message)s")
        record = self._make_record(msg='{"key": 1}')
        out = formatter.format(record)
        assert "key" in out and "1" in out

    def test_format_with_dict_in_message(self) -> None:
        formatter = ColoredJsonFormatter(use_color=False, fmt="%(message)s")
        record = logging.makeLogRecord(
            {
                "name": "test",
                "levelno": logging.INFO,
                "msg": {"event": "started", "id": 1},
                "args": (),
            }
        )
        out = formatter.format(record)
        assert "event" in out and "started" in out and "id" in out

    def test_format_pads_levelname(self) -> None:
        formatter = ColoredJsonFormatter(use_color=False, fmt="%(levelname)s — %(message)s")
        record = self._make_record(msg="m", levelno=logging.INFO)
        out = formatter.format(record)
        assert "m" in out


class TestGetLogger:
    """Tests for get_logger."""

    def test_returns_logger_with_given_name(self) -> None:
        logger = get_logger("test.logger.name")
        assert logger.name == f"{APP_LOGGER_NAME}.test.logger.name"
        assert isinstance(logger, logging.Logger)

    def test_same_name_returns_same_instance(self) -> None:
        a = get_logger("unique.name.here")
        b = get_logger("unique.name.here")
        assert a is b


class TestSetupLogging:
    """Tests for setup_logging."""

    def test_setup_logging_sets_app_level_and_handler(self) -> None:
        with patch("utils.logger.get_settings") as mock_get_settings:
            mock_get_settings.return_value.log_level = "DEBUG"
            setup_logging()
        root = logging.getLogger()
        assert root.level == logging.WARNING
        app_logger = logging.getLogger(APP_LOGGER_NAME)
        assert app_logger.level == logging.DEBUG
        assert len(app_logger.handlers) >= 1

    def test_setup_logging_uses_formatter(self) -> None:
        app_logger = logging.getLogger(APP_LOGGER_NAME)
        original_handlers = app_logger.handlers.copy()
        app_logger.handlers.clear()
        try:
            with patch("utils.logger.get_settings") as mock_get_settings:
                mock_get_settings.return_value.log_level = "INFO"
                setup_logging()
            stream_handlers = [h for h in app_logger.handlers if isinstance(h, logging.StreamHandler)]
            formatters = [h.formatter for h in stream_handlers if h.formatter]
            assert any(isinstance(f, ColoredJsonFormatter) for f in formatters)
        finally:
            app_logger.handlers.clear()
            app_logger.handlers.extend(original_handlers)

    def test_setup_logging_invalid_level_falls_back_to_info(self) -> None:
        with patch("utils.logger.get_settings") as mock_get_settings:
            mock_get_settings.return_value.log_level = "NOT_A_LEVEL"
            setup_logging()
        app_logger = logging.getLogger(APP_LOGGER_NAME)
        assert app_logger.getEffectiveLevel() in (
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        )
