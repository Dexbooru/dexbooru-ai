import json
import logging
import sys
from typing import Any

from utils.config import get_settings

# ANSI color codes for terminal output
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"
_CYAN = "\033[36m"

_LEVEL_COLORS = {
    logging.DEBUG: _DIM,
    logging.INFO: _GREEN,
    logging.WARNING: _YELLOW,
    logging.ERROR: _RED,
    logging.CRITICAL: _BOLD + _RED,
}


def _serialize_message(msg: Any, *args: Any) -> str:
    """Convert message and args to string; serialize dicts/lists as pretty JSON."""
    if isinstance(msg, (dict, list)):
        return json.dumps(msg, indent=2, default=str)
    if args:
        formatted_args = tuple(json.dumps(a, indent=2, default=str) if isinstance(a, (dict, list)) else a for a in args)
        try:
            return str(msg) % formatted_args
        except (TypeError, ValueError):
            return " ".join((str(msg), *(str(a) for a in formatted_args)))
    return str(msg)


class ColoredJsonFormatter(logging.Formatter):
    """Formatter with colors, readable dates, and JSON serialization for dicts/lists."""

    DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self, use_color: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.use_color = use_color and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        record.msg = _serialize_message(record.msg, *record.args)
        record.args = ()

        levelname = record.levelname
        if self.use_color and record.levelno in _LEVEL_COLORS:
            color = _LEVEL_COLORS[record.levelno]
            record.levelname = f"{color}{levelname:8}{_RESET}"
            record.name = f"{_CYAN}{record.name}{_RESET}"
            record.asctime = f"{_DIM}{self.formatTime(record, self.DATE_FMT)}{_RESET}"
        else:
            record.levelname = f"{levelname:8}"
            record.asctime = self.formatTime(record, self.DATE_FMT)

        return super().format(record)


APP_LOGGER_NAME = "app"


def setup_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    if not isinstance(level, int):
        level = logging.INFO

    root = logging.getLogger()
    root.setLevel(logging.WARNING)

    app_logger = logging.getLogger(APP_LOGGER_NAME)
    app_logger.setLevel(level)
    if not app_logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        handler.setFormatter(
            ColoredJsonFormatter(
                use_color=True,
                fmt="%(asctime)s %(levelname)s %(name)s — %(message)s",
                datefmt=ColoredJsonFormatter.DATE_FMT,
            )
        )
        app_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{APP_LOGGER_NAME}.{name}")
