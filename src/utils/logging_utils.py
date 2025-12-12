"""
Logging utilities for Fred-LLM.

Provides consistent logging configuration across the project.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Default log directory
DEFAULT_LOG_DIR = Path("logs")


def get_logger(
    name: str,
    level: str | int = logging.INFO,
    log_to_file: bool = False,
    log_dir: Path | str | None = None,
) -> logging.Logger:
    """
    Get a configured logger.

    Args:
        name: Logger name (typically __name__).
        level: Logging level.
        log_to_file: Whether to also log to a file.
        log_dir: Directory for log files (default: logs/).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler (if requested)
        if log_to_file:
            file_handler = _create_file_handler(log_dir or DEFAULT_LOG_DIR)
            logger.addHandler(file_handler)

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)

    return logger


def _create_file_handler(log_dir: Path | str) -> logging.FileHandler:
    """Create a file handler with timestamped log file."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"fred_llm_{timestamp}.log"

    handler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    return handler


def setup_logging(
    level: str = "INFO",
    log_file: Path | str | None = None,
    log_dir: Path | str | None = None,
    format_string: str | None = None,
    log_to_file: bool = False,
) -> Path | None:
    """
    Setup global logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional specific file path for logging output.
        log_dir: Optional directory for auto-generated log files.
        format_string: Optional custom format string.
        log_to_file: If True and no log_file specified, auto-generate log file.

    Returns:
        Path to log file if file logging is enabled, None otherwise.
    """
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    file_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s | "
        "%(funcName)s:%(lineno)d | %(message)s"
    )

    level_int = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    log_file_path: Path | None = None

    # Determine log file path
    if log_file:
        log_file_path = Path(log_file)
    elif log_to_file:
        log_dir_path = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
        log_dir_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = log_dir_path / f"fred_llm_{timestamp}.log"

    if log_file_path:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S")
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=level_int,
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

    if log_file_path:
        logging.info(f"Logging to file: {log_file_path}")

    return log_file_path


class LogContext:
    """Context manager for temporary logging configuration."""

    def __init__(
        self,
        logger: logging.Logger,
        level: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize log context.

        Args:
            logger: Logger to configure.
            level: Temporary logging level.
            extra: Extra context to add to log messages.
        """
        self.logger = logger
        self.level = level
        self.extra = extra or {}
        self._original_level: int | None = None

    def __enter__(self) -> logging.Logger:
        """Enter context and apply temporary configuration."""
        if self.level is not None:
            self._original_level = self.logger.level
            self.logger.setLevel(self.level)
        return self.logger

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and restore original configuration."""
        if self._original_level is not None:
            self.logger.setLevel(self._original_level)
