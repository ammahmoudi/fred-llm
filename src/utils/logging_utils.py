"""
Logging utilities for Fred-LLM.

Provides consistent logging configuration across the project.
"""

import logging
import sys
from pathlib import Path
from typing import Any


def get_logger(name: str, level: str | int = logging.INFO) -> logging.Logger:
    """
    Get a configured logger.

    Args:
        name: Logger name (typically __name__).
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)

    return logger


def setup_logging(
    level: str = "INFO",
    log_file: Path | str | None = None,
    format_string: str | None = None,
) -> None:
    """
    Setup global logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional file path for logging output.
        format_string: Optional custom format string.
    """
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    level_int = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=level_int,
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


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
