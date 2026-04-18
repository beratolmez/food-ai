"""Logging utilities for HealfoAI ML pipeline."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from datetime import datetime


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    """Create a configured logger instance.

    Args:
        name: Logger name (typically __name__ of the calling module).
        level: Logging level (default: INFO).
        log_file: Optional path to write logs to a file.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_timestamped_dir(base_dir: Path, prefix: str = "run") -> Path:
    """Create a timestamped directory for experiment outputs.

    Args:
        base_dir: Parent directory for the timestamped folder.
        prefix: Prefix for the folder name.

    Returns:
        Path to the created directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
