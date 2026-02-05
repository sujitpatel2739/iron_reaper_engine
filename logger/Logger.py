import logging
import logging.handlers
import os
import sys
import threading
from typing import Optional


class Logger:
    """Centralized, non-instantiable logger for the Engine project.

    - Non-instantiable: trying to create an instance raises TypeError.
    - Use class methods like `Logger.info(...)`, `Logger.configure(...)`.
    - Under the hood it wraps a single `logging.Logger` instance.
    """

    _name = "Engine"
    _logger = logging.getLogger(_name)
    _configured = False
    _lock = threading.RLock()

    def __new__(cls, *args, **kwargs):
        raise TypeError("Logger is non-instantiable. Use class methods instead.")

    @classmethod
    def configure(
        cls,
        log_file: Optional[str] = None,
        level: int = logging.INFO,
        console: bool = True,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
    ) -> None:
        """Configure the central logger.

        Args:
            log_file: optional path to a log file (uses rotating file handler).
            level: logging level.
            console: enable stream handler to stdout.
            max_bytes: rotate file when it exceeds this size.
            backup_count: how many rotated files to keep.
            fmt: optional format string for log messages.
            datefmt: optional date format string.
        """
        with cls._lock:
            # Remove existing handlers when reconfiguring
            if cls._configured:
                for h in list(cls._logger.handlers):
                    cls._logger.removeHandler(h)

            cls._logger.setLevel(level)

            fmt = fmt or "%(asctime)s | %(levelname)s | %(name)s | %(module)s:%(lineno)d | %(message)s"
            datefmt = datefmt or "%Y-%m-%d %H:%M:%S"
            formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

            if console:
                sh = logging.StreamHandler(sys.stdout)
                sh.setFormatter(formatter)
                cls._logger.addHandler(sh)

            if log_file:
                log_dir = os.path.dirname(log_file)
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)
                fh = logging.handlers.RotatingFileHandler(
                    log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
                )
                fh.setFormatter(formatter)
                cls._logger.addHandler(fh)

            cls._configured = True

    @classmethod
    def get_logger(cls) -> logging.Logger:
        """Return the underlying `logging.Logger` instance (configures defaults if needed)."""
        if not cls._configured:
            # sensible default: console output only
            cls.configure()
        return cls._logger

    # Convenience wrappers
    @classmethod
    def debug(cls, msg: str, *args, **kwargs) -> None:
        cls.get_logger().debug(msg, *args, **kwargs)

    @classmethod
    def info(cls, msg: str, *args, **kwargs) -> None:
        cls.get_logger().info(msg, *args, **kwargs)

    @classmethod
    def warning(cls, msg: str, *args, **kwargs) -> None:
        cls.get_logger().warning(msg, *args, **kwargs)

    @classmethod
    def warn(cls, msg: str, *args, **kwargs) -> None:  # backward compat
        cls.get_logger().warning(msg, *args, **kwargs)

    @classmethod
    def error(cls, msg: str, *args, **kwargs) -> None:
        cls.get_logger().error(msg, *args, **kwargs)

    @classmethod
    def exception(cls, msg: str, *args, **kwargs) -> None:
        cls.get_logger().exception(msg, *args, **kwargs)

    @classmethod
    def critical(cls, msg: str, *args, **kwargs) -> None:
        cls.get_logger().critical(msg, *args, **kwargs)

    @classmethod
    def log(cls, level: int, msg: str, *args, **kwargs) -> None:
        cls.get_logger().log(level, msg, *args, **kwargs)

    # Handler management and utilities
    @classmethod
    def add_handler(cls, handler: logging.Handler) -> None:
        with cls._lock:
            cls.get_logger().addHandler(handler)

    @classmethod
    def remove_handler(cls, handler: logging.Handler) -> None:
        with cls._lock:
            cls.get_logger().removeHandler(handler)

    @classmethod
    def set_level(cls, level: int) -> None:
        cls.get_logger().setLevel(level)


__all__ = ["Logger"]
