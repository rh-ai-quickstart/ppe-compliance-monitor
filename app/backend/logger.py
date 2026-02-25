import logging
import os
import sys


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_initialized = False


def _init_root_logger():
    global _initialized
    if _initialized:
        return
    _initialized = True

    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with consistent formatting.

    Usage:
        from logger import get_logger
        log = get_logger(__name__)
        log.info("Server started on port %s", port)
    """
    _init_root_logger()
    return logging.getLogger(name)
