"""
Structured logging module for NexusRAG.
"""

import sys
import logging
from typing import Any

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("NexusRAG")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

logger = setup_logging()
