"""Shared utilities: logging, timing, seeding, memory helpers, path helpers."""
from __future__ import annotations

import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is a pinned dependency
    psutil = None


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


LOGGER = get_logger("business_entity_resolution")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def memory_usage_mb() -> Optional[float]:
    if psutil is None:
        return None
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


@contextmanager
def timer(label: str, logger: Optional[logging.Logger] = None) -> Iterator[None]:
    log = logger or LOGGER
    start = time.perf_counter()
    mem_before = memory_usage_mb()
    log.info("[START] %s", label)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        mem_after = memory_usage_mb()
        mem_str = ""
        if mem_before is not None and mem_after is not None:
            mem_str = f" | mem {mem_before:.0f}MB -> {mem_after:.0f}MB"
        log.info("[DONE]  %s | %.2fs%s", label, elapsed, mem_str)


def chunked(iterable, size: int) -> Iterator[List]:
    """Yield successive chunks (lists) of length `size` from a sequence-like iterable."""
    buf: List = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf
