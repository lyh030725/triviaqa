"""Controls and metadata helpers for reproducible generation runs."""

from __future__ import annotations

import importlib
import platform
import random
from importlib import metadata
from typing import Any

import numpy as np


def _torch() -> Any | None:
    """Import PyTorch only when a caller needs its optional functionality."""
    try:
        return importlib.import_module("torch")
    except (ImportError, OSError):
        return None


def set_seed(seed: int) -> None:
    """Reset all available random-number generators for a generation run."""
    random.seed(seed)
    np.random.seed(seed)

    torch = _torch()
    if torch is None:
        return

    torch.manual_seed(seed)
    cuda = getattr(torch, "cuda", None)
    if cuda is not None:
        manual_seed_all = getattr(cuda, "manual_seed_all", None)
        if callable(manual_seed_all):
            manual_seed_all(seed)

    use_deterministic_algorithms = getattr(torch, "use_deterministic_algorithms", None)
    if callable(use_deterministic_algorithms):
        use_deterministic_algorithms(True)

    cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
    if cudnn is not None:
        cudnn.deterministic = True
        cudnn.benchmark = False


def package_version(name: str) -> str | None:
    """Return the installed distribution version, or ``None`` when absent."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _torch_environment(torch: Any) -> dict[str, object]:
    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    cuda_available = bool(is_available()) if callable(is_available) else False

    environment: dict[str, object] = {
        "version": getattr(torch, "__version__", None),
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "cuda_available": cuda_available,
        "gpu_name": None,
        "bf16_supported": None,
    }
    if not cuda_available:
        return environment

    get_device_name = getattr(cuda, "get_device_name", None)
    if callable(get_device_name):
        try:
            environment["gpu_name"] = get_device_name(0)
        except (RuntimeError, TypeError):
            pass

    is_bf16_supported = getattr(cuda, "is_bf16_supported", None)
    if callable(is_bf16_supported):
        try:
            environment["bf16_supported"] = bool(is_bf16_supported())
        except RuntimeError:
            pass

    return environment


def collect_runtime_environment() -> dict[str, object]:
    """Capture runtime and optional PyTorch metadata without requiring a GPU."""
    environment: dict[str, object] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in ("triviaqa-tts", "numpy", "datasets", "huggingface-hub", "soundfile", "soxr", "qwen-tts")
        },
    }
    torch = _torch()
    if torch is not None:
        environment["torch"] = _torch_environment(torch)
    return environment
