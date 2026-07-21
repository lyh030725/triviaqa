"""Shared interface for text-to-speech backends."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class TTSBackend(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def native_sample_rate(self) -> int | None: ...

    def synthesize(self, text: str, seed: int) -> tuple[np.ndarray, int]: ...
