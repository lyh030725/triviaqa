from __future__ import annotations

import random
import sys
from types import SimpleNamespace
from typing import get_type_hints

import numpy as np

from triviaqa_tts import reproducibility


class FakeTorch:
    __version__ = "9.9.9"
    version = SimpleNamespace(cuda="12.8")

    def __init__(self) -> None:
        self.manual_seed_calls: list[int] = []
        self.cuda = SimpleNamespace(
            is_available=lambda: True,
            manual_seed_all=self.cuda_manual_seed_all,
            get_device_name=lambda index: f"Fake GPU {index}",
            is_bf16_supported=lambda: True,
        )
        self.backends = SimpleNamespace(cudnn=SimpleNamespace(deterministic=False, benchmark=True))

    def manual_seed(self, seed: int) -> None:
        self.manual_seed_calls.append(seed)

    def cuda_manual_seed_all(self, seed: int) -> None:
        self.cuda_seed_calls.append(seed)

    @property
    def cuda_seed_calls(self) -> list[int]:
        if not hasattr(self, "_cuda_seed_calls"):
            self._cuda_seed_calls: list[int] = []
        return self._cuda_seed_calls


def test_set_seed_repeats_python_and_numpy_sequences() -> None:
    reproducibility.set_seed(42)
    first = (random.random(), np.random.random())

    reproducibility.set_seed(42)
    second = (random.random(), np.random.random())

    assert second == first


def test_set_seed_configures_torch_determinism_when_available(monkeypatch) -> None:
    fake_torch = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    reproducibility.set_seed(42)

    assert fake_torch.manual_seed_calls == [42]
    assert fake_torch.cuda_seed_calls == [42]
    assert fake_torch.backends.cudnn.deterministic is True
    assert fake_torch.backends.cudnn.benchmark is False


def test_collect_runtime_environment_reports_torch_gpu_details(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())

    environment = reproducibility.collect_runtime_environment()

    assert environment["python_version"]
    assert environment["platform"]
    assert environment["packages"]
    assert environment["torch"] == {
        "version": "9.9.9",
        "cuda_version": "12.8",
        "cuda_available": True,
        "gpu_name": "Fake GPU 0",
        "bf16_supported": True,
    }


def test_tts_backend_protocol_has_requested_annotations() -> None:
    from triviaqa_tts.tts.base import TTSBackend

    hints = get_type_hints(TTSBackend.synthesize)

    assert hints["text"] is str
    assert hints["seed"] is int
    assert hints["return"] == tuple[np.ndarray, int]
