from __future__ import annotations

import importlib
import os
from pathlib import Path

import numpy as np
import pytest

from triviaqa_tts.audio import write_validated_wav
from triviaqa_tts.config import load_config
from triviaqa_tts.reproducibility import set_seed
from triviaqa_tts.tts.qwen3_backend import Qwen3Backend


@pytest.mark.gpu
def test_real_qwen_aiden_synthesis_writes_validated_24khz_wav(tmp_path: Path) -> None:
    if os.environ.get("RUN_QWEN_GPU_TEST") != "1":
        pytest.skip("set RUN_QWEN_GPU_TEST=1 to enable the real Qwen GPU test")

    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        pytest.skip("RUN_QWEN_GPU_TEST=1 but CUDA is unavailable")

    config = load_config(Path("configs/pilot.yaml"))
    assert config.tts.speaker == "Aiden"
    assert config.tts.language == "English"

    backend = Qwen3Backend.from_config(config.tts)
    set_seed(config.tts.seed)
    waveform, sample_rate = backend.synthesize("What is two plus two?", config.tts.seed)

    samples = np.asarray(waveform)
    assert sample_rate == 24_000
    assert samples.size > 0
    assert np.isfinite(samples).all()

    output = tmp_path / "qwen_aiden.wav"
    metrics = write_validated_wav(samples, sample_rate, output, config)
    assert output.stat().st_size > 0
    assert metrics.sample_rate == 24_000
    assert metrics.channels == 1
    assert metrics.subtype == "PCM_16"
