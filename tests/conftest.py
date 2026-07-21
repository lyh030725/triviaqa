from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from triviaqa_tts.config import AppConfig, PathsConfig, SelectionConfig
from triviaqa_tts.data.manifest import write_canonical_jsonl


class MockBackend:
    model_id = "mock/sine-tts"
    model_revision = "f" * 40
    dtype = "float32"
    attention_implementation = "mock"
    native_sample_rate = 24_000

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def synthesize(self, text: str, seed: int) -> tuple[np.ndarray, int]:
        self.calls.append((text, seed))
        sample_count = round(self.native_sample_rate * 0.4)
        time = np.arange(sample_count, dtype=np.float32) / self.native_sample_rate
        waveform = 0.2 * np.sin(2 * np.pi * 440 * time)
        return waveform.astype(np.float32), self.native_sample_rate


@pytest.fixture
def manifest_records() -> list[dict[str, object]]:
    return [
        {
            "dataset": "mandarjoshi/trivia_qa",
            "configuration": "unfiltered.nocontext",
            "split": "validation",
            "dataset_revision": "a" * 40,
            "original_index": 3,
            "question_id": "safe/../one",
            "question": "  Who  &amp; why ",
            "answer_value": "Because",
            "aliases": ["Because", "For a reason"],
            "normalized_aliases": ["because", "for a reason"],
        },
        {
            "dataset": "mandarjoshi/trivia_qa",
            "configuration": "unfiltered.nocontext",
            "split": "validation",
            "dataset_revision": "a" * 40,
            "original_index": 8,
            "question_id": "two",
            "question": "Already complete!",
            "answer_value": "Yes",
            "aliases": ["Yes"],
            "normalized_aliases": ["yes"],
        },
    ]


@pytest.fixture
def synthesis_config(tmp_path: Path, manifest_records: list[dict[str, object]]) -> AppConfig:
    manifest_path = tmp_path / "manifest.jsonl"
    write_canonical_jsonl(manifest_records, manifest_path)
    return AppConfig(
        selection=replace(SelectionConfig(), manifest_path=manifest_path),
        paths=PathsConfig(
            data_dir=tmp_path / "data",
            audio_dir=tmp_path / "audio",
            logs_dir=tmp_path / "logs",
            cache_dir=tmp_path / "cache",
        ),
    )
