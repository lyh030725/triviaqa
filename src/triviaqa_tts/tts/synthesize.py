"""Deterministic record selection and strict synthesis resume primitives."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

from triviaqa_tts.audio.validation import AudioValidationError, validate_wav
from triviaqa_tts.config import AppConfig, AudioConfig
from triviaqa_tts.data.manifest import (
    ManifestFormatError,
    iter_jsonl,
    normalize_question,
    sha256_file,
)

_COMPLETE_STATUSES = frozenset({"success", "warning"})
_EXPECTED_TTS_FIELDS = (
    "model_id",
    "model_revision",
    "speaker",
    "language",
    "seed",
    "generation",
)


def select_records[T](
    records: Iterable[T],
    start_index: int,
    end_index: int | None,
    shard_index: int,
    num_shards: int,
    limit: int | None,
) -> list[T]:
    """Select a range, then a global-ordinal shard, then an optional limit."""
    _validate_selection(start_index, end_index, shard_index, num_shards, limit)

    selected: list[T] = []
    for ordinal, record in enumerate(records):
        if ordinal < start_index:
            continue
        if end_index is not None and ordinal >= end_index:
            break
        if ordinal % num_shards != shard_index:
            continue
        selected.append(record)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def load_latest_metadata(path: Path) -> dict[str, dict[str, object]]:
    """Load the last complete metadata record for every question ID."""
    metadata_path = Path(path)
    if not metadata_path.exists():
        return {}

    latest: dict[str, dict[str, object]] = {}
    for record in iter_jsonl(metadata_path, allow_partial_last_line=True):
        question_id = record.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise ManifestFormatError(
                f"metadata record in {metadata_path} must contain a non-empty string question_id"
            )
        latest[question_id] = record
    return latest


def is_complete(
    record: Mapping[str, object],
    metadata: Mapping[str, object] | None,
    wav_path: Path,
    expected: Mapping[str, object],
    audio_config: AppConfig | AudioConfig,
) -> bool:
    """Return whether a WAV and its latest metadata satisfy every resume invariant."""
    path = Path(wav_path)
    if metadata is None or metadata.get("status") not in _COMPLETE_STATUSES:
        return False

    question_id = record.get("question_id")
    question = record.get("question")
    if (
        not isinstance(question_id, str)
        or not isinstance(question, str)
        or metadata.get("question_id") != question_id
    ):
        return False

    question_sha256 = hashlib.sha256(normalize_question(question).encode("utf-8")).hexdigest()
    if metadata.get("question_sha256") != question_sha256:
        return False
    if "question_sha256" in expected and expected["question_sha256"] != question_sha256:
        return False

    for field in _EXPECTED_TTS_FIELDS:
        if field not in expected or metadata.get(field) != expected[field]:
            return False

    audio = audio_config.audio if isinstance(audio_config, AppConfig) else audio_config
    stored_format = {
        "sample_rate": audio.sample_rate,
        "channels": audio.channels,
        "subtype": audio.subtype,
    }
    if any(metadata.get(field) != value for field, value in stored_format.items()):
        return False

    try:
        if not path.is_file():
            return False
        if metadata.get("audio_sha256") != sha256_file(path):
            return False
        validate_wav(path, audio_config)
    except (AudioValidationError, OSError):
        return False
    return True


def append_jsonl_fsync(path: Path, record: Mapping[str, object]) -> None:
    """Append one compact JSONL record and durably synchronize it."""
    payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    with Path(path).open("a", encoding="utf-8", newline="") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _validate_selection(
    start_index: int,
    end_index: int | None,
    shard_index: int,
    num_shards: int,
    limit: int | None,
) -> None:
    integer_values = (start_index, shard_index, num_shards)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_values):
        raise ValueError("selection indices and shard count must be integers")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if isinstance(end_index, bool) or (end_index is not None and not isinstance(end_index, int)):
        raise ValueError("end_index must be an integer or None")
    if end_index is not None and end_index < start_index:
        raise ValueError("end_index must be at least start_index")
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    if isinstance(limit, bool) or (limit is not None and not isinstance(limit, int)):
        raise ValueError("limit must be an integer or None")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
