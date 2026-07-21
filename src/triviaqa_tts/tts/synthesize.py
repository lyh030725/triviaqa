"""Deterministic record selection and strict synthesis resume primitives."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

from triviaqa_tts.audio.validation import validate_wav
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
    try:
        return _is_complete(record, metadata, wav_path, expected, audio_config)
    except Exception:
        return False


def _is_complete(
    record: Mapping[str, object],
    metadata: Mapping[str, object] | None,
    wav_path: Path,
    expected: Mapping[str, object],
    audio_config: AppConfig | AudioConfig,
) -> bool:
    """Compare resume metadata and its WAV, failing closed for malformed values."""
    path = Path(wav_path)
    if (
        not isinstance(record, Mapping)
        or not isinstance(metadata, Mapping)
        or not isinstance(expected, Mapping)
    ):
        return False

    status = metadata.get("status")
    if not isinstance(status, str) or status not in _COMPLETE_STATUSES:
        return False

    question_id = record.get("question_id")
    question = record.get("question")
    if (
        not isinstance(question_id, str)
        or not isinstance(question, str)
        or "question_id" not in metadata
        or not _json_values_equal(metadata["question_id"], question_id)
    ):
        return False

    question_sha256 = hashlib.sha256(normalize_question(question).encode("utf-8")).hexdigest()
    if (
        "question_sha256" not in metadata
        or not _json_values_equal(metadata["question_sha256"], question_sha256)
    ):
        return False
    if "question_sha256" in expected and not _json_values_equal(
        expected["question_sha256"], question_sha256
    ):
        return False

    for field in _EXPECTED_TTS_FIELDS:
        if (
            field not in expected
            or field not in metadata
            or not _json_values_equal(metadata[field], expected[field])
        ):
            return False

    audio = audio_config.audio if isinstance(audio_config, AppConfig) else audio_config
    stored_format = {
        "sample_rate": audio.sample_rate,
        "channels": audio.channels,
        "subtype": audio.subtype,
    }
    if any(
        field not in metadata or not _json_values_equal(metadata[field], value)
        for field, value in stored_format.items()
    ):
        return False

    if not path.is_file() or "audio_sha256" not in metadata:
        return False
    if not _json_values_equal(metadata["audio_sha256"], sha256_file(path)):
        return False
    validate_wav(path, audio_config)
    return True


def _json_values_equal(actual: object, expected: object) -> bool:
    """Compare JSON values without Python's cross-type equality coercions."""
    if type(actual) is not type(expected):
        return False
    if actual is None:
        return True
    if isinstance(actual, (bool, int, float, str)):
        return actual == expected
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(
            _json_values_equal(item, expected_item)
            for item, expected_item in zip(actual, expected, strict=True)
        )
    if isinstance(actual, dict):
        if not all(isinstance(key, str) for key in actual):
            return False
        if not all(isinstance(key, str) for key in expected):
            return False
        return actual.keys() == expected.keys() and all(
            _json_values_equal(actual[key], expected[key]) for key in actual
        )
    return False


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
