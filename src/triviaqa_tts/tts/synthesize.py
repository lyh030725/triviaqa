"""Deterministic record selection and strict synthesis resume primitives."""

from __future__ import annotations

import errno
import hashlib
import importlib
import json
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from triviaqa_tts.audio.validation import validate_wav, write_validated_wav
from triviaqa_tts.config import AppConfig, AudioConfig, TTSConfig
from triviaqa_tts.data.manifest import (
    ManifestFormatError,
    iter_jsonl,
    normalize_question,
    safe_audio_stem,
    sha256_file,
)
from triviaqa_tts.reproducibility import collect_runtime_environment, set_seed
from triviaqa_tts.tts.base import TTSBackend

_COMPLETE_STATUSES = frozenset({"success", "warning"})
_EXPECTED_TTS_FIELDS = (
    "model_id",
    "model_revision",
    "speaker",
    "language",
    "seed",
    "generation",
)
_MANIFEST_STRING_FIELDS = (
    "dataset",
    "configuration",
    "split",
    "dataset_revision",
    "question_id",
    "question",
    "answer_value",
)
_MANIFEST_LIST_FIELDS = ("aliases", "normalized_aliases")
_FATAL_ERRNOS = frozenset({errno.EIO, errno.ENOSPC})


@dataclass(frozen=True)
class SynthesisSummary:
    """Counts describing one synthesis invocation."""

    selected: int
    processed: int
    skipped: int
    success: int
    warning: int
    failure: int


BackendFactory = Callable[[TTSConfig], TTSBackend]


def synthesize_manifest(
    config: AppConfig,
    backend_factory: BackendFactory,
) -> SynthesisSummary:
    """Synthesize selected manifest records as independent durable transactions."""
    if not isinstance(config, AppConfig):
        raise TypeError("config must be an AppConfig")
    if not callable(backend_factory):
        raise TypeError("backend_factory must be callable")

    records = select_records(
        iter_jsonl(config.selection.manifest_path, allow_partial_last_line=False),
        config.selection.start_index,
        config.selection.end_index,
        config.selection.shard_index,
        config.selection.num_shards,
        config.selection.limit,
    )
    for record in records:
        _validate_manifest_record(record)
    _validate_json_value(config.tts.generation, "tts.generation")

    selected_count = len(records)
    if config.selection.dry_run:
        return SynthesisSummary(selected_count, 0, selected_count, 0, 0, 0)
    if not records:
        return SynthesisSummary(0, 0, 0, 0, 0, 0)

    metadata_path = config.paths.logs_dir / "metadata.jsonl"
    failures_path = config.paths.logs_dir / "failures.jsonl"
    latest = load_latest_metadata(metadata_path) if config.selection.resume else {}

    # Construction is outside the sample loop so model resolution/load failures
    # invalidate the run rather than being misreported as one bad question.
    backend = backend_factory(config.tts)
    expected = _expected_tts_metadata(config, backend)
    environment = collect_runtime_environment()
    _validate_json_value(environment, "runtime environment")

    config.paths.audio_dir.mkdir(parents=True, exist_ok=True)
    config.paths.logs_dir.mkdir(parents=True, exist_ok=True)

    processed = skipped = success = warning = failure = 0
    for record in records:
        wav_path = _audio_path(config.paths.audio_dir, record["question_id"])
        if config.selection.resume and is_complete(
            record,
            latest.get(record["question_id"]),
            wav_path,
            expected,
            config,
        ):
            skipped += 1
            continue

        tts_text = normalize_question(record["question"])
        question_sha256 = hashlib.sha256(tts_text.encode("utf-8")).hexdigest()
        base_metadata = _base_metadata(
            record,
            tts_text=tts_text,
            question_sha256=question_sha256,
            expected=expected,
            environment=environment,
        )
        oom_retried = False
        try:
            for attempt in range(2):
                set_seed(config.tts.seed)
                try:
                    waveform, native_sample_rate = backend.synthesize(tts_text, config.tts.seed)
                    break
                except Exception as error:
                    if attempt == 0 and _is_cuda_oom(error):
                        oom_retried = True
                        _clear_cuda_cache()
                        continue
                    raise

            waveform_array = np.asarray(waveform)
            metrics = write_validated_wav(
                waveform_array,
                native_sample_rate,
                wav_path,
                config,
            )
            audio_sha256 = sha256_file(wav_path)
            metadata = {
                **base_metadata,
                "audio_path": str(wav_path),
                "audio_sha256": audio_sha256,
                "native_sample_rate": native_sample_rate,
                "waveform_shape": list(waveform_array.shape),
                **asdict(metrics),
            }
            metadata["warning_reasons"] = list(metrics.warning_reasons)
        except Exception as error:
            if _contains_fatal_os_error(error):
                raise
            failure_record = {
                **base_metadata,
                "audio_path": str(wav_path),
                "status": "failure",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "cuda_oom": _is_cuda_oom(error),
                "cuda_oom_retried": oom_retried,
            }
            # A failure-log append error is fatal because continuing would silently
            # discard the audit trail for an attempted sample.
            append_jsonl_fsync(failures_path, failure_record)
            processed += 1
            failure += 1
            continue

        # The metadata append is the durable commit record; append errors are fatal.
        append_jsonl_fsync(metadata_path, metadata)
        processed += 1
        if metrics.status == "warning":
            warning += 1
        else:
            success += 1

    return SynthesisSummary(selected_count, processed, skipped, success, warning, failure)


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
    if "question_sha256" not in metadata or not _json_values_equal(
        metadata["question_sha256"], question_sha256
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


def _validate_manifest_record(record: Mapping[str, object]) -> None:
    if not isinstance(record, Mapping):
        raise ManifestFormatError("manifest record must be an object")
    for field in _MANIFEST_STRING_FIELDS:
        value = record.get(field)
        if not isinstance(value, str) or (field == "question_id" and not value):
            raise ManifestFormatError(f"manifest field {field} must be a valid string")
    original_index = record.get("original_index")
    if isinstance(original_index, bool) or not isinstance(original_index, int):
        raise ManifestFormatError("manifest field original_index must be an integer")
    for field in _MANIFEST_LIST_FIELDS:
        value = record.get(field)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ManifestFormatError(f"manifest field {field} must be a list of strings")


def _validate_json_value(value: object, location: str) -> None:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{location} must contain only finite JSON values") from error


def _expected_tts_metadata(config: AppConfig, backend: TTSBackend) -> dict[str, object]:
    model_id = getattr(backend, "model_id", None)
    if not isinstance(model_id, str) or not model_id:
        raise TypeError("backend.model_id must be a non-empty string")
    revision = getattr(backend, "model_revision", config.tts.model_revision)
    if revision is not None and not isinstance(revision, str):
        raise TypeError("backend model revision must be a string or None")
    attention = getattr(backend, "attention_implementation", config.tts.attention)
    dtype = getattr(backend, "dtype", config.tts.dtype)
    return {
        "model_id": model_id,
        "model_revision": revision,
        "speaker": config.tts.speaker,
        "language": config.tts.language,
        "seed": config.tts.seed,
        "dtype": dtype if isinstance(dtype, (str, int, float, bool)) else str(dtype),
        "attention_implementation": (
            attention
            if isinstance(attention, (str, int, float, bool)) or attention is None
            else str(attention)
        ),
        "generation": config.tts.generation,
    }


def _base_metadata(
    record: Mapping[str, object],
    *,
    tts_text: str,
    question_sha256: str,
    expected: Mapping[str, object],
    environment: Mapping[str, object],
) -> dict[str, object]:
    return {
        "dataset": record["dataset"],
        "configuration": record["configuration"],
        "split": record["split"],
        "dataset_revision": record["dataset_revision"],
        "original_index": record["original_index"],
        "question_id": record["question_id"],
        "question": record["question"],
        "tts_text": tts_text,
        "question_sha256": question_sha256,
        "answer_value": record["answer_value"],
        "aliases": record["aliases"],
        "normalized_aliases": record["normalized_aliases"],
        **expected,
        "environment": environment,
    }


def _audio_path(audio_dir: Path, question_id: object) -> Path:
    if not isinstance(question_id, str) or not question_id:
        raise ManifestFormatError("question_id must be a non-empty string")
    directory = Path(audio_dir).resolve()
    path = (directory / f"{safe_audio_stem(question_id)}.wav").resolve()
    if path.parent != directory:
        raise ManifestFormatError("resolved audio path escapes configured audio directory")
    return path


def _exception_chain(error: BaseException) -> Iterable[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _contains_fatal_os_error(error: BaseException) -> bool:
    return any(
        isinstance(item, OSError) and item.errno in _FATAL_ERRNOS
        for item in _exception_chain(error)
    )


def _is_cuda_oom(error: BaseException) -> bool:
    for item in _exception_chain(error):
        message = str(item).casefold()
        type_name = type(item).__name__.casefold()
        module_name = type(item).__module__.casefold()
        if "cuda out of memory" in message or "cuda error: out of memory" in message:
            return True
        if "outofmemoryerror" in type_name and (
            "torch" in module_name or "cuda" in message or "cuda" in module_name
        ):
            return True
    return False


def _clear_cuda_cache() -> None:
    try:
        torch = importlib.import_module("torch")
        empty_cache = getattr(getattr(torch, "cuda", None), "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
    except (ImportError, OSError, RuntimeError):
        return


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
