"""Prepare revision-pinned TriviaQA manifests and deterministic pilot selections."""

from __future__ import annotations

import json
import os
import random
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from triviaqa_tts.config import AppConfig
from triviaqa_tts.data.manifest import example_to_record, write_canonical_jsonl

_SELECTION_ALGORITHM_VERSION = "python-random-sample-v1"
_EXPECTED_VALIDATION_COUNT = 11_313
_EXPECTED_PILOT_SIZE = 100
_EXPECTED_PILOT_SEED = 20_260_721


@dataclass(frozen=True)
class PreparationResult:
    """Paths and immutable identifiers produced by dataset preparation."""

    dataset_revision: str
    manifest_path: Path
    metadata_path: Path
    pilot_manifest_path: Path
    pilot_ids_path: Path
    manifest_sha256: str
    example_count: int
    pilot_indices: list[int]

    @property
    def resolved_revision(self) -> str:
        """Return the resolved dataset revision under an explicit alias."""
        return self.dataset_revision


def resolve_dataset_revision(
    dataset_name: str,
    revision: str | None = None,
    *,
    hub_api: object | None = None,
) -> str:
    """Resolve a dataset branch, tag, or revision to its immutable Hub SHA."""
    if hub_api is None:
        from huggingface_hub import HfApi

        hub_api = HfApi()

    dataset_info = getattr(hub_api, "dataset_info", None)
    if not callable(dataset_info):
        raise TypeError("hub_api must provide a callable dataset_info method")
    info = dataset_info(repo_id=dataset_name, revision=revision or "main")
    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or not sha.strip():
        raise ValueError(f"Hub did not return a resolved SHA for dataset {dataset_name}")
    return sha


def validate_triviaqa_schema(dataset: object) -> None:
    """Validate the declared subset of the TriviaQA feature contract that is consumed."""
    features = getattr(dataset, "features", None)
    if not isinstance(features, Mapping):
        raise ValueError("dataset features must be a mapping")

    for field_name in ("question", "question_id"):
        if field_name not in features:
            raise ValueError(f"dataset schema is missing {field_name}")
        if not _is_string_feature(features[field_name]):
            raise ValueError(f"dataset schema field {field_name} must be a string")

    answer = features.get("answer")
    if not isinstance(answer, Mapping):
        raise ValueError("dataset schema field answer must be a struct")

    if "value" not in answer:
        raise ValueError("dataset schema is missing answer.value")
    if not _is_string_feature(answer["value"]):
        raise ValueError("dataset schema field answer.value must be a string")

    for field_name in ("aliases", "normalized_aliases"):
        qualified_name = f"answer.{field_name}"
        if field_name not in answer:
            raise ValueError(f"dataset schema is missing {qualified_name}")
        if not _is_string_sequence_feature(answer[field_name]):
            raise ValueError(f"dataset schema field {qualified_name} must be a sequence of strings")


def sample_pilot_indices(count: int, size: int, seed: int) -> list[int]:
    """Select unique pilot indices reproducibly and return them in source order."""
    if count <= 0:
        raise ValueError("count must be positive")
    if size <= 0:
        raise ValueError("size must be positive")
    if size > count:
        raise ValueError("size must not exceed count")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return sorted(random.Random(seed).sample(range(count), size))


def _validate_pilot_selection(config: AppConfig) -> None:
    if config.selection.pilot_size != _EXPECTED_PILOT_SIZE:
        raise ValueError(f"selection.pilot_size must be {_EXPECTED_PILOT_SIZE}")
    if config.selection.pilot_seed != _EXPECTED_PILOT_SEED:
        raise ValueError(f"selection.pilot_seed must be {_EXPECTED_PILOT_SEED}")


def prepare_dataset(
    config: AppConfig,
    *,
    dataset_loader: Callable[..., object] | None = None,
    hub_api: object | None = None,
) -> PreparationResult:
    """Load, validate, and atomically write full and pilot TriviaQA manifests."""
    _validate_pilot_selection(config)
    if dataset_loader is None:
        from datasets import load_dataset

        dataset_loader = load_dataset

    revision = resolve_dataset_revision(
        config.dataset.name,
        config.dataset.revision,
        hub_api=hub_api,
    )
    dataset = dataset_loader(
        config.dataset.name,
        config.dataset.configuration,
        split=config.dataset.split,
        revision=revision,
        cache_dir=str(config.paths.cache_dir),
    )

    validate_triviaqa_schema(dataset)
    rows = _validated_rows(dataset)
    _validate_validation_count(dataset, config.dataset.split, len(rows))
    pilot_indices = sample_pilot_indices(
        len(rows), config.selection.pilot_size, config.selection.pilot_seed
    )
    records = [
        example_to_record(
            row,
            original_index=index,
            dataset_revision=revision,
            dataset=config.dataset.name,
            configuration=config.dataset.configuration,
            split=config.dataset.split,
        )
        for index, row in enumerate(rows)
    ]

    manifest_path = config.selection.manifest_path
    metadata_path = manifest_path.with_suffix(".meta.json")
    pilot_manifest_path = config.paths.data_dir / f"pilot_{config.selection.pilot_size}.jsonl"
    pilot_ids_path = config.paths.data_dir / "pilot_question_ids.txt"
    for parent in {manifest_path.parent, config.paths.data_dir}:
        parent.mkdir(parents=True, exist_ok=True)

    manifest_sha256 = _atomic_write_jsonl(records, manifest_path)
    pilot_records = [records[index] for index in pilot_indices]
    _atomic_write_jsonl(pilot_records, pilot_manifest_path)
    _atomic_write_text(
        "".join(f"{record['question_id']}\n" for record in pilot_records), pilot_ids_path
    )

    question_ids = [str(record["question_id"]) for record in records]
    duplicate_question_id_count = sum(count - 1 for count in Counter(question_ids).values())
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": config.dataset.name,
        "configuration": config.dataset.configuration,
        "split": config.dataset.split,
        "dataset_revision": revision,
        "example_count": len(records),
        "manifest_sha256": manifest_sha256,
        "duplicate_question_id_count": duplicate_question_id_count,
        "empty_question_count": sum(not str(record["question"]).strip() for record in records),
        "empty_answer_count": sum(not str(record["answer_value"]).strip() for record in records),
        "pilot_size": config.selection.pilot_size,
        "pilot_seed": config.selection.pilot_seed,
        "pilot_indices": pilot_indices,
        "selection_algorithm_version": _SELECTION_ALGORITHM_VERSION,
    }
    _atomic_write_json(metadata, metadata_path)

    return PreparationResult(
        dataset_revision=revision,
        manifest_path=manifest_path,
        metadata_path=metadata_path,
        pilot_manifest_path=pilot_manifest_path,
        pilot_ids_path=pilot_ids_path,
        manifest_sha256=manifest_sha256,
        example_count=len(records),
        pilot_indices=pilot_indices,
    )


def _is_string_feature(feature: object) -> bool:
    return getattr(feature, "dtype", None) == "string"


def _is_string_sequence_feature(feature: object) -> bool:
    nested_feature = getattr(feature, "feature", None)
    return nested_feature is not None and _is_string_feature(nested_feature)


def _validated_rows(dataset: object) -> list[Mapping[str, object]]:
    try:
        iterator = iter(dataset)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError("dataset must be iterable") from error

    rows: list[Mapping[str, object]] = []
    for index, row in enumerate(iterator):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {index} must be a mapping")
        _validate_runtime_row(row, index)
        rows.append(row)
    return rows


def _validate_runtime_row(row: Mapping[str, object], index: int) -> None:
    for field_name in ("question", "question_id"):
        if not isinstance(row.get(field_name), str):
            raise ValueError(f"row {index} field {field_name} must be a string")

    answer = row.get("answer")
    if not isinstance(answer, Mapping):
        raise ValueError(f"row {index} field answer must be a mapping")
    if not isinstance(answer.get("value"), str):
        raise ValueError(f"row {index} field answer.value must be a string")
    for field_name in ("aliases", "normalized_aliases"):
        value = answer.get(field_name)
        if not _is_string_sequence(value):
            raise ValueError(f"row {index} field answer.{field_name} must be a list of strings")


def _is_string_sequence(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and all(isinstance(item, str) for item in value)
    )


def _validate_validation_count(dataset: object, split: str, actual_count: int) -> None:
    if actual_count != _EXPECTED_VALIDATION_COUNT:
        raise ValueError(
            f"dataset split {split} contains {actual_count} rows; expected exactly "
            f"{_EXPECTED_VALIDATION_COUNT}"
        )
    info = getattr(dataset, "info", None)
    splits = getattr(info, "splits", None)
    if not isinstance(splits, Mapping):
        raise ValueError(f"dataset split {split} is missing info.splits metadata")
    if split not in splits:
        raise ValueError(f"dataset split {split} is missing from info.splits metadata")
    advertised_count = getattr(splits[split], "num_examples", None)
    if advertised_count != _EXPECTED_VALIDATION_COUNT:
        raise ValueError(
            f"dataset split {split} metadata advertises {advertised_count}; expected exactly "
            f"{_EXPECTED_VALIDATION_COUNT}"
        )


def _atomic_write_jsonl(records: Iterable[Mapping[str, object]], path: Path) -> str:
    return _atomic_replace(path, lambda temporary: write_canonical_jsonl(records, temporary))


def _atomic_write_json(value: Mapping[str, Any], path: Path) -> None:
    def write(temporary: Path) -> None:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

    _atomic_replace(path, write)


def _atomic_write_text(value: str, path: Path) -> None:
    def write(temporary: Path) -> None:
        temporary.write_text(value, encoding="utf-8", newline="\n")

    _atomic_replace(path, write)


def _atomic_replace[T](path: Path, writer: Callable[[Path], T]) -> T:
    path = Path(path)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        result = writer(temporary_path)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return result
