"""Deterministic conversion, serialization, and reading of manifest records."""

from __future__ import annotations

import hashlib
import html
import json
import re
import warnings
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

_DEFAULT_DATASET = "mandarjoshi/trivia_qa"
_DEFAULT_CONFIGURATION = "unfiltered.nocontext"
_DEFAULT_SPLIT = "validation"
_SAFE_STEM_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


class ManifestFormatError(ValueError):
    """Raised when a JSONL manifest contains a malformed complete record."""


def example_to_record(
    row: Mapping[str, object],
    *,
    original_index: int,
    dataset_revision: str,
    dataset: str = _DEFAULT_DATASET,
    configuration: str = _DEFAULT_CONFIGURATION,
    split: str = _DEFAULT_SPLIT,
) -> dict[str, object]:
    """Convert a TriviaQA-like row to the canonical manifest record shape."""
    answer = row["answer"]
    if not isinstance(answer, Mapping):
        raise ManifestFormatError("answer must be a mapping")

    return {
        "dataset": dataset,
        "configuration": configuration,
        "split": split,
        "dataset_revision": dataset_revision,
        "original_index": original_index,
        "question_id": row["question_id"],
        "question": row["question"],
        "answer_value": answer["value"],
        "aliases": answer["aliases"],
        "normalized_aliases": answer["normalized_aliases"],
    }


def write_canonical_jsonl(records: Iterable[Mapping[str, object]], path: Path) -> str:
    """Write compact, UTF-8 JSON Lines and return its SHA-256 digest."""
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            )
            stream.write("\n")
    return sha256_file(path)


def iter_jsonl(path: Path, allow_partial_last_line: bool = True) -> Iterator[dict[str, object]]:
    """Yield JSON objects, tolerating only an invalid unterminated final record."""
    raw_lines = Path(path).read_bytes().splitlines(keepends=True)
    for index, raw_line in enumerate(raw_lines, start=1):
        is_final = index == len(raw_lines)
        has_newline = raw_line.endswith(b"\n")
        try:
            value = json.loads(raw_line.decode("utf-8"))
        except UnicodeDecodeError as error:
            raise ManifestFormatError(f"invalid JSONL record at line {index} in {path}") from error
        except json.JSONDecodeError as error:
            if (
                allow_partial_last_line
                and is_final
                and not has_newline
                and _is_truncated_json_object(error)
            ):
                warnings.warn(
                    f"ignoring incomplete final JSONL record in {path}", UserWarning, stacklevel=2
                )
                return
            raise ManifestFormatError(f"invalid JSONL record at line {index} in {path}") from error
        if not isinstance(value, dict):
            raise ManifestFormatError(f"JSONL record at line {index} in {path} must be an object")
        yield value


def _is_truncated_json_object(error: json.JSONDecodeError) -> bool:
    """Return whether the decoder error can only result from an unfinished object."""
    document = error.doc
    if not document.lstrip().startswith("{"):
        return False
    return error.msg == "Unterminated string starting at" or error.pos == len(document)


def normalize_question(text: str) -> str:
    """Apply the minimal normalization used before speech synthesis."""
    normalized = " ".join(html.unescape(text).split())
    if normalized and normalized[-1] not in ".!?;:":
        return f"{normalized}?"
    return normalized


def safe_audio_stem(question_id: str) -> str:
    """Return a readable filename stem with a collision-resistant ID suffix."""
    prefix = _SAFE_STEM_COMPONENT.sub("_", question_id).strip("._-") or "question"
    return f"{prefix}_{hashlib.sha256(question_id.encode('utf-8')).hexdigest()[:12]}"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
