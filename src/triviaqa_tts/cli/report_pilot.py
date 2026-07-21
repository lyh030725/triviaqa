"""Create durable pilot audio quality reports from synthesis JSONL logs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from triviaqa_tts.config import load_config
from triviaqa_tts.data.manifest import iter_jsonl

_REPORT_SEED = 20_260_721
_REPORT_NAME = "pilot_audio_report"
_CSV_COLUMNS = (
    "question_id",
    "status",
    "audio_path",
    "question",
    "duration_seconds",
    "clipped_sample_count",
    "leading_silence_seconds",
    "trailing_silence_seconds",
    "warning_reasons",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the pilot-report command parser."""
    parser = argparse.ArgumentParser(description="Report pilot synthesis audio quality.")
    parser.add_argument("--config", type=Path, required=True, help="path to the YAML config file")
    return parser


def build_pilot_report(
    metadata: Iterable[Mapping[str, object]],
    failures: Iterable[Mapping[str, object]],
    sample_size: int = 10,
    seed: int = _REPORT_SEED,
) -> dict[str, object]:
    """Aggregate synthesis logs and select a deterministic listening sample."""
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size <= 0:
        raise ValueError("sample_size must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    audio_items = [_audio_item(item) for item in metadata]
    failure_count = sum(1 for item in failures if _failure_item(item))
    success_count = sum(item["status"] == "success" for item in audio_items)
    warning_count = sum(item["status"] == "warning" for item in audio_items)
    durations = [item["duration_seconds"] for item in audio_items]

    listening_items = [
        {"audio_path": item["audio_path"], "question": item["question"]} for item in audio_items
    ]
    if len(listening_items) > sample_size:
        indices = sorted(random.Random(seed).sample(range(len(listening_items)), sample_size))
        listening_items = [listening_items[index] for index in indices]

    return {
        "counts": {
            "success": success_count,
            "warning": warning_count,
            "failure": failure_count,
        },
        "durations": {
            "total_seconds": sum(durations),
            "average_seconds": sum(durations) / len(durations) if durations else 0.0,
            "minimum_seconds": min(durations) if durations else 0.0,
            "maximum_seconds": max(durations) if durations else 0.0,
        },
        "quality_counts": {
            "clipping": sum(item["clipped_sample_count"] > 0 for item in audio_items),
            "short": sum("duration_below_minimum" in item["warning_reasons"] for item in audio_items),
            "leading_silence": sum(
                "leading_silence_exceeds_maximum" in item["warning_reasons"]
                for item in audio_items
            ),
            "trailing_silence": sum(
                "trailing_silence_exceeds_maximum" in item["warning_reasons"]
                for item in audio_items
            ),
        },
        "listening_sample": listening_items,
    }


def _audio_item(item: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(item, Mapping):
        raise ValueError("metadata record must be an object")
    status = _required_string(item, "status")
    if status not in {"success", "warning"}:
        raise ValueError("metadata status must be success or warning")
    warning_reasons = item.get("warning_reasons", [])
    if not isinstance(warning_reasons, list) or not all(
        isinstance(reason, str) for reason in warning_reasons
    ):
        raise ValueError("metadata warning_reasons must be a list of strings")
    return {
        "question_id": _required_string(item, "question_id"),
        "status": status,
        "audio_path": _required_string(item, "audio_path"),
        "question": _required_string(item, "question"),
        "duration_seconds": _required_number(item, "duration_seconds"),
        "clipped_sample_count": _required_nonnegative_integer(item, "clipped_sample_count"),
        "leading_silence_seconds": _required_nonnegative_number(item, "leading_silence_seconds"),
        "trailing_silence_seconds": _required_nonnegative_number(item, "trailing_silence_seconds"),
        "warning_reasons": warning_reasons,
    }


def _failure_item(item: Mapping[str, object]) -> bool:
    if not isinstance(item, Mapping) or item.get("status") != "failure":
        raise ValueError("failure log record must have status failure")
    return True


def _required_string(item: Mapping[str, object], name: str) -> str:
    value = item.get(name)
    if not isinstance(value, str):
        raise ValueError(f"metadata {name} must be a string")
    return value


def _required_number(item: Mapping[str, object], name: str) -> float:
    value = item.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metadata {name} must be a number")
    return float(value)


def _required_nonnegative_number(item: Mapping[str, object], name: str) -> float:
    value = _required_number(item, name)
    if value < 0:
        raise ValueError(f"metadata {name} must be non-negative")
    return value


def _required_nonnegative_integer(item: Mapping[str, object], name: str) -> int:
    value = item.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"metadata {name} must be a non-negative integer")
    return value


def _load_optional_jsonl(path: Path) -> list[dict[str, object]]:
    return list(iter_jsonl(path, allow_partial_last_line=True)) if path.exists() else []


def _write_json_atomically(path: Path, report: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(report, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_csv(path: Path, metadata: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for record in metadata:
            item = _audio_item(record)
            writer.writerow(
                {
                    **item,
                    "warning_reasons": ";".join(item["warning_reasons"]),
                }
            )


def main(argv: Sequence[str] | None = None) -> int:
    """Write pilot quality reports and print the deterministic listening sample."""
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        logs_dir = config.paths.logs_dir
        metadata = _load_optional_jsonl(logs_dir / "metadata.jsonl")
        failures = _load_optional_jsonl(logs_dir / "failures.jsonl")
        report = build_pilot_report(metadata, failures)
        _write_json_atomically(logs_dir / f"{_REPORT_NAME}.json", report)
        _write_csv(logs_dir / f"{_REPORT_NAME}.csv", metadata)
    except Exception as error:
        print(f"fatal: {error}")
        return 1

    for item in report["listening_sample"]:
        print(f"{item['audio_path']}\t{item['question']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
