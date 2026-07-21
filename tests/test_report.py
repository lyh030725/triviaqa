from __future__ import annotations

import json
from pathlib import Path

import pytest


def _metadata(index: int, **overrides: object) -> dict[str, object]:
    return {
        "question_id": f"q{index}",
        "question": f"Question {index}?",
        "audio_path": f"/audio/q{index}.wav",
        "status": "success",
        "duration_seconds": float(index + 1),
        "clipped_sample_count": 0,
        "leading_silence_seconds": 0.0,
        "trailing_silence_seconds": 0.0,
        "warning_reasons": [],
        **overrides,
    }


def test_build_pilot_report_aggregates_metrics_and_selects_reproducible_sample() -> None:
    from triviaqa_tts.cli.report_pilot import build_pilot_report

    metadata = [
        _metadata(0),
        _metadata(1, status="warning", clipped_sample_count=2, warning_reasons=["clipping_detected"]),
        _metadata(2, status="warning", warning_reasons=["duration_below_minimum"]),
        _metadata(
            3,
            status="warning",
            warning_reasons=[
                "leading_silence_exceeds_maximum",
                "trailing_silence_exceeds_maximum",
            ],
        ),
    ] + [_metadata(index) for index in range(4, 12)]
    failures = [{"question_id": "bad", "question": "Bad question?", "status": "failure"}]

    report = build_pilot_report(metadata, failures, sample_size=10, seed=20260721)

    assert report["counts"] == {"success": 9, "warning": 3, "failure": 1}
    assert report["durations"] == {
        "total_seconds": 78.0,
        "average_seconds": 6.5,
        "minimum_seconds": 1.0,
        "maximum_seconds": 12.0,
    }
    assert report["quality_counts"] == {
        "clipping": 1,
        "short": 1,
        "leading_silence": 1,
        "trailing_silence": 1,
    }
    assert len(report["listening_sample"]) == 10
    assert report["listening_sample"] == build_pilot_report(
        metadata, failures, sample_size=10, seed=20260721
    )["listening_sample"]
    assert all(set(item) == {"audio_path", "question"} for item in report["listening_sample"])


def test_build_pilot_report_uses_all_audio_items_when_fewer_than_sample_size() -> None:
    from triviaqa_tts.cli.report_pilot import build_pilot_report

    report = build_pilot_report([_metadata(0), _metadata(1)], [], sample_size=10)

    assert report["listening_sample"] == [
        {"audio_path": "/audio/q0.wav", "question": "Question 0?"},
        {"audio_path": "/audio/q1.wav", "question": "Question 1?"},
    ]


def test_report_main_writes_json_and_csv_and_ignores_partial_jsonl_tail(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from triviaqa_tts.cli.report_pilot import main

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "metadata.jsonl").write_text(
        json.dumps(_metadata(0), separators=(",", ":")) + "\n" + '{"question_id":"partial',
        encoding="utf-8",
    )
    (logs / "failures.jsonl").write_text(
        json.dumps({"question_id": "bad", "status": "failure"}) + "\n", encoding="utf-8"
    )
    config = tmp_path / "config.yaml"
    config.write_text(f"paths:\n  logs_dir: {logs}\n", encoding="utf-8")

    with pytest.warns(UserWarning, match="incomplete final JSONL record"):
        assert main(["--config", str(config)]) == 0

    report_path = logs / "pilot_audio_report.json"
    csv_path = logs / "pilot_audio_report.csv"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["counts"] == {"failure": 1, "success": 1, "warning": 0}
    assert csv_path.read_text(encoding="utf-8").splitlines() == [
        "question_id,status,audio_path,question,duration_seconds,clipped_sample_count,leading_silence_seconds,trailing_silence_seconds,warning_reasons",
        "q0,success,/audio/q0.wav,Question 0?,1.0,0,0.0,0.0,",
    ]
    assert "/audio/q0.wav\tQuestion 0?" in capsys.readouterr().out
