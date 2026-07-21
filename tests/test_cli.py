from __future__ import annotations

from pathlib import Path

import pytest


def test_synthesis_parser_accepts_all_selection_overrides(tmp_path: Path) -> None:
    from triviaqa_tts.cli.synthesize_questions import build_parser

    config = tmp_path / "config.yaml"
    args = build_parser().parse_args(
        [
            "--config",
            str(config),
            "--limit",
            "5",
            "--start-index",
            "2",
            "--end-index",
            "9",
            "--shard-index",
            "1",
            "--num-shards",
            "3",
            "--resume",
            "--dry-run",
        ]
    )

    assert args.config == config
    assert args.limit == 5
    assert args.start_index == 2
    assert args.end_index == 9
    assert args.shard_index == 1
    assert args.num_shards == 3
    assert args.resume is True
    assert args.overwrite is False
    assert args.dry_run is True


@pytest.mark.parametrize(
    "argv",
    [
        ["--config", "settings.yaml", "--shard-index", "1", "--num-shards", "1"],
        ["--config", "settings.yaml", "--resume", "--overwrite"],
    ],
)
def test_synthesis_main_rejects_invalid_selection_overrides(argv: list[str]) -> None:
    from triviaqa_tts.cli.synthesize_questions import main

    assert main(argv) == 1


def test_synthesis_dry_run_merges_config_and_does_not_construct_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from triviaqa_tts.cli import synthesize_questions

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"dataset":"mandarjoshi/trivia_qa","configuration":"unfiltered.nocontext",'
        '"split":"validation","dataset_revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        '"original_index":0,'
        '"question_id":"q1","question":"Question?","answer_value":"answer",'
        '"aliases":[],"normalized_aliases":[]}\n',
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        "selection:\n"
        f"  manifest_path: {manifest}\n"
        "paths:\n"
        f"  logs_dir: {tmp_path / 'logs'}\n"
        f"  audio_dir: {tmp_path / 'audio'}\n",
        encoding="utf-8",
    )

    def forbidden_factory(*args: object, **kwargs: object) -> object:
        raise AssertionError("backend construction must remain lazy during dry run")

    monkeypatch.setattr(synthesize_questions, "qwen3_backend_factory", forbidden_factory)

    assert synthesize_questions.main(
        ["--config", str(config), "--limit", "1", "--dry-run"]
    ) == 0
    output = capsys.readouterr().out
    assert "selected=1" in output
    assert "processed=0" in output
    assert "skipped=1" in output

