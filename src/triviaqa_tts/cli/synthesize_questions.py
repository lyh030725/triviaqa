"""Command-line entry point for reproducible TriviaQA question synthesis."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from triviaqa_tts.config import apply_cli_overrides, load_config
from triviaqa_tts.tts.qwen3_backend import Qwen3Backend
from triviaqa_tts.tts.synthesize import SynthesisSummary, synthesize_manifest


def build_parser() -> argparse.ArgumentParser:
    """Build the synthesis command parser."""
    parser = argparse.ArgumentParser(description="Synthesize selected TriviaQA questions.")
    parser.add_argument("--config", type=Path, required=True, help="path to the YAML config file")
    parser.add_argument("--limit", type=int, help="maximum number of selected questions")
    parser.add_argument("--start-index", type=int, help="inclusive manifest index")
    parser.add_argument("--end-index", type=int, help="exclusive manifest index")
    parser.add_argument("--shard-index", type=int, help="zero-based shard index")
    parser.add_argument("--num-shards", type=int, help="number of deterministic shards")
    parser.add_argument(
        "--resume", action="store_true", help="skip complete matching outputs"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="regenerate selected outputs"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="select records without loading a model"
    )
    return parser


def qwen3_backend_factory(config):
    """Construct the approved backend only when synthesis needs it."""
    return Qwen3Backend.from_config(config)


def _selection_overrides(args: argparse.Namespace) -> dict[str, object]:
    return {
        "limit": args.limit,
        "start_index": args.start_index,
        "end_index": args.end_index,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "resume": True if args.resume else None,
        "overwrite": True if args.overwrite else None,
        "dry_run": True if args.dry_run else None,
    }


def _format_summary(summary: SynthesisSummary) -> str:
    return " ".join(
        f"{name}={getattr(summary, name)}"
        for name in ("selected", "processed", "skipped", "success", "warning", "failure")
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run synthesis and return a process-compatible status code."""
    args = build_parser().parse_args(argv)
    try:
        config = apply_cli_overrides(load_config(args.config), **_selection_overrides(args))
        summary = synthesize_manifest(config, qwen3_backend_factory)
    except Exception as error:
        print(f"fatal: {error}")
        return 1

    print(_format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
