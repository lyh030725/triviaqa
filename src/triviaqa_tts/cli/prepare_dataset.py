"""CLI for preparing revision-pinned TriviaQA manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Sequence

from triviaqa_tts.config import load_config
from triviaqa_tts.data.prepare import prepare_dataset


def build_parser() -> argparse.ArgumentParser:
    """Build the dataset preparation argument parser."""
    parser = argparse.ArgumentParser(
        description="Prepare full and deterministic pilot TriviaQA manifests."
    )
    parser.add_argument("--config", type=Path, required=True, help="path to the YAML config file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare dataset artifacts from a YAML configuration."""
    args = build_parser().parse_args(argv)
    result = prepare_dataset(load_config(args.config))
    print(
        json.dumps(
            {
                "dataset_revision": result.dataset_revision,
                "example_count": result.example_count,
                "manifest_path": str(result.manifest_path),
                "manifest_sha256": result.manifest_sha256,
                "pilot_manifest_path": str(result.pilot_manifest_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
