#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
cd -- "${REPO_ROOT}"

exec "${REPO_ROOT}/.venv/bin/python" -m triviaqa_tts.cli.synthesize_questions \
  --config "${REPO_ROOT}/configs/validation.yaml" "$@"
