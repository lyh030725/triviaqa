#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
if [[ -f "${REPO_ROOT}/.env.runpod" ]]; then
  source "${REPO_ROOT}/.env.runpod"
fi
unset TRANSFORMERS_CACHE
cd -- "${REPO_ROOT}"

"${REPO_ROOT}/.venv/bin/python" -m triviaqa_tts.cli.synthesize_questions \
  --config "${REPO_ROOT}/configs/pilot.yaml" "$@"
"${REPO_ROOT}/.venv/bin/python" -m triviaqa_tts.cli.report_pilot \
  --config "${REPO_ROOT}/configs/pilot.yaml"
