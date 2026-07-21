#!/usr/bin/env bash
set -Eeuo pipefail

trap 'printf "bootstrap failed at line %s\n" "${LINENO}" >&2' ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
CACHE_ROOT="/workspace/huggingface_cache"

if [[ "$(uname -s)" != "Linux" ]]; then
  printf 'RunPod bootstrap requires Linux.\n' >&2
  exit 1
fi
if [[ ! -d /workspace || ! -w /workspace ]]; then
  printf 'RunPod bootstrap requires a writable /workspace directory.\n' >&2
  exit 1
fi
if ! command -v apt-get >/dev/null 2>&1; then
  printf 'RunPod bootstrap requires an apt-based image.\n' >&2
  exit 1
fi

cd -- "${REPO_ROOT}"

if (( EUID == 0 )); then
  APT=(apt-get)
elif command -v sudo >/dev/null 2>&1; then
  APT=(sudo apt-get)
else
  printf 'System package installation requires root or sudo.\n' >&2
  exit 1
fi

"${APT[@]}" update
DEBIAN_FRONTEND=noninteractive "${APT[@]}" install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  curl \
  ffmpeg \
  git \
  libsndfile1 \
  ninja-build \
  sox \
  pkg-config

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${UV_INSTALL_DIR:-${XDG_BIN_HOME:-${HOME}/.local/bin}}:${PATH}"
fi
if ! command -v uv >/dev/null 2>&1; then
  printf 'uv installation did not place uv on PATH.\n' >&2
  exit 1
fi

uv python install 3.12
uv venv --python 3.12 --allow-existing "${REPO_ROOT}/.venv"
uv sync --locked --extra tts

mkdir -p \
  "${CACHE_ROOT}/hub" \
  "${CACHE_ROOT}/datasets"
printf '%s\n' \
  "export HF_HOME=${CACHE_ROOT}" \
  "export HF_HUB_CACHE=${CACHE_ROOT}/hub" \
  "export HUGGINGFACE_HUB_CACHE=${CACHE_ROOT}/hub" \
  "export HF_DATASETS_CACHE=${CACHE_ROOT}/datasets" \
  "unset TRANSFORMERS_CACHE" \
  > "${REPO_ROOT}/.env.runpod"

export HF_HOME="${CACHE_ROOT}"
export HF_HUB_CACHE="${CACHE_ROOT}/hub"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hub"
export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
unset TRANSFORMERS_CACHE

"${PYTHON}" - <<'PY'
from __future__ import annotations

import importlib.metadata
import platform

EXPECTED_TORCH_VERSION = "2.11.0"
EXPECTED_TORCHAUDIO_VERSION = "2.11.0"
EXPECTED_CUDA_VERSION_PREFIX = "12.8"

try:
    import torch
    import torchaudio
except (ImportError, OSError) as error:
    raise SystemExit(f"PyTorch/TorchAudio import failed: {error}") from error


if platform.python_version_tuple()[:2] != ("3", "12"):
    raise SystemExit(f"Python 3.12 is required, found {platform.python_version()}")
torch_version = torch.__version__.split("+", 1)[0]
if torch_version != EXPECTED_TORCH_VERSION:
    raise SystemExit(
        f"PyTorch 2.11.0 is required, found {torch.__version__}; rerun locked bootstrap"
    )
torchaudio_version = torchaudio.__version__.split("+", 1)[0]
if torchaudio_version != EXPECTED_TORCHAUDIO_VERSION:
    raise SystemExit(
        f"TorchAudio 2.11.0 is required, found {torchaudio.__version__}; rerun locked bootstrap"
    )
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is unavailable in PyTorch; select a CUDA-enabled RunPod GPU image with a "
        "CUDA 12.8-compatible NVIDIA driver"
    )
cuda_version = torch.version.cuda
if not cuda_version or not cuda_version.startswith(EXPECTED_CUDA_VERSION_PREFIX):
    raise SystemExit(
        f"PyTorch CUDA 12.8 runtime is required, found torch.version.cuda={cuda_version!r}; "
        "verify the cu128 uv index and regenerate the environment"
    )
if not torch.cuda.is_bf16_supported():
    raise SystemExit(
        f"GPU {torch.cuda.get_device_name(0)!r} does not support the configured bfloat16 dtype"
    )

print(f"Python: {platform.python_version()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")
print(f"TorchAudio: {torchaudio.__version__}")
print(f"CUDA runtime: {cuda_version}")
for distribution in ("qwen-tts", "transformers", "soundfile"):
    print(f"{distribution}: {importlib.metadata.version(distribution)}")
PY

df -h /workspace

if [[ "${INSTALL_FLASH_ATTN:-1}" == "0" ]]; then
  printf 'Skipping optional FlashAttention installation; Qwen will use SDPA fallback.\n'
elif "${PYTHON}" -c 'import flash_attn' >/dev/null 2>&1; then
  printf 'FlashAttention is already installed; automatic attention selection is enabled.\n'
elif MAX_JOBS=4 uv pip install --python "${PYTHON}" --no-build-isolation flash-attn; then
  if "${PYTHON}" -c 'import flash_attn; print(f"FlashAttention: {flash_attn.__version__}")'; then
    printf 'FlashAttention installed; automatic attention selection is enabled.\n'
  else
    printf 'FlashAttention import failed; Qwen will use SDPA fallback.\n' >&2
  fi
else
  printf 'Optional flash-attn installation failed; Qwen will use SDPA fallback.\n' >&2
fi

printf 'Bootstrap complete. Load cache variables with: source %s/.env.runpod\n' "${REPO_ROOT}"
