from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIRECTORY = REPOSITORY_ROOT / "configs"
SCRIPT_DIRECTORY = REPOSITORY_ROOT / "scripts"


def _load_yaml(name: str) -> dict[str, object]:
    loaded = yaml.safe_load((CONFIG_DIRECTORY / name).read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _target_dependencies(makefile: str, target: str) -> list[str]:
    match = re.search(rf"^{re.escape(target)}\s*:(?P<dependencies>[^\n]*)$", makefile, re.MULTILINE)
    assert match is not None, f"missing make target: {target}"
    return match.group("dependencies").split()


def test_configs_share_the_approved_generation_contract() -> None:
    pilot = _load_yaml("pilot.yaml")
    validation = _load_yaml("validation.yaml")

    expected_tts = {
        "backend": "qwen3",
        "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "model_revision": None,
        "language": "English",
        "speaker": "Aiden",
        "seed": 42,
        "dtype": "bfloat16",
        "attention": "auto",
        "generation": {},
    }
    expected_audio = {"sample_rate": 24_000, "channels": 1, "subtype": "PCM_16"}
    expected_validation = {
        "minimum_duration_seconds": 0.3,
        "maximum_duration_seconds": 60.0,
        "clipping_threshold": 0.999,
        "silence_threshold": 0.01,
        "maximum_zero_ratio": 0.99,
        "maximum_leading_silence_seconds": 2.0,
        "maximum_trailing_silence_seconds": 3.0,
    }

    for config in (pilot, validation):
        assert config["dataset"] == {
            "name": "mandarjoshi/trivia_qa",
            "configuration": "unfiltered.nocontext",
            "split": "validation",
            "revision": None,
        }
        assert config["tts"] == expected_tts
        assert config["audio"] == expected_audio
        assert config["audio_validation"] == expected_validation
        assert config["paths"]["cache_dir"] == "/workspace/huggingface_cache"
        assert config["selection"]["pilot_size"] == 100
        assert config["selection"]["pilot_seed"] == 20_260_721


def test_configs_separate_pilot_and_validation_inputs_and_outputs() -> None:
    pilot = _load_yaml("pilot.yaml")
    validation = _load_yaml("validation.yaml")

    assert pilot["selection"]["manifest_path"] == "data/pilot_100.jsonl"
    assert pilot["selection"]["limit"] == 100
    assert validation["selection"]["manifest_path"] == "data/validation_manifest.jsonl"
    assert validation["selection"]["limit"] is None

    assert pilot["paths"]["audio_dir"] == "audio/qwen3_aiden_seed42/pilot"
    assert validation["paths"]["audio_dir"] == "audio/qwen3_aiden_seed42/validation"
    assert pilot["paths"]["logs_dir"] == "logs/pilot"
    assert validation["paths"]["logs_dir"] == "logs/validation"


def test_shell_scripts_are_executable_fail_fast_wrappers() -> None:
    expected_modules = {
        "prepare_dataset.sh": "triviaqa_tts.cli.prepare_dataset",
        "synthesize_pilot.sh": "triviaqa_tts.cli.synthesize_questions",
        "synthesize_validation.sh": "triviaqa_tts.cli.synthesize_questions",
    }

    for path in sorted(SCRIPT_DIRECTORY.glob("*.sh")):
        assert os.access(path, os.X_OK), f"{path.name} must be executable"
        text = path.read_text(encoding="utf-8")
        assert text.startswith("#!/usr/bin/env bash\n")
        assert "set -Eeuo pipefail" in text

    assert {path.name for path in SCRIPT_DIRECTORY.glob("*.sh")} == {
        "bootstrap_runpod.sh",
        *expected_modules,
    }
    for name, module in expected_modules.items():
        text = (SCRIPT_DIRECTORY / name).read_text(encoding="utf-8")
        assert 'REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"' in text
        assert 'cd -- "${REPO_ROOT}"' in text
        assert f'"${{REPO_ROOT}}/.venv/bin/python" -m {module}' in text


def test_tts_extra_pins_the_approved_pytorch_cuda_stack() -> None:
    metadata = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["optional-dependencies"]["tts"] == [
        "qwen-tts==0.1.1",
        "torch==2.11.0",
        "torchaudio==2.11.0",
    ]
    assert metadata["tool"]["uv"]["index"] == [
        {
            "name": "pytorch-cu128",
            "url": "https://download.pytorch.org/whl/cu128",
            "explicit": True,
        }
    ]
    assert metadata["tool"]["uv"]["sources"] == {
        "torch": {"index": "pytorch-cu128"},
        "torchaudio": {"index": "pytorch-cu128"},
    }


def test_lockfile_resolves_only_the_approved_cuda_12_8_stack() -> None:
    lock_text = (REPOSITORY_ROOT / "uv.lock").read_text(encoding="utf-8")
    lock = tomllib.loads(lock_text)
    packages = {package["name"]: package for package in lock["package"]}

    expected_source = {"registry": "https://download.pytorch.org/whl/cu128"}
    assert packages["torch"]["version"] == "2.11.0+cu128"
    assert packages["torch"]["source"] == expected_source
    assert packages["torchaudio"]["version"] == "2.11.0+cu128"
    assert packages["torchaudio"]["source"] == expected_source
    assert packages["cuda-toolkit"]["version"].startswith("12.8.")

    package_names = set(packages)
    assert {
        "nvidia-cublas-cu12",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cudnn-cu12",
        "nvidia-cusparse-cu12",
    } <= package_names
    assert not {name for name in package_names if "cu13" in name}
    assert "/whl/cu13" not in lock_text


def test_bootstrap_covers_runpod_prerequisites_and_sdpa_fallback() -> None:
    text = (SCRIPT_DIRECTORY / "bootstrap_runpod.sh").read_text(encoding="utf-8")

    for required in (
        "ffmpeg",
        "libsndfile1",
        "build-essential",
        "git",
        "curl",
        "uv python install 3.12",
        "uv sync --locked --extra tts",
        "/workspace/huggingface_cache",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "TRANSFORMERS_CACHE",
        "torch.cuda.is_available()",
        "torch.cuda.is_bf16_supported()",
        "torch.cuda.get_device_name(0)",
        "import torchaudio",
        'EXPECTED_TORCH_VERSION = "2.11.0"',
        'EXPECTED_TORCHAUDIO_VERSION = "2.11.0"',
        'EXPECTED_CUDA_VERSION_PREFIX = "12.8"',
        "PyTorch 2.11.0 is required",
        "TorchAudio 2.11.0 is required",
        "PyTorch CUDA 12.8 runtime is required",
        "torch.__version__",
        "torchaudio.__version__",
        "MAX_JOBS=4",
        "flash-attn",
        "SDPA",
    ):
        assert required in text


def test_makefile_exposes_only_explicit_validation_execution() -> None:
    text = (REPOSITORY_ROOT / "Makefile").read_text(encoding="utf-8")
    expected_targets = {"setup", "prepare", "pilot", "test", "validation"}

    for target in expected_targets:
        assert _target_dependencies(text, target) == []
    for target in expected_targets - {"validation"}:
        assert "validation" not in _target_dependencies(text, target)

    assert "./scripts/bootstrap_runpod.sh" in text
    assert "./scripts/prepare_dataset.sh" in text
    assert "./scripts/synthesize_pilot.sh" in text
    assert "./scripts/synthesize_validation.sh" in text
    assert '.venv/bin/python -m pytest -m "not gpu"' in text
