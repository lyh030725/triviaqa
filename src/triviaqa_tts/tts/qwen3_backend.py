"""Lazy, revision-pinned integration with the official Qwen3-TTS API."""

from __future__ import annotations

import importlib
import re
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from triviaqa_tts.config import TTSConfig

_OFFICIAL_SPEAKER = "Aiden"
_OFFICIAL_LANGUAGE = "English"
_SUPPORTED_ATTENTION = frozenset({"auto", "sdpa", "flash_attention_2"})


def resolve_model_revision(
    model_id: str,
    revision: str | None = None,
    *,
    hub_api: object | None = None,
) -> str:
    """Resolve a model branch, tag, or revision to an immutable Hub SHA."""
    if hub_api is None:
        from huggingface_hub import HfApi

        hub_api = HfApi()

    model_info = getattr(hub_api, "model_info", None)
    if not callable(model_info):
        raise TypeError("hub_api must provide a callable model_info method")
    info = model_info(repo_id=model_id, revision=revision or "main")
    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or not sha.strip():
        raise ValueError(f"Hub did not return a resolved SHA for model {model_id}")
    return sha


class Qwen3Backend:
    """One-model Qwen3-TTS CustomVoice backend for the approved project voice."""

    def __init__(
        self,
        *,
        model: object,
        model_id: str,
        model_revision: str,
        dtype: object,
        attention_implementation: str,
    ) -> None:
        self._model = model
        self._model_id = model_id
        self._model_revision = model_revision
        self._dtype = dtype
        self._attention_implementation = attention_implementation
        self._native_sample_rate: int | None = None

    @classmethod
    def from_config(
        cls,
        config: TTSConfig,
        *,
        hub_api: object | None = None,
        qwen_model_cls: object | None = None,
        torch_module: object | None = None,
    ) -> Qwen3Backend:
        """Resolve and load exactly the Qwen model and voice requested by config."""
        _validate_config(config)

        if torch_module is None:
            torch_module = importlib.import_module("torch")
        if qwen_model_cls is None:
            qwen_tts = importlib.import_module("qwen_tts")
            qwen_model_cls = getattr(qwen_tts, "Qwen3TTSModel")

        dtype = _resolve_dtype(torch_module, config.dtype)
        attention = _resolve_attention(config.attention, torch_module, dtype)
        revision = resolve_model_revision(
            config.model_id,
            config.model_revision,
            hub_api=hub_api,
        )

        from_pretrained = getattr(qwen_model_cls, "from_pretrained", None)
        if not callable(from_pretrained):
            raise TypeError("qwen_model_cls must provide a callable from_pretrained method")
        model = from_pretrained(
            config.model_id,
            revision=revision,
            device_map="cuda:0",
            dtype=dtype,
            attn_implementation=attention,
        )
        _validate_supported_value(
            model,
            method_name="get_supported_speakers",
            expected=_OFFICIAL_SPEAKER,
            value_kind="speaker",
        )
        _validate_supported_value(
            model,
            method_name="get_supported_languages",
            expected=_OFFICIAL_LANGUAGE,
            value_kind="language",
        )
        return cls(
            model=model,
            model_id=config.model_id,
            model_revision=revision,
            dtype=dtype,
            attention_implementation=attention,
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_revision(self) -> str:
        return self._model_revision

    @property
    def dtype(self) -> object:
        return self._dtype

    @property
    def attention_implementation(self) -> str:
        return self._attention_implementation

    @property
    def native_sample_rate(self) -> int | None:
        return self._native_sample_rate

    def synthesize(self, text: str, seed: int) -> tuple[np.ndarray, int]:
        """Synthesize one question with the official Aiden CustomVoice call."""
        del seed  # Seeding is reset centrally immediately before each backend call.
        generate = getattr(self._model, "generate_custom_voice", None)
        if not callable(generate):
            raise TypeError("loaded model must provide generate_custom_voice")
        result = generate(
            text=text,
            language=_OFFICIAL_LANGUAGE,
            speaker=_OFFICIAL_SPEAKER,
            instruct="",
        )
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError("Qwen generation must return (waveforms, sample_rate)")
        waveforms, sample_rate = result
        if (
            not isinstance(waveforms, Sequence)
            or isinstance(waveforms, (str, bytes))
            or len(waveforms) != 1
        ):
            raise ValueError("Qwen generation must return exactly one waveform")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("Qwen sample rate must be a positive integer")

        try:
            waveform = np.asarray(waveforms[0])
            is_finite = bool(np.isfinite(waveform).all())
        except (TypeError, ValueError) as error:
            raise ValueError("Qwen waveform must be NumPy-compatible and finite") from error
        if waveform.size == 0:
            raise ValueError("Qwen waveform must not be empty")
        if not is_finite:
            raise ValueError("Qwen waveform must contain only finite values")

        self._native_sample_rate = sample_rate
        return waveform, sample_rate


def _validate_config(config: TTSConfig) -> None:
    if config.backend.casefold() != "qwen3":
        raise ValueError("tts.backend must be qwen3")
    if config.speaker.casefold() != _OFFICIAL_SPEAKER.casefold():
        raise ValueError(f"tts.speaker must be {_OFFICIAL_SPEAKER}")
    if config.language.casefold() != _OFFICIAL_LANGUAGE.casefold():
        raise ValueError(f"tts.language must be {_OFFICIAL_LANGUAGE}")
    if config.attention not in _SUPPORTED_ATTENTION:
        choices = ", ".join(sorted(_SUPPORTED_ATTENTION))
        raise ValueError(f"tts.attention must be one of: {choices}")
    if config.generation:
        raise ValueError("tts.generation must be empty for the fixed CustomVoice API call")


def _resolve_dtype(torch_module: object, configured_dtype: str) -> object:
    attribute_name = {"bf16": "bfloat16", "fp16": "float16"}.get(
        configured_dtype.casefold(), configured_dtype.casefold()
    )
    dtype = getattr(torch_module, attribute_name, None)
    if dtype is None:
        raise ValueError(f"torch does not support configured dtype {configured_dtype!r}")
    return dtype


def _resolve_attention(requested: str, torch_module: object, dtype: object) -> str:
    if requested == "sdpa":
        return "sdpa"
    if _flash_attention_is_compatible(torch_module, dtype):
        return "flash_attention_2"
    return "sdpa"


def _flash_attention_is_compatible(torch_module: object, dtype: object) -> bool:
    try:
        importlib.import_module("flash_attn")
    except (ImportError, OSError, RuntimeError):
        return False

    if _version_tuple(getattr(torch_module, "__version__", "")) < (2, 2):
        return False
    cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
    if _version_tuple(cuda_version) < (12, 0):
        return False

    cuda = getattr(torch_module, "cuda", None)
    try:
        if cuda is None or not cuda.is_available():
            return False
        if not cuda.is_bf16_supported():
            return False
        if cuda.get_device_capability(0)[0] < 8:
            return False
    except (AttributeError, RuntimeError, TypeError):
        return False

    supported_dtypes = {
        getattr(torch_module, "bfloat16", None),
        getattr(torch_module, "float16", None),
    }
    return dtype in supported_dtypes


def _version_tuple(version: object) -> tuple[int, int]:
    if not isinstance(version, str):
        return (0, 0)
    match = re.match(r"\s*(\d+)\.(\d+)", version)
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


def _validate_supported_value(
    model: object,
    *,
    method_name: str,
    expected: str,
    value_kind: str,
) -> None:
    get_supported = getattr(model, method_name, None)
    if not callable(get_supported):
        raise TypeError(f"loaded model must provide {method_name}")
    values: Any = get_supported()
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ValueError(f"model supported {value_kind}s must be an iterable of strings")
    normalized = {value.casefold() for value in values if isinstance(value, str)}
    if expected.casefold() not in normalized:
        raise ValueError(f"{expected} is not a supported model {value_kind}")
