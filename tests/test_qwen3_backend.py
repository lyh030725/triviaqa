from __future__ import annotations

import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from triviaqa_tts.config import TTSConfig


class FakeHubApi:
    def __init__(self, sha: str = "a" * 40) -> None:
        self.sha = sha
        self.calls: list[tuple[str, str | None]] = []

    def model_info(self, repo_id: str, *, revision: str | None) -> SimpleNamespace:
        self.calls.append((repo_id, revision))
        return SimpleNamespace(sha=self.sha)


class FakeModel:
    def __init__(
        self,
        *,
        speakers: list[str] | None = None,
        languages: list[str] | None = None,
        result: tuple[object, object] | None = None,
    ) -> None:
        self.speakers = speakers or ["Aiden", "Ryan"]
        self.languages = languages or ["English", "Chinese"]
        self.result = result or ([np.array([0.25, -0.25], dtype=np.float32)], 24_000)
        self.generation_calls: list[dict[str, object]] = []

    def get_supported_speakers(self) -> list[str]:
        return self.speakers

    def get_supported_languages(self) -> list[str]:
        return self.languages

    def generate_custom_voice(self, **kwargs: object) -> tuple[object, object]:
        self.generation_calls.append(kwargs)
        return self.result


class FakeModelClass:
    def __init__(self, model: FakeModel) -> None:
        self.model = model
        self.calls: list[tuple[str, dict[str, object]]] = []

    def from_pretrained(self, model_id: str, **kwargs: object) -> FakeModel:
        self.calls.append((model_id, kwargs))
        return self.model


class FakeTorch:
    __version__ = "2.7.1+cu128"
    bfloat16 = object()
    float16 = object()
    version = SimpleNamespace(cuda="12.8")

    def __init__(self, *, flash_compatible: bool = True) -> None:
        self.cuda = SimpleNamespace(
            is_available=lambda: flash_compatible,
            is_bf16_supported=lambda: flash_compatible,
            get_device_capability=lambda index: (8, 9) if flash_compatible else (7, 5),
        )


def load_backend(
    *,
    config: TTSConfig | None = None,
    model: FakeModel | None = None,
    torch_module: FakeTorch | None = None,
    hub_api: FakeHubApi | None = None,
):
    from triviaqa_tts.tts.qwen3_backend import Qwen3Backend

    model = model or FakeModel()
    model_class = FakeModelClass(model)
    hub_api = hub_api or FakeHubApi()
    backend = Qwen3Backend.from_config(
        config or TTSConfig(),
        hub_api=hub_api,
        qwen_model_cls=model_class,
        torch_module=torch_module or FakeTorch(flash_compatible=False),
    )
    return backend, model, model_class, hub_api


def test_from_config_pins_model_sha_and_loads_official_bfloat16_model() -> None:
    config = replace(TTSConfig(), model_revision="release-tag", attention="sdpa")
    backend, _, model_class, hub_api = load_backend(config=config)

    assert hub_api.calls == [(config.model_id, "release-tag")]
    assert model_class.calls == [
        (
            config.model_id,
            {
                "revision": "a" * 40,
                "device_map": "cuda:0",
                "dtype": backend.dtype,
                "attn_implementation": "sdpa",
            },
        )
    ]
    assert backend.model_id == config.model_id
    assert backend.model_revision == "a" * 40
    assert backend.dtype is FakeTorch.bfloat16
    assert backend.attention_implementation == "sdpa"
    assert backend.native_sample_rate is None


def test_synthesize_uses_exact_official_custom_voice_arguments_and_unwraps_result() -> None:
    backend, model, _, _ = load_backend()

    waveform, sample_rate = backend.synthesize("Who invented the telephone?", seed=42)

    assert model.generation_calls == [
        {
            "text": "Who invented the telephone?",
            "language": "English",
            "speaker": "Aiden",
            "instruct": "",
        }
    ]
    np.testing.assert_array_equal(waveform, np.array([0.25, -0.25], dtype=np.float32))
    assert sample_rate == 24_000
    assert backend.native_sample_rate == 24_000


def test_supported_speaker_and_language_matching_is_case_insensitive() -> None:
    config = replace(TTSConfig(), speaker="aIdEn", language="ENGLISH")
    model = FakeModel(speakers=["aiden"], languages=["english"])

    backend, _, _, _ = load_backend(config=config, model=model)

    backend.synthesize("Question?", seed=42)
    assert model.generation_calls[0]["speaker"] == "Aiden"
    assert model.generation_calls[0]["language"] == "English"


@pytest.mark.parametrize(
    ("config", "model", "match"),
    [
        (replace(TTSConfig(), speaker="Ryan"), FakeModel(), "speaker.*Aiden"),
        (replace(TTSConfig(), language="Chinese"), FakeModel(), "language.*English"),
        (TTSConfig(), FakeModel(speakers=["Ryan"]), "Aiden.*supported"),
        (TTSConfig(), FakeModel(languages=["Chinese"]), "English.*supported"),
    ],
    ids=(
        "configured-speaker",
        "configured-language",
        "model-speaker-list",
        "model-language-list",
    ),
)
def test_loader_rejects_unapproved_or_unsupported_voice_settings(
    config: TTSConfig, model: FakeModel, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        load_backend(config=config, model=model)


def test_auto_attention_uses_flash_attention_2_when_importable_and_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "flash_attn", SimpleNamespace())

    backend, _, model_class, _ = load_backend(torch_module=FakeTorch(flash_compatible=True))

    assert backend.attention_implementation == "flash_attention_2"
    assert model_class.calls[0][1]["attn_implementation"] == "flash_attention_2"


@pytest.mark.parametrize("flash_importable", [False, True])
def test_auto_attention_falls_back_to_sdpa_unless_flash_is_importable_and_compatible(
    monkeypatch: pytest.MonkeyPatch, flash_importable: bool
) -> None:
    if flash_importable:
        monkeypatch.setitem(sys.modules, "flash_attn", SimpleNamespace())
    else:
        monkeypatch.setitem(sys.modules, "flash_attn", None)

    backend, _, _, _ = load_backend(torch_module=FakeTorch(flash_compatible=False))

    assert backend.attention_implementation == "sdpa"


@pytest.mark.parametrize(
    ("result", "match"),
    [
        (([], 24_000), "exactly one waveform"),
        (([np.zeros(2), np.ones(2)], 24_000), "exactly one waveform"),
        (([np.array([np.nan])], 24_000), "finite"),
        (([np.array([np.inf])], 24_000), "finite"),
        (([np.zeros(2)], 0), "positive integer"),
        (([np.zeros(2)], 24_000.0), "positive integer"),
        (([np.zeros(2)], True), "positive integer"),
    ],
)
def test_synthesize_rejects_malformed_model_results(
    result: tuple[object, object], match: str
) -> None:
    backend, _, _, _ = load_backend(model=FakeModel(result=result))

    with pytest.raises(ValueError, match=match):
        backend.synthesize("Question?", seed=42)


def test_importing_backend_does_not_import_torch_or_qwen_tts() -> None:
    __import__("triviaqa_tts.tts.qwen3_backend")

    assert "torch" not in sys.modules
    assert "qwen_tts" not in sys.modules
