from __future__ import annotations

import pytest

from triviaqa_tts.config import (
    AppConfig,
    AudioValidationConfig,
    apply_cli_overrides,
    load_config,
)


def test_load_config_and_reject_conflicting_modes(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("dataset:\n  name: mandarjoshi/trivia_qa\n", encoding="utf-8")

    config = load_config(path)

    assert config.tts.model_id == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    assert config.audio.sample_rate == 24_000
    with pytest.raises(ValueError, match="mutually exclusive"):
        apply_cli_overrides(config, resume=True, overwrite=True)
    assert config.audio_validation.maximum_zero_ratio == 0.99


@pytest.mark.parametrize("maximum_zero_ratio", [-0.01, 1.01])
def test_maximum_zero_ratio_must_be_between_zero_and_one(
    maximum_zero_ratio: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="audio_validation.maximum_zero_ratio must be between 0 and 1",
    ):
        AppConfig(audio_validation=AudioValidationConfig(maximum_zero_ratio=maximum_zero_ratio))
