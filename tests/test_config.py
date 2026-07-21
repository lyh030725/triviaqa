from __future__ import annotations

import pytest

from triviaqa_tts.config import apply_cli_overrides, load_config


def test_load_config_and_reject_conflicting_modes(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("dataset:\n  name: mandarjoshi/trivia_qa\n", encoding="utf-8")

    config = load_config(path)

    assert config.tts.model_id == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    assert config.audio.sample_rate == 24_000
    with pytest.raises(ValueError, match="mutually exclusive"):
        apply_cli_overrides(config, resume=True, overwrite=True)
