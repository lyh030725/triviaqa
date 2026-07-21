from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from triviaqa_tts.audio import (
    AudioValidationError,
    prepare_waveform,
    validate_wav,
    write_validated_wav,
)
from triviaqa_tts.config import AppConfig, AudioConfig, AudioValidationConfig


@pytest.fixture
def audio_config() -> AppConfig:
    return AppConfig(
        audio=AudioConfig(sample_rate=24_000, channels=1, subtype="PCM_16"),
        audio_validation=AudioValidationConfig(
            minimum_duration_seconds=0.3,
            maximum_duration_seconds=2.0,
            clipping_threshold=0.999,
            silence_threshold=0.01,
            maximum_leading_silence_seconds=0.1,
            maximum_trailing_silence_seconds=0.1,
        ),
    )


def tone(sample_rate: int, duration: float = 0.5, amplitude: float = 0.25) -> np.ndarray:
    time = np.arange(round(sample_rate * duration), dtype=np.float32) / sample_rate
    return (amplitude * np.sin(2 * np.pi * 440 * time)).astype(np.float32)


def test_mono_pcm16_24khz_round_trip(tmp_path: Path, audio_config: AppConfig) -> None:
    output = tmp_path / "tone.wav"

    metrics = write_validated_wav(tone(24_000), 24_000, output, audio_config)

    info = sf.info(output)
    decoded, sample_rate = sf.read(output, dtype="float32")
    assert sample_rate == 24_000
    assert info.format == "WAV"
    assert info.subtype == "PCM_16"
    assert info.channels == 1
    assert decoded.dtype == np.float32
    assert metrics.sample_rate == 24_000
    assert metrics.channels == 1
    assert metrics.subtype == "PCM_16"
    assert metrics.sample_count == len(decoded)
    assert metrics.duration_seconds == pytest.approx(0.5)
    assert metrics.peak_amplitude == pytest.approx(float(np.max(np.abs(decoded))))
    assert metrics.warning_reasons == ()
    assert metrics.status == "success"


def test_prepare_waveform_averages_stereo_without_normalizing(audio_config: AppConfig) -> None:
    stereo = np.array([[0.2, 0.6], [-0.4, 0.2]], dtype=np.float32)

    prepared = prepare_waveform(stereo, 24_000, audio_config)

    np.testing.assert_allclose(prepared, np.array([0.4, -0.1], dtype=np.float32))
    assert prepared.dtype == np.float32


def test_prepare_waveform_accepts_channel_first_singleton(audio_config: AppConfig) -> None:
    waveform = np.array([[0.1, -0.2, 0.3]], dtype=np.float32)

    prepared = prepare_waveform(waveform, 24_000, audio_config)

    np.testing.assert_array_equal(prepared, waveform[0])


def test_resamples_16khz_to_24khz(tmp_path: Path, audio_config: AppConfig) -> None:
    output = tmp_path / "resampled.wav"

    metrics = write_validated_wav(tone(16_000), 16_000, output, audio_config)

    info = sf.info(output)
    assert info.samplerate == 24_000
    assert info.frames == pytest.approx(12_000, abs=1)
    assert metrics.duration_seconds == pytest.approx(0.5, abs=1 / 24_000)


def test_equal_sample_rate_does_not_call_soxr(
    monkeypatch: pytest.MonkeyPatch, audio_config: AppConfig
) -> None:
    def unexpected_resample(*args: object, **kwargs: object) -> None:
        raise AssertionError("soxr must not be called when rates match")

    monkeypatch.setattr("triviaqa_tts.audio.validation.soxr.resample", unexpected_resample)

    np.testing.assert_array_equal(
        prepare_waveform(np.array([0.1, -0.2], dtype=np.float32), 24_000, audio_config),
        np.array([0.1, -0.2], dtype=np.float32),
    )


def test_resampling_uses_hq_quality(
    monkeypatch: pytest.MonkeyPatch, audio_config: AppConfig
) -> None:
    observed: dict[str, object] = {}

    def fake_resample(
        waveform: np.ndarray, native_rate: int, target_rate: int, *, quality: str
    ) -> np.ndarray:
        observed.update(
            waveform=waveform,
            native_rate=native_rate,
            target_rate=target_rate,
            quality=quality,
        )
        return np.array([0.25], dtype=np.float32)

    monkeypatch.setattr("triviaqa_tts.audio.validation.soxr.resample", fake_resample)

    prepared = prepare_waveform(np.array([0.5], dtype=np.float32), 16_000, audio_config)

    np.testing.assert_array_equal(prepared, np.array([0.25], dtype=np.float32))
    assert observed["native_rate"] == 16_000
    assert observed["target_rate"] == 24_000
    assert observed["quality"] == "HQ"


@pytest.mark.parametrize(
    ("waveform", "message"),
    [
        (np.array([], dtype=np.float32), "empty"),
        (np.array([0.0, np.nan, 0.1], dtype=np.float32), "NaN"),
        (np.array([0.0, np.inf, 0.1], dtype=np.float32), "Inf"),
        (np.array([0.0, -np.inf, 0.1], dtype=np.float32), "Inf"),
    ],
)
def test_invalid_input_is_rejected_before_writing(
    tmp_path: Path, audio_config: AppConfig, waveform: np.ndarray, message: str
) -> None:
    output = tmp_path / "bad.wav"

    with pytest.raises(AudioValidationError, match=message):
        write_validated_wav(waveform, 24_000, output, audio_config)

    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_clipping_is_a_warning_and_preserves_file(
    tmp_path: Path, audio_config: AppConfig
) -> None:
    output = tmp_path / "clipped.wav"
    waveform = tone(24_000)
    waveform[:2] = np.array([1.0, -1.0], dtype=np.float32)

    metrics = write_validated_wav(waveform, 24_000, output, audio_config)

    assert output.exists()
    assert metrics.clipped_sample_count == 2
    assert "clipping_detected" in metrics.warning_reasons
    assert metrics.status == "warning"


def test_short_audio_is_a_warning_and_preserves_file(
    tmp_path: Path, audio_config: AppConfig
) -> None:
    output = tmp_path / "short.wav"

    metrics = write_validated_wav(tone(24_000, duration=0.2), 24_000, output, audio_config)

    assert output.exists()
    assert "duration_below_minimum" in metrics.warning_reasons


def test_long_audio_is_a_warning_and_preserves_file(
    tmp_path: Path, audio_config: AppConfig
) -> None:
    output = tmp_path / "long.wav"

    metrics = write_validated_wav(tone(24_000, duration=2.1), 24_000, output, audio_config)

    assert output.exists()
    assert "duration_above_maximum" in metrics.warning_reasons


def test_leading_and_trailing_silence_are_measured_from_decoded_audio(
    tmp_path: Path, audio_config: AppConfig
) -> None:
    output = tmp_path / "silence.wav"
    waveform = np.concatenate(
        [
            np.zeros(3_600, dtype=np.float32),
            np.full(4_800, 0.5, dtype=np.float32),
            np.zeros(4_800, dtype=np.float32),
        ]
    )

    metrics = write_validated_wav(waveform, 24_000, output, audio_config)

    assert metrics.leading_silence_seconds == pytest.approx(0.15)
    assert metrics.trailing_silence_seconds == pytest.approx(0.2)
    assert "leading_silence_exceeds_maximum" in metrics.warning_reasons
    assert "trailing_silence_exceeds_maximum" in metrics.warning_reasons


def test_zero_ratio_uses_decoded_samples(tmp_path: Path, audio_config: AppConfig) -> None:
    output = tmp_path / "zeros.wav"
    waveform = np.resize(np.array([0.0, 0.5, 0.0, -0.5], dtype=np.float32), 12_000)

    metrics = write_validated_wav(waveform, 24_000, output, audio_config)

    assert metrics.zero_ratio == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("writer", "message"),
    [
        (lambda path: path.write_bytes(b"not a wav"), "decode"),
        (
            lambda path: sf.write(
                path, np.ones((4_800, 2), dtype=np.float32), 24_000, subtype="PCM_16"
            ),
            "channel",
        ),
        (lambda path: sf.write(path, np.ones(4_800), 16_000, subtype="PCM_16"), "sample rate"),
        (
            lambda path: sf.write(path, np.ones(4_800), 24_000, subtype="FLOAT"),
            "subtype",
        ),
        (
            lambda path: sf.write(
                path, np.array([], dtype=np.float32), 24_000, subtype="PCM_16"
            ),
            "empty",
        ),
    ],
)
def test_validate_wav_rejects_required_invariants(
    tmp_path: Path,
    audio_config: AppConfig,
    writer: object,
    message: str,
) -> None:
    output = tmp_path / "invalid.wav"
    writer(output)  # type: ignore[operator]

    with pytest.raises(AudioValidationError, match=message):
        validate_wav(output, audio_config)


def test_validation_failure_cleans_temp_and_does_not_publish(
    tmp_path: Path, audio_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "bad.wav"

    def fail_validation(path: Path, config: AppConfig) -> None:
        assert path.parent == output.parent
        assert path != output
        raise AudioValidationError("forced validation failure")

    monkeypatch.setattr("triviaqa_tts.audio.validation.validate_wav", fail_validation)

    with pytest.raises(AudioValidationError, match="forced"):
        write_validated_wav(tone(24_000), 24_000, output, audio_config)

    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_validation_failure_preserves_existing_destination(
    tmp_path: Path, audio_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing.wav"
    output.write_bytes(b"original")

    def fail_validation(path: Path, config: AppConfig) -> None:
        raise AudioValidationError("forced validation failure")

    monkeypatch.setattr("triviaqa_tts.audio.validation.validate_wav", fail_validation)

    with pytest.raises(AudioValidationError, match="forced"):
        write_validated_wav(tone(24_000), 24_000, output, audio_config)

    assert output.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [output]


def test_plain_audio_config_uses_default_validation_thresholds(tmp_path: Path) -> None:
    config = AudioConfig()
    output = tmp_path / "plain.wav"

    metrics = write_validated_wav(tone(24_000), 24_000, output, config)

    assert metrics.status == "success"


def test_all_silent_audio_has_full_silence_durations(
    tmp_path: Path, audio_config: AppConfig
) -> None:
    config = replace(
        audio_config,
        audio_validation=replace(
            audio_config.audio_validation,
            minimum_duration_seconds=0.1,
            maximum_leading_silence_seconds=1.0,
            maximum_trailing_silence_seconds=1.0,
        ),
    )

    metrics = write_validated_wav(
        np.zeros(4_800, dtype=np.float32), 24_000, tmp_path / "silent.wav", config
    )

    assert metrics.leading_silence_seconds == pytest.approx(0.2)
    assert metrics.trailing_silence_seconds == pytest.approx(0.2)
    assert metrics.zero_ratio == 1.0
