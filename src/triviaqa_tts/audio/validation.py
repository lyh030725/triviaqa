"""Convert waveforms and publish only decoded, validated PCM16 WAV files."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from triviaqa_tts.config import AppConfig, AudioConfig, AudioValidationConfig

AudioSettings = AppConfig | AudioConfig


class AudioValidationError(ValueError):
    """Raised when audio cannot satisfy a required storage invariant."""


@dataclass(frozen=True)
class AudioMetrics:
    """Quality and storage metrics computed from the decoded output WAV."""

    sample_rate: int
    channels: int
    subtype: str
    sample_count: int
    duration_seconds: float
    peak_amplitude: float
    clipped_sample_count: int
    zero_ratio: float
    leading_silence_seconds: float
    trailing_silence_seconds: float
    warning_reasons: tuple[str, ...]
    status: str


def prepare_waveform(
    waveform: np.ndarray, native_sample_rate: int, config: AudioSettings
) -> np.ndarray:
    """Return finite mono float32 audio at the configured target rate."""
    audio, _ = _settings(config)
    _validate_audio_config(audio)
    if isinstance(native_sample_rate, bool) or not isinstance(native_sample_rate, int):
        raise AudioValidationError("native sample rate must be a positive integer")
    if native_sample_rate <= 0:
        raise AudioValidationError("native sample rate must be positive")

    try:
        source = np.asarray(waveform)
    except (TypeError, ValueError) as error:
        raise AudioValidationError(f"waveform must be numeric: {error}") from error
    if source.size == 0:
        raise AudioValidationError("waveform is empty")
    if not np.issubdtype(source.dtype, np.number) or np.issubdtype(
        source.dtype, np.complexfloating
    ):
        raise AudioValidationError("waveform must contain real numeric samples")
    if np.isnan(source).any():
        raise AudioValidationError("waveform contains NaN samples")
    if np.isinf(source).any():
        raise AudioValidationError("waveform contains Inf samples")

    with np.errstate(over="ignore", invalid="ignore"):
        samples = source.astype(np.float32, copy=False)
    if np.isnan(samples).any():
        raise AudioValidationError("waveform contains NaN samples")
    if np.isinf(samples).any():
        raise AudioValidationError("waveform contains Inf samples")

    mono = _to_mono(samples)
    if native_sample_rate != audio.sample_rate:
        mono = np.asarray(
            soxr.resample(mono, native_sample_rate, audio.sample_rate, quality="HQ"),
            dtype=np.float32,
        )
    else:
        mono = np.asarray(mono, dtype=np.float32)

    if mono.size == 0:
        raise AudioValidationError("prepared waveform is empty")
    if np.isnan(mono).any():
        raise AudioValidationError("prepared waveform contains NaN samples")
    if np.isinf(mono).any():
        raise AudioValidationError("prepared waveform contains Inf samples")
    return np.ascontiguousarray(mono)


def write_validated_wav(
    waveform: np.ndarray,
    native_sample_rate: int,
    output_path: Path,
    config: AudioSettings,
) -> AudioMetrics:
    """Write, decode, validate, and atomically publish a WAV file."""
    audio, _ = _settings(config)
    prepared = prepare_waveform(waveform, native_sample_rate, config)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            sf.write(
                temporary_path,
                prepared,
                audio.sample_rate,
                format="WAV",
                subtype="PCM_16",
            )
        except (OSError, RuntimeError, sf.SoundFileError) as error:
            raise AudioValidationError(f"unable to write temporary WAV: {error}") from error

        metrics = validate_wav(temporary_path, config)
        os.replace(temporary_path, destination)
        temporary_path = None
        return metrics
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def validate_wav(path: Path, config: AudioSettings) -> AudioMetrics:
    """Decode a WAV and return metrics after enforcing required invariants."""
    audio, validation = _settings(config)
    _validate_audio_config(audio)
    wav_path = Path(path)
    try:
        if not wav_path.is_file() or wav_path.stat().st_size <= 0:
            raise AudioValidationError(f"audio file is missing or empty: {wav_path}")
    except OSError as error:
        raise AudioValidationError(f"unable to inspect audio file {wav_path}: {error}") from error

    try:
        info = sf.info(wav_path)
        decoded, decoded_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    except (OSError, RuntimeError, sf.SoundFileError) as error:
        raise AudioValidationError(f"unable to decode audio file {wav_path}: {error}") from error

    if info.format != "WAV":
        raise AudioValidationError(f"audio format must be WAV, got {info.format}")
    if info.subtype != audio.subtype:
        raise AudioValidationError(
            f"audio subtype must be {audio.subtype}, got {info.subtype}"
        )
    if info.channels != audio.channels or decoded.shape[1] != audio.channels:
        raise AudioValidationError(
            f"audio channel count must be {audio.channels}, got {info.channels}"
        )
    if info.samplerate != audio.sample_rate or decoded_rate != audio.sample_rate:
        raise AudioValidationError(
            f"audio sample rate must be {audio.sample_rate}, got {info.samplerate}"
        )
    if info.frames <= 0 or decoded.shape[0] <= 0:
        raise AudioValidationError("decoded audio is empty")

    samples = decoded[:, 0]
    if np.isnan(samples).any():
        raise AudioValidationError("decoded audio contains NaN samples")
    if np.isinf(samples).any():
        raise AudioValidationError("decoded audio contains Inf samples")
    if np.any(samples < -1.0) or np.any(samples > 1.0):
        raise AudioValidationError("decoded audio samples are outside the valid [-1, 1] range")

    sample_count = int(samples.size)
    duration_seconds = sample_count / audio.sample_rate
    if duration_seconds <= 0:
        raise AudioValidationError("decoded audio duration must be positive")

    absolute = np.abs(samples)
    peak_amplitude = float(np.max(absolute))
    clipped_sample_count = int(np.count_nonzero(absolute >= validation.clipping_threshold))
    zero_ratio = float(np.count_nonzero(samples == 0.0) / sample_count)
    silent = absolute <= validation.silence_threshold
    leading_samples = _edge_run_length(silent)
    trailing_samples = _edge_run_length(silent[::-1])
    leading_silence_seconds = leading_samples / audio.sample_rate
    trailing_silence_seconds = trailing_samples / audio.sample_rate

    warnings: list[str] = []
    if duration_seconds < validation.minimum_duration_seconds:
        warnings.append("duration_below_minimum")
    if duration_seconds > validation.maximum_duration_seconds:
        warnings.append("duration_above_maximum")
    if clipped_sample_count:
        warnings.append("clipping_detected")
    if leading_silence_seconds > validation.maximum_leading_silence_seconds:
        warnings.append("leading_silence_exceeds_maximum")
    if trailing_silence_seconds > validation.maximum_trailing_silence_seconds:
        warnings.append("trailing_silence_exceeds_maximum")

    warning_reasons = tuple(warnings)
    return AudioMetrics(
        sample_rate=info.samplerate,
        channels=info.channels,
        subtype=info.subtype,
        sample_count=sample_count,
        duration_seconds=duration_seconds,
        peak_amplitude=peak_amplitude,
        clipped_sample_count=clipped_sample_count,
        zero_ratio=zero_ratio,
        leading_silence_seconds=leading_silence_seconds,
        trailing_silence_seconds=trailing_silence_seconds,
        warning_reasons=warning_reasons,
        status="warning" if warning_reasons else "success",
    )


def _settings(config: AudioSettings) -> tuple[AudioConfig, AudioValidationConfig]:
    if isinstance(config, AppConfig):
        return config.audio, config.audio_validation
    if isinstance(config, AudioConfig):
        return config, AudioValidationConfig()
    raise TypeError("config must be an AppConfig or AudioConfig")


def _validate_audio_config(audio: AudioConfig) -> None:
    if audio.sample_rate <= 0:
        raise AudioValidationError("target sample rate must be positive")
    if audio.channels != 1:
        raise AudioValidationError("target audio must have one channel")
    if audio.subtype != "PCM_16":
        raise AudioValidationError("target audio subtype must be PCM_16")


def _to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples
    if samples.ndim != 2:
        raise AudioValidationError("waveform must be one- or two-dimensional")
    if 0 in samples.shape:
        raise AudioValidationError("waveform is empty")
    if samples.shape[1] == 1:
        return samples[:, 0]
    if samples.shape[0] == 1:
        return samples[0]
    if samples.shape[1] <= 8 or samples.shape[0] > 8:
        return np.mean(samples, axis=1, dtype=np.float32)
    return np.mean(samples, axis=0, dtype=np.float32)


def _edge_run_length(mask: np.ndarray) -> int:
    non_silent = np.flatnonzero(~mask)
    return int(non_silent[0]) if non_silent.size else int(mask.size)
