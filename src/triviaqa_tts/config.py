"""Typed, validated configuration loading for the TriviaQA TTS pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from types import UnionType
from typing import Union, get_args, get_origin, get_type_hints

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "mandarjoshi/trivia_qa"
    configuration: str = "unfiltered.nocontext"
    split: str = "validation"
    revision: str | None = None


@dataclass(frozen=True)
class SelectionConfig:
    manifest_path: Path = field(
        default_factory=lambda: (Path.cwd() / "data/validation_manifest.jsonl").resolve()
    )
    pilot_size: int = 100
    pilot_seed: int = 20_260_721
    start_index: int = 0
    end_index: int | None = None
    shard_index: int = 0
    num_shards: int = 1
    limit: int | None = None
    resume: bool = False
    overwrite: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class TTSConfig:
    backend: str = "qwen3"
    model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    model_revision: str | None = None
    language: str = "English"
    speaker: str = "Aiden"
    seed: int = 42
    dtype: str = "bfloat16"
    attention: str = "auto"
    generation: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 24_000
    channels: int = 1
    subtype: str = "PCM_16"


@dataclass(frozen=True)
class AudioValidationConfig:
    minimum_duration_seconds: float = 0.3
    maximum_duration_seconds: float = 60.0
    clipping_threshold: float = 0.999
    silence_threshold: float = 0.01
    maximum_leading_silence_seconds: float = 2.0
    maximum_trailing_silence_seconds: float = 3.0


@dataclass(frozen=True)
class PathsConfig:
    data_dir: Path = field(default_factory=lambda: (Path.cwd() / "data").resolve())
    audio_dir: Path = field(
        default_factory=lambda: (Path.cwd() / "audio/qwen3_aiden_seed42").resolve()
    )
    logs_dir: Path = field(default_factory=lambda: (Path.cwd() / "logs").resolve())
    cache_dir: Path = Path("/workspace/huggingface_cache")


@dataclass(frozen=True)
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    audio_validation: AudioValidationConfig = field(default_factory=AudioValidationConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def __post_init__(self) -> None:
        _validate_config(self)


def load_config(path: Path) -> AppConfig:
    """Load a YAML mapping into a validated application configuration."""
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"unable to read configuration file {path}: {error}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML in configuration file {path}: {error}") from error

    if loaded is None:
        loaded = {}
    return _build_dataclass(AppConfig, loaded, "configuration")


def apply_cli_overrides(config: AppConfig, **values: object) -> AppConfig:
    """Return a validated copy with non-null selection CLI values applied."""
    selection_fields = {item.name for item in fields(SelectionConfig)}
    unknown = set(values).difference(selection_fields)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown CLI override: {names}")

    selection_values = {
        item.name: getattr(config.selection, item.name) for item in fields(SelectionConfig)
    }
    selection_values.update({name: value for name, value in values.items() if value is not None})
    selection = _build_dataclass(SelectionConfig, selection_values, "selection")
    return replace(config, selection=selection)


def _build_dataclass[T](cls: type[T], value: object, location: str) -> T:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a mapping")

    known = {item.name for item in fields(cls)}
    unknown = set(value).difference(known)
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise ValueError(f"unknown configuration key(s) at {location}: {names}")

    hints = get_type_hints(cls)
    parsed: dict[str, object] = {}
    for item in fields(cls):
        if item.name in value:
            parsed[item.name] = _parse_value(
                value[item.name], hints[item.name], f"{location}.{item.name}"
            )
    return cls(**parsed)


def _parse_value(value: object, annotation: object, location: str) -> object:
    if annotation is object:
        return value
    if is_dataclass(annotation):
        return _build_dataclass(annotation, value, location)
    if annotation is Path:
        if not isinstance(value, (str, Path)):
            raise ValueError(f"{location} must be a path string")
        path = Path(value)
        return path if path.is_absolute() else (Path.cwd() / path).resolve()

    origin = get_origin(annotation)
    if origin is dict:
        if not isinstance(value, Mapping):
            raise ValueError(f"{location} must be a mapping")
        return dict(value)
    if origin in (Union, UnionType):
        options = get_args(annotation)
        if value is None and type(None) in options:
            return None
        for option in options:
            if option is not type(None):
                try:
                    return _parse_value(value, option, location)
                except ValueError:
                    continue
        raise ValueError(f"{location} has an invalid value")
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{location} must be a number")
        return float(value)
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{location} must be an integer")
        return value
    if annotation is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{location} must be a boolean")
        return value
    if annotation is str:
        if not isinstance(value, str):
            raise ValueError(f"{location} must be a string")
        return value
    return value


def _validate_config(config: AppConfig) -> None:
    dataset = config.dataset
    if dataset.name != "mandarjoshi/trivia_qa":
        raise ValueError("dataset.name must be 'mandarjoshi/trivia_qa'")
    if dataset.configuration != "unfiltered.nocontext":
        raise ValueError("dataset.configuration must be 'unfiltered.nocontext'")
    if dataset.split != "validation":
        raise ValueError("dataset.split must be 'validation'")

    selection = config.selection
    if selection.pilot_size <= 0:
        raise ValueError("selection.pilot_size must be positive")
    if selection.pilot_seed < 0:
        raise ValueError("selection.pilot_seed must be non-negative")
    if selection.start_index < 0:
        raise ValueError("selection.start_index must be non-negative")
    if selection.end_index is not None and selection.end_index < selection.start_index:
        raise ValueError("selection.end_index must be at least selection.start_index")
    if selection.limit is not None and selection.limit <= 0:
        raise ValueError("selection.limit must be positive")
    if selection.num_shards <= 0:
        raise ValueError("selection.num_shards must be positive")
    if not 0 <= selection.shard_index < selection.num_shards:
        raise ValueError("selection.shard_index must satisfy 0 <= shard_index < num_shards")
    if selection.resume and selection.overwrite:
        raise ValueError("selection.resume and selection.overwrite are mutually exclusive")

    audio = config.audio
    if audio.sample_rate <= 0:
        raise ValueError("audio.sample_rate must be positive")
    if audio.channels != 1:
        raise ValueError("audio.channels must be 1 for mono output")
    if audio.subtype != "PCM_16":
        raise ValueError("audio.subtype must be PCM_16")

    validation = config.audio_validation
    if validation.minimum_duration_seconds <= 0:
        raise ValueError("audio_validation.minimum_duration_seconds must be positive")
    if validation.maximum_duration_seconds <= 0:
        raise ValueError("audio_validation.maximum_duration_seconds must be positive")
    if validation.maximum_duration_seconds < validation.minimum_duration_seconds:
        raise ValueError(
            "audio_validation.maximum_duration_seconds must be at least minimum_duration_seconds"
        )
    if not 0 <= validation.clipping_threshold <= 1:
        raise ValueError("audio_validation.clipping_threshold must be between 0 and 1")
    if validation.silence_threshold < 0:
        raise ValueError("audio_validation.silence_threshold must be non-negative")
    if validation.maximum_leading_silence_seconds < 0:
        raise ValueError("audio_validation.maximum_leading_silence_seconds must be non-negative")
    if validation.maximum_trailing_silence_seconds < 0:
        raise ValueError("audio_validation.maximum_trailing_silence_seconds must be non-negative")
