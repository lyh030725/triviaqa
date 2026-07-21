from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from triviaqa_tts.audio.validation import write_validated_wav
from triviaqa_tts.config import AudioConfig
from triviaqa_tts.data.manifest import ManifestFormatError, normalize_question, sha256_file
from triviaqa_tts.tts.synthesize import (
    append_jsonl_fsync,
    is_complete,
    load_latest_metadata,
)


@pytest.fixture
def resume_case(tmp_path: Path):
    record = {"question_id": "q-1", "question": " What &amp; why  "}
    wav_path = tmp_path / "q-1.wav"
    audio_config = AudioConfig()
    time = np.arange(int(audio_config.sample_rate * 0.4), dtype=np.float32)
    waveform = 0.2 * np.sin(2 * np.pi * 440 * time / audio_config.sample_rate)
    write_validated_wav(waveform, audio_config.sample_rate, wav_path, audio_config)
    expected = {
        "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "model_revision": "a" * 40,
        "speaker": "Aiden",
        "language": "English",
        "seed": 42,
        "generation": {},
    }
    question_sha256 = hashlib.sha256(
        normalize_question(record["question"]).encode("utf-8")
    ).hexdigest()
    metadata = {
        "question_id": record["question_id"],
        "status": "success",
        "audio_sha256": sha256_file(wav_path),
        "question_sha256": question_sha256,
        **expected,
        "sample_rate": audio_config.sample_rate,
        "channels": audio_config.channels,
        "subtype": audio_config.subtype,
    }
    return record, metadata, wav_path, expected, audio_config


@pytest.mark.parametrize("status", ["success", "warning"])
def test_complete_accepts_only_matching_success_or_warning(resume_case, status):
    record, metadata, wav_path, expected, audio_config = resume_case
    metadata["status"] = status

    assert is_complete(record, metadata, wav_path, expected, audio_config)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("status", "failure"),
        ("audio_sha256", "0" * 64),
        ("question_sha256", "0" * 64),
        ("model_id", "different/model"),
        ("model_revision", "b" * 40),
        ("speaker", "Ryan"),
        ("language", "Chinese"),
        ("seed", 43),
        ("generation", {"temperature": 0.7}),
        ("sample_rate", 16_000),
        ("channels", 2),
        ("subtype", "FLOAT"),
    ],
)
def test_complete_rejects_metadata_mismatch(resume_case, field, replacement):
    record, metadata, wav_path, expected, audio_config = resume_case
    metadata[field] = replacement

    assert not is_complete(record, metadata, wav_path, expected, audio_config)


def test_complete_rejects_missing_metadata(resume_case):
    record, _, wav_path, expected, audio_config = resume_case

    assert not is_complete(record, None, wav_path, expected, audio_config)


def test_complete_rejects_absent_wav(resume_case):
    record, metadata, wav_path, expected, audio_config = resume_case
    wav_path.unlink()

    assert not is_complete(record, metadata, wav_path, expected, audio_config)


def test_complete_rejects_invalid_wav_even_when_hash_matches(resume_case):
    record, metadata, wav_path, expected, audio_config = resume_case
    wav_path.write_bytes(b"not a wav")
    metadata["audio_sha256"] = sha256_file(wav_path)

    assert not is_complete(record, metadata, wav_path, expected, audio_config)


def test_complete_rejects_wrong_question_identity(resume_case):
    record, metadata, wav_path, expected, audio_config = resume_case
    metadata["question_id"] = "another-question"

    assert not is_complete(record, metadata, wav_path, expected, audio_config)


@pytest.mark.parametrize(
    "field",
    [
        "model_id",
        "model_revision",
        "speaker",
        "language",
        "seed",
        "generation",
        "sample_rate",
        "channels",
        "subtype",
    ],
)
def test_complete_rejects_missing_required_identity_or_format_field(resume_case, field):
    record, metadata, wav_path, expected, audio_config = resume_case
    del metadata[field]

    assert not is_complete(record, metadata, wav_path, expected, audio_config)


@pytest.mark.parametrize("status", [[], {"state": "success"}, 1, True, None])
def test_complete_fails_closed_for_malformed_status(resume_case, status):
    record, metadata, wav_path, expected, audio_config = resume_case
    metadata["status"] = status

    assert not is_complete(record, metadata, wav_path, expected, audio_config)


def test_complete_requires_present_nullable_model_revision(resume_case):
    record, metadata, wav_path, expected, audio_config = resume_case
    expected["model_revision"] = None
    metadata["model_revision"] = None
    del metadata["model_revision"]

    assert not is_complete(record, metadata, wav_path, expected, audio_config)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("seed", 42.0),
        ("channels", True),
        ("generation", {"nested": [1, {"enabled": 1}]}),
    ],
)
def test_complete_requires_exact_json_value_types(resume_case, field, replacement):
    record, metadata, wav_path, expected, audio_config = resume_case
    expected["generation"] = {"nested": [1, {"enabled": True}]}
    if field != "generation":
        metadata["generation"] = expected["generation"]
    metadata[field] = replacement

    assert not is_complete(record, metadata, wav_path, expected, audio_config)


def test_load_latest_metadata_keeps_last_record_and_ignores_partial_tail(tmp_path: Path):
    path = tmp_path / "metadata.jsonl"
    path.write_bytes(
        b'{"question_id":"q1","status":"failure"}\n'
        b'{"question_id":"q2","status":"success"}\n'
        b'{"question_id":"q1","status":"warning"}\n'
        b'{"question_id":"q3","status":"succ'
    )

    with pytest.warns(UserWarning, match="incomplete final JSONL record"):
        latest = load_latest_metadata(path)

    assert latest == {
        "q1": {"question_id": "q1", "status": "warning"},
        "q2": {"question_id": "q2", "status": "success"},
    }


def test_load_latest_metadata_returns_empty_for_absent_file(tmp_path: Path):
    assert load_latest_metadata(tmp_path / "missing.jsonl") == {}


@pytest.mark.parametrize(
    "payload",
    [
        b'{"question_id":"q1"}\nnot-json\n',
        b'{"question_id":"q1"}\n{"question_id": nope}\n',
        b'{"question_id":"q1"}\nnot-json',
    ],
)
def test_load_latest_metadata_rejects_corrupt_complete_or_garbage_tail(
    tmp_path: Path, payload: bytes
):
    path = tmp_path / "metadata.jsonl"
    path.write_bytes(payload)

    with pytest.raises(ManifestFormatError):
        load_latest_metadata(path)


def test_load_latest_metadata_requires_string_question_id(tmp_path: Path):
    path = tmp_path / "metadata.jsonl"
    path.write_text('{"status":"success"}\n', encoding="utf-8")

    with pytest.raises(ManifestFormatError, match="question_id"):
        load_latest_metadata(path)


def test_append_jsonl_fsync_writes_compact_unicode_record_and_syncs(tmp_path: Path, monkeypatch):
    path = tmp_path / "metadata.jsonl"
    calls: list[int] = []
    monkeypatch.setattr(os, "fsync", calls.append)
    record = {"question_id": "한글", "generation": {}, "status": "success"}

    append_jsonl_fsync(path, record)
    append_jsonl_fsync(path, {"question_id": "q2", "status": "warning"})

    assert path.read_bytes() == (
        b'{"question_id":"\xed\x95\x9c\xea\xb8\x80","generation":{},"status":"success"}\n'
        b'{"question_id":"q2","status":"warning"}\n'
    )
    assert len(calls) == 2


def test_complete_does_not_mutate_inputs(resume_case):
    record, metadata, wav_path, expected, audio_config = resume_case
    original_record = copy.deepcopy(record)
    original_metadata = copy.deepcopy(metadata)
    original_expected = copy.deepcopy(expected)

    is_complete(record, metadata, wav_path, expected, audio_config)

    assert record == original_record
    assert metadata == original_metadata
    assert expected == original_expected
