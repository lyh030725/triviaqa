from __future__ import annotations

import errno
import hashlib
from dataclasses import asdict, replace
from pathlib import Path

import soundfile as sf
import pytest

from triviaqa_tts.config import AppConfig
from triviaqa_tts.data.manifest import iter_jsonl, safe_audio_stem, sha256_file
from triviaqa_tts.tts.synthesize import SynthesisSummary, synthesize_manifest

from conftest import MockBackend


def test_mock_pipeline_writes_complete_metadata_and_resume_is_idempotent(
    synthesis_config: AppConfig,
    manifest_records: list[dict[str, object]],
) -> None:
    backend = MockBackend()

    summary = synthesize_manifest(synthesis_config, lambda config: backend)

    assert summary == SynthesisSummary(
        selected=2, processed=2, skipped=0, success=2, warning=0, failure=0
    )
    assert backend.calls == [("Who and why?", 42), ("Already complete!", 42)]
    metadata_path = synthesis_config.paths.logs_dir / "metadata.jsonl"
    metadata = list(iter_jsonl(metadata_path, allow_partial_last_line=False))
    assert len(metadata) == 2

    for source, item in zip(manifest_records, metadata, strict=True):
        wav_path = (
            synthesis_config.paths.audio_dir / f"{safe_audio_stem(source['question_id'])}.wav"
        )
        assert Path(item["audio_path"]) == wav_path
        assert wav_path.parent == synthesis_config.paths.audio_dir
        assert wav_path.is_file()
        info = sf.info(wav_path)
        assert (info.format, info.subtype, info.samplerate, info.channels) == (
            "WAV",
            "PCM_16",
            24_000,
            1,
        )
        assert item["question_id"] == source["question_id"]
        assert item["question"] == source["question"]
        assert item["answer_value"] == source["answer_value"]
        assert item["aliases"] == source["aliases"]
        assert item["normalized_aliases"] == source["normalized_aliases"]
        assert (
            item["question_sha256"] == hashlib.sha256(item["tts_text"].encode("utf-8")).hexdigest()
        )
        assert item["audio_sha256"] == sha256_file(wav_path)
        assert item["dataset_revision"] == "a" * 40
        assert item["model_id"] == backend.model_id
        assert item["model_revision"] == backend.model_revision
        assert item["speaker"] == "Aiden"
        assert item["language"] == "English"
        assert item["seed"] == 42
        assert item["dtype"] == synthesis_config.tts.dtype
        assert item["attention_implementation"] == "mock"
        assert item["generation"] == {}
        assert item["environment"]["python_version"]
        assert item["native_sample_rate"] == 24_000
        assert item["waveform_shape"] == [9_600]
        assert item["sample_rate"] == 24_000
        assert item["channels"] == 1
        assert item["subtype"] == "PCM_16"
        assert item["sample_count"] == 9_600
        assert item["duration_seconds"] == 0.4
        assert item["peak_amplitude"] > 0
        assert item["clipped_sample_count"] == 0
        assert item["zero_ratio"] >= 0
        assert item["leading_silence_seconds"] >= 0
        assert item["trailing_silence_seconds"] >= 0
        assert item["warning_reasons"] == []
        assert item["status"] == "success"

    before = metadata_path.read_bytes()
    second_backend = MockBackend()
    resume_config = replace(
        synthesis_config,
        selection=replace(synthesis_config.selection, resume=True),
    )
    resumed = synthesize_manifest(resume_config, lambda config: second_backend)

    assert resumed == SynthesisSummary(
        selected=2, processed=0, skipped=2, success=0, warning=0, failure=0
    )
    assert second_backend.calls == []
    assert metadata_path.read_bytes() == before


def test_resume_regenerates_when_resolved_attention_changes(
    synthesis_config: AppConfig,
) -> None:
    synthesize_manifest(synthesis_config, lambda config: MockBackend())

    backend = MockBackend()
    backend.attention_implementation = "different-resolved-attention"
    resume_config = replace(
        synthesis_config,
        selection=replace(synthesis_config.selection, resume=True),
    )

    summary = synthesize_manifest(resume_config, lambda config: backend)

    assert summary.processed == 2
    assert summary.skipped == 0
    assert len(backend.calls) == 2


def test_resume_regenerates_when_configured_dtype_changes(
    synthesis_config: AppConfig,
) -> None:
    synthesize_manifest(synthesis_config, lambda config: MockBackend())

    backend = MockBackend()
    resume_config = replace(
        synthesis_config,
        selection=replace(synthesis_config.selection, resume=True),
        tts=replace(synthesis_config.tts, dtype="float16"),
    )

    summary = synthesize_manifest(resume_config, lambda config: backend)

    assert summary.processed == 2
    assert summary.skipped == 0
    assert len(backend.calls) == 2


@pytest.mark.parametrize(
    ("validator_name", "fatal_errno"),
    [("sha256_file", errno.EIO), ("validate_wav", errno.ENOSPC)],
)
def test_resume_aborts_on_fatal_os_error_before_backend_generation(
    synthesis_config: AppConfig,
    monkeypatch: pytest.MonkeyPatch,
    validator_name: str,
    fatal_errno: int,
) -> None:
    synthesize_manifest(synthesis_config, lambda config: MockBackend())
    backend = MockBackend()
    resume_config = replace(
        synthesis_config,
        selection=replace(synthesis_config.selection, resume=True),
    )

    def raise_fatal(*args, **kwargs):
        raise OSError(fatal_errno, "fatal resume validation")

    monkeypatch.setattr(f"triviaqa_tts.tts.synthesize.{validator_name}", raise_fatal)

    with pytest.raises(OSError) as raised:
        synthesize_manifest(resume_config, lambda config: backend)

    assert raised.value.errno == fatal_errno
    assert backend.calls == []


def test_sample_failure_is_appended_and_next_sample_succeeds(
    synthesis_config: AppConfig,
) -> None:
    class FailFirstBackend(MockBackend):
        def synthesize(self, text: str, seed: int):
            if text == "Who and why?":
                self.calls.append((text, seed))
                raise RuntimeError("intentional sample failure")
            return super().synthesize(text, seed)

    backend = FailFirstBackend()

    summary = synthesize_manifest(synthesis_config, lambda config: backend)

    assert summary == SynthesisSummary(
        selected=2, processed=2, skipped=0, success=1, warning=0, failure=1
    )
    failures = list(
        iter_jsonl(
            synthesis_config.paths.logs_dir / "failures.jsonl",
            allow_partial_last_line=False,
        )
    )
    assert len(failures) == 1
    assert failures[0]["question_id"] == "safe/../one"
    assert failures[0]["tts_text"] == "Who and why?"
    assert failures[0]["error_type"] == "RuntimeError"
    assert failures[0]["error_message"] == "intentional sample failure"
    assert failures[0]["status"] == "failure"
    metadata = list(
        iter_jsonl(
            synthesis_config.paths.logs_dir / "metadata.jsonl",
            allow_partial_last_line=False,
        )
    )
    assert [item["question_id"] for item in metadata] == ["two"]
    assert (synthesis_config.paths.audio_dir / f"{safe_audio_stem('two')}.wav").is_file()


def test_cuda_oom_clears_cache_and_retries_once(synthesis_config: AppConfig, monkeypatch) -> None:
    class OomBackend(MockBackend):
        def __init__(self) -> None:
            super().__init__()
            self.attempts: dict[str, int] = {}

        def synthesize(self, text: str, seed: int):
            self.calls.append((text, seed))
            self.attempts[text] = self.attempts.get(text, 0) + 1
            if text == "Who and why?":
                raise RuntimeError("CUDA out of memory")
            return MockBackend.synthesize(self, text, seed)

    cleared: list[None] = []
    monkeypatch.setattr(
        "triviaqa_tts.tts.synthesize._clear_cuda_cache", lambda: cleared.append(None)
    )
    backend = OomBackend()

    summary = synthesize_manifest(synthesis_config, lambda config: backend)

    assert summary.failure == 1
    assert summary.success == 1
    assert backend.attempts["Who and why?"] == 2
    assert len(cleared) == 1


def test_dry_run_never_constructs_backend(synthesis_config: AppConfig) -> None:
    dry_run_config = replace(
        synthesis_config,
        selection=replace(synthesis_config.selection, dry_run=True),
    )

    def forbidden_factory(config):
        raise AssertionError("backend factory must not be called")

    summary = synthesize_manifest(dry_run_config, forbidden_factory)

    assert asdict(summary) == {
        "selected": 2,
        "processed": 0,
        "skipped": 2,
        "success": 0,
        "warning": 0,
        "failure": 0,
    }
    assert not synthesis_config.paths.audio_dir.exists()
    assert not synthesis_config.paths.logs_dir.exists()
