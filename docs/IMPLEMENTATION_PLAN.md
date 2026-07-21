# TriviaQA Question TTS Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible RunPod pipeline that prepares TriviaQA validation manifests, synthesizes questions with Qwen3-TTS 1.7B CustomVoice/Aiden, validates 24kHz mono PCM16 WAV files, and resumes safely.

**Architecture:** A deterministic data-preparation layer emits immutable JSONL manifests, then a single-sample orchestration loop selects ranges or disjoint shards and calls a minimal TTS backend. Each sample is written transactionally, decoded and validated, then represented by hash-bound append-only metadata; pilot reporting is derived from that metadata.

**Tech Stack:** Python 3.12, uv, Hugging Face datasets/huggingface_hub, qwen-tts 0.1.1, PyTorch/torchaudio CUDA wheels, NumPy, SoundFile, soxr, PyYAML, pytest.

## Global Constraints

- Dataset is exactly `mandarjoshi/trivia_qa`, configuration `unfiltered.nocontext`, split `validation`; do not consume evidence fields.
- Resolve and record immutable dataset/model commit SHAs; the observed SHAs are dataset `0f7faf33a3908546c6fd5b73a660e0f8ff173c2f` and model `0c0e3051f131929182e2c023b9537f8b1c68adfe`.
- Default model is exactly `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`; speaker `Aiden`; language `English`; seed `42`; dtype `bfloat16`.
- Pilot selection is size `100`, seed `20260721`, sampled randomly rather than taking a prefix.
- Stored audio is WAV, 24,000Hz, mono, PCM 16-bit; Qwen's observed native input/output sample rate is 24,000Hz.
- Every question resets Python, NumPy, PyTorch CPU, and PyTorch CUDA RNGs immediately before synthesis.
- Moshi, EPAD, voice-QA inference, ASR evaluation, answer-accuracy evaluation, CosyVoice, databases, and queues are prohibited.
- Default unit tests must not download Qwen weights or require a GPU; real-model tests use the `gpu` marker.
- Full validation synthesis begins only through an explicit `make validation` or equivalent CLI command.

## File and Interface Map

- `src/triviaqa_tts/config.py`: frozen configuration dataclasses and YAML/CLI validation.
- `src/triviaqa_tts/data/manifest.py`: record conversion, canonical JSONL, safe JSONL reads, hashing, question normalization, filenames.
- `src/triviaqa_tts/data/prepare.py`: Hub revision resolution, schema validation, validation manifest, deterministic pilot files.
- `src/triviaqa_tts/reproducibility.py`: RNG reset and runtime environment capture.
- `src/triviaqa_tts/tts/base.py`: `TTSBackend` protocol.
- `src/triviaqa_tts/tts/qwen3_backend.py`: official Qwen loader and CustomVoice call.
- `src/triviaqa_tts/audio/validation.py`: conversion, resampling, PCM16 atomic writes, decoded validation and quality metrics.
- `src/triviaqa_tts/tts/synthesize.py`: filters, sharding, resume decisions, synthesis transaction, metadata/failure append.
- `src/triviaqa_tts/cli/*.py`: stable CLI entry points and pilot report generation.
- `configs/*.yaml`, `scripts/*.sh`, `Makefile`: RunPod operational interface.
- `README.md`, `AGENTS.md`: user and contributor operating rules.

---

### Task 1: Packaging and Typed Configuration

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/triviaqa_tts/__init__.py`
- Create: `src/triviaqa_tts/config.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: YAML mappings and a config path.
- Produces: `load_config(path: Path) -> AppConfig`, `apply_cli_overrides(config: AppConfig, **values: object) -> AppConfig`, and frozen dataclasses `DatasetConfig`, `SelectionConfig`, `TTSConfig`, `AudioConfig`, `AudioValidationConfig`, `PathsConfig`, `AppConfig`.

- [ ] **Step 1: Add package metadata and a failing config test**

Use `requires-python = ">=3.12,<3.13"`, runtime dependencies `datasets`, `huggingface-hub`, `numpy`, `PyYAML`, `soundfile`, `soxr`, and optional `tts = ["qwen-tts==0.1.1"]`; add pytest/ruff dev dependencies and `gpu` marker. Test exact defaults and rejection of simultaneous `resume=True, overwrite=True`:

```python
def test_load_config_and_reject_conflicting_modes(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("dataset:\n  name: mandarjoshi/trivia_qa\n", encoding="utf-8")
    config = load_config(path)
    assert config.tts.model_id == "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    assert config.audio.sample_rate == 24_000
    with pytest.raises(ValueError, match="mutually exclusive"):
        apply_cli_overrides(config, resume=True, overwrite=True)
```

- [ ] **Step 2: Verify the test fails**

Run: `uv run pytest tests/test_config.py -v`

Expected: FAIL because `triviaqa_tts.config` does not exist.

- [ ] **Step 3: Implement strict dataclass parsing**

Implement recursive mapping-to-dataclass construction without silently accepting unknown keys. Resolve relative paths against the repository working directory, keep `/workspace/huggingface_cache` absolute, and validate pilot size/seed, shard bounds, positive rates/durations, `PCM_16`, and the fixed dataset scope.

```python
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
```

- [ ] **Step 4: Run focused tests and lint**

Run: `uv run pytest tests/test_config.py -v && uv run ruff check src/triviaqa_tts/config.py tests/test_config.py`

Expected: all config tests PASS and Ruff exits 0.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore src/triviaqa_tts tests/test_config.py
git commit -m "build: add package and typed configuration"
```

### Task 2: Deterministic Manifest Primitives

**Files:**
- Create: `src/triviaqa_tts/data/__init__.py`
- Create: `src/triviaqa_tts/data/manifest.py`
- Create: `tests/test_manifest.py`

**Interfaces:**
- Consumes: TriviaQA-like `Mapping[str, object]`, original index, scope strings, and resolved revision.
- Produces: `example_to_record(...) -> dict[str, object]`, `write_canonical_jsonl(records, path) -> str`, `iter_jsonl(path, allow_partial_last_line=True)`, `normalize_question(text) -> str`, `safe_audio_stem(question_id) -> str`, `sha256_file(path) -> str`.

- [ ] **Step 1: Write failing tests for schema preservation and deterministic bytes**

```python
def test_example_to_record_preserves_answer_lists():
    row = {"question_id": "tc/2", "question": "  Who  &amp; why ",
           "answer": {"value": "Answer", "aliases": ["A", "Answer"],
                      "normalized_aliases": ["a", "answer"]}}
    record = example_to_record(row, original_index=7, dataset_revision="abc")
    assert record["question"] == "  Who  &amp; why "
    assert record["aliases"] == ["A", "Answer"]
    assert record["normalized_aliases"] == ["a", "answer"]

def test_normalize_question_is_minimal():
    assert normalize_question("  Who  &amp; why ") == "Who & why?"
```

Also test identical JSONL SHA across writes, UTF-8 LF output, hashed safe filenames, and ignoring only an invalid non-newline-terminated final record.

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/test_manifest.py -v`

Expected: FAIL because manifest functions are undefined.

- [ ] **Step 3: Implement canonical serialization and strict partial-line logic**

Serialize one compact JSON object per line with insertion-ordered keys and `ensure_ascii=False`. Raise `ManifestFormatError` for corrupt complete/middle lines; warn and ignore only the last incomplete line. Use `html.unescape`, `" ".join(text.split())`, and append `?` only when the final character is not in `.!?;:`. Build stems as `<sanitized-prefix>_<first-12-question-id-sha256>`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_manifest.py -v`

Expected: all manifest tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/triviaqa_tts/data tests/test_manifest.py
git commit -m "feat: add deterministic manifest primitives"
```

### Task 3: TriviaQA Preparation and Pilot Sampling

**Files:**
- Create: `src/triviaqa_tts/data/prepare.py`
- Create: `src/triviaqa_tts/cli/__init__.py`
- Create: `src/triviaqa_tts/cli/prepare_dataset.py`
- Create: `tests/test_prepare.py`
- Create: `tests/test_selection.py`

**Interfaces:**
- Consumes: `AppConfig`, an injectable dataset loader, and injectable Hub API.
- Produces: `resolve_dataset_revision(...) -> str`, `validate_triviaqa_schema(dataset) -> None`, `sample_pilot_indices(count, size, seed) -> list[int]`, `prepare_dataset(config) -> PreparationResult`, CLI `main(argv=None) -> int`.

- [ ] **Step 1: Write failing schema and selection tests**

```python
def test_pilot_sampling_is_reproducible_and_not_prefix():
    first = sample_pilot_indices(11_313, 100, 20260721)
    second = sample_pilot_indices(11_313, 100, 20260721)
    assert first == second == sorted(first)
    assert first != list(range(100))
    assert len(set(first)) == 100
```

Use fake dataset `features` matching the observed nested answer structure and assert missing `answer.normalized_aliases` fails before writing files.

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/test_prepare.py tests/test_selection.py -v`

Expected: FAIL because preparation functions do not exist.

- [ ] **Step 3: Implement revision pinning, preparation, and atomic sidecars**

Resolve `main` or configured revision through `HfApi.dataset_info(...).sha`, pass the SHA to `load_dataset`, validate features and runtime values, write all 11,313 records, compute duplicate/empty counts, and atomically write `validation_manifest.meta.json`. Use `sorted(random.Random(seed).sample(range(count), size))` and copy selected canonical records to pilot JSONL and IDs text.

- [ ] **Step 4: Add CLI argument parsing and run tests**

Run: `uv run pytest tests/test_prepare.py tests/test_selection.py tests/test_manifest.py -v`

Expected: all tests PASS with no network calls.

- [ ] **Step 5: Commit**

```bash
git add src/triviaqa_tts/data/prepare.py src/triviaqa_tts/cli tests/test_prepare.py tests/test_selection.py
git commit -m "feat: prepare TriviaQA manifests and pilot sample"
```

### Task 4: Reproducibility and Backend Contract

**Files:**
- Create: `src/triviaqa_tts/reproducibility.py`
- Create: `src/triviaqa_tts/tts/__init__.py`
- Create: `src/triviaqa_tts/tts/base.py`
- Create: `tests/test_reproducibility.py`

**Interfaces:**
- Produces: requested `TTSBackend` protocol, `set_seed(seed: int) -> None`, `collect_runtime_environment() -> dict[str, object]`, `package_version(name: str) -> str | None`.

- [ ] **Step 1: Write failing deterministic RNG tests**

Monkeypatch a fake torch module to assert calls to `manual_seed`, `cuda.manual_seed_all`, and cuDNN flags, and verify Python/NumPy sequences repeat after `set_seed(42)`.

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/test_reproducibility.py -v`

Expected: FAIL because reproducibility module is absent.

- [ ] **Step 3: Implement lazy PyTorch support and the exact protocol**

```python
class TTSBackend(Protocol):
    @property
    def model_id(self) -> str: ...
    @property
    def native_sample_rate(self) -> int | None: ...
    def synthesize(self, text: str, seed: int) -> tuple[np.ndarray, int]: ...
```

Do not make importing the package require torch. Collect Python/platform/package versions and, when torch exists, CUDA version, availability, GPU name, and bf16 support.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_reproducibility.py -v`

Expected: PASS in a CPU-only environment.

- [ ] **Step 5: Commit**

```bash
git add src/triviaqa_tts/reproducibility.py src/triviaqa_tts/tts tests/test_reproducibility.py
git commit -m "feat: add reproducibility controls and TTS protocol"
```

### Task 5: Audio Conversion, Atomic Storage, and Validation

**Files:**
- Create: `src/triviaqa_tts/audio/__init__.py`
- Create: `src/triviaqa_tts/audio/validation.py`
- Create: `tests/test_audio_validation.py`

**Interfaces:**
- Consumes: NumPy waveform, native rate, target `AudioConfig`, and output path.
- Produces: `prepare_waveform(...) -> np.ndarray`, `write_validated_wav(...) -> AudioMetrics`, `validate_wav(path, config) -> AudioMetrics`, dataclass `AudioMetrics`, exception `AudioValidationError`.

- [ ] **Step 1: Write failing tests with generated waveforms**

Cover mono PCM16 24kHz round-trip, stereo averaging, 16kHz-to-24kHz resampling, empty data, NaN, Inf, clipping, short warning, leading/trailing silence, zero ratio, and no final file when validation fails.

```python
def test_nan_is_rejected(tmp_path, audio_config):
    waveform = np.array([0.0, np.nan, 0.1], dtype=np.float32)
    with pytest.raises(AudioValidationError, match="NaN"):
        write_validated_wav(waveform, 24_000, tmp_path / "bad.wav", audio_config)
    assert not (tmp_path / "bad.wav").exists()
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/test_audio_validation.py -v`

Expected: FAIL because audio validation module is absent.

- [ ] **Step 3: Implement conversion and same-directory temporary writes**

Reject non-finite values before SoundFile conversion; normalize shape without amplitude normalization; resample only when rates differ using `soxr.resample(..., quality="HQ")`; write `format="WAV", subtype="PCM_16"`; decode the temp file; compute metrics from decoded float32; `os.replace` only after required validation passes. Quality threshold breaches populate warning reasons and do not fail storage.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_audio_validation.py -v`

Expected: all audio tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/triviaqa_tts/audio tests/test_audio_validation.py
git commit -m "feat: validate and atomically store WAV audio"
```

### Task 6: Official Qwen3-TTS Backend

**Files:**
- Create: `src/triviaqa_tts/tts/qwen3_backend.py`
- Create: `tests/test_qwen3_backend.py`

**Interfaces:**
- Consumes: `TTSConfig`, resolved model revision, optional injected Qwen model class.
- Produces: `Qwen3Backend.from_config(config) -> Qwen3Backend`, properties required by `TTSBackend`, and `synthesize(text, seed)` using official `generate_custom_voice`.

- [ ] **Step 1: Write failing mocked API tests**

Assert loader receives model ID, SHA revision, `device_map="cuda:0"`, bf16 dtype, and resolved attention; assert the model's supported speakers/languages contain Aiden/English; assert generation receives exact text, `language="English"`, `speaker="Aiden"`, `instruct=""`; assert one waveform and returned sample rate are unwrapped.

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/test_qwen3_backend.py -v`

Expected: FAIL because Qwen backend is absent.

- [ ] **Step 3: Implement lazy imports and compatibility checks**

Import torch and `Qwen3TTSModel` only inside loader code. Resolve model SHA with `HfApi.model_info`. Choose `flash_attention_2` only when importable and requested/auto, otherwise `sdpa`; do not silently downgrade model size, dtype, speaker, or language. Validate returned list length, finite NumPy-compatible waveform, positive integer sample rate, and expose observed native rate after first call.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_qwen3_backend.py tests/test_reproducibility.py -v`

Expected: PASS without installing qwen-tts or downloading weights because dependencies are injected/mocked.

- [ ] **Step 5: Commit**

```bash
git add src/triviaqa_tts/tts/qwen3_backend.py tests/test_qwen3_backend.py
git commit -m "feat: integrate official Qwen3-TTS CustomVoice API"
```

### Task 7: Deterministic Selection and Strict Resume

**Files:**
- Create: `src/triviaqa_tts/tts/synthesize.py`
- Create: `tests/test_resume.py`
- Extend: `tests/test_selection.py`

**Interfaces:**
- Produces: `select_records(records, start_index, end_index, shard_index, num_shards, limit)`, `load_latest_metadata(path)`, `is_complete(record, metadata, wav_path, expected, audio_config) -> bool`, `append_jsonl_fsync(path, record) -> None`.

- [ ] **Step 1: Write failing shard and resume matrix tests**

Assert shard sets are pairwise disjoint and their union equals the selected range. Parametrize resume invalidation for absent WAV, missing metadata, failure status, audio hash mismatch, question hash mismatch, invalid WAV, model/revision/speaker/language/seed/generation mismatch, and accept only a fully matching success or warning record.

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/test_resume.py tests/test_selection.py -v`

Expected: FAIL because selection/resume functions do not exist.

- [ ] **Step 3: Implement exact filtering order and resume validation**

Apply inclusive start/exclusive end, global ordinal modulo sharding, then limit. Read metadata with strict partial-last-line handling and retain the latest valid record per question ID. Recompute WAV SHA and call `validate_wav`; compare every identity/config field listed in the spec. Append compact JSON plus LF in one write, flush, and `os.fsync`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_resume.py tests/test_selection.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/triviaqa_tts/tts/synthesize.py tests/test_resume.py tests/test_selection.py
git commit -m "feat: add deterministic sharding and strict resume"
```

### Task 8: Synthesis Transaction and Mock End-to-End Smoke Test

**Files:**
- Extend: `src/triviaqa_tts/tts/synthesize.py`
- Create: `tests/conftest.py`
- Create: `tests/test_smoke.py`

**Interfaces:**
- Consumes: manifest records, `AppConfig`, and a `TTSBackend` factory.
- Produces: `synthesize_manifest(config, backend_factory) -> SynthesisSummary` and complete metadata/failure JSONL records.

- [ ] **Step 1: Write a failing mock pipeline test**

Define `MockBackend.synthesize` returning a short deterministic sine wave. Process two records and assert safe WAV paths, PCM16/24kHz/mono, answer and aliases, raw question and normalized `tts_text`, question/audio hashes, environment/model fields, metrics, and successful second `resume` creating no new metadata. Add a backend that fails one sample and prove the next sample succeeds.

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/test_smoke.py -v`

Expected: FAIL because orchestration is incomplete.

- [ ] **Step 3: Implement sample transactions and failure policy**

For each selected record: normalize/hash, reset seed, synthesize, atomically store/validate, hash, append success or warning metadata. Record ordinary exceptions in failures and continue. Identify CUDA OOM by torch exception/type/message, clear CUDA cache, retry once, then record failure. Re-raise config/schema/model-load, ENOSPC, EIO, and metadata-write errors as fatal. Dry-run must never construct the backend.

- [ ] **Step 4: Run smoke and regression tests**

Run: `uv run pytest tests/test_smoke.py tests/test_resume.py tests/test_audio_validation.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/triviaqa_tts/tts/synthesize.py tests/conftest.py tests/test_smoke.py
git commit -m "feat: synthesize questions transactionally"
```

### Task 9: Synthesis CLI and Pilot Reports

**Files:**
- Create: `src/triviaqa_tts/cli/synthesize_questions.py`
- Create: `src/triviaqa_tts/cli/report_pilot.py`
- Create: `tests/test_cli.py`
- Create: `tests/test_report.py`

**Interfaces:**
- Produces: synthesis `main(argv=None) -> int`, `build_pilot_report(metadata, failures, sample_size=10, seed=20260721) -> dict`, report `main(argv=None) -> int`.

- [ ] **Step 1: Write failing CLI/report tests**

Assert every requested option parses and invalid shard/resume-overwrite combinations exit non-zero. Feed success, warning, clipping, short, and failure fixtures to the report and assert exact counts, duration aggregates, and deterministic ten-item listening sample (or all items when fewer than ten).

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/test_cli.py tests/test_report.py -v`

Expected: FAIL because CLI/report modules are absent.

- [ ] **Step 3: Implement CLIs and JSON/CSV outputs**

Use argparse with `--config` required and the eight requested overrides. Print selected/processed/skipped/success/warning/failure totals. Write report JSON atomically and one CSV row per audio item with status and metrics; print ten paths and questions for listening. Return non-zero on fatal errors but not for isolated sample failures after producing a summary.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_cli.py tests/test_report.py tests/test_smoke.py -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/triviaqa_tts/cli tests/test_cli.py tests/test_report.py
git commit -m "feat: add synthesis CLI and pilot quality reports"
```

### Task 10: Configurations, RunPod Bootstrap, Scripts, and Makefile

**Files:**
- Create: `configs/pilot.yaml`
- Create: `configs/validation.yaml`
- Create: `scripts/bootstrap_runpod.sh`
- Create: `scripts/prepare_dataset.sh`
- Create: `scripts/synthesize_pilot.sh`
- Create: `scripts/synthesize_validation.sh`
- Create: `Makefile`
- Create: `tests/test_operational_files.py`

**Interfaces:**
- Produces the exact `make setup`, `make prepare`, `make pilot`, `make test`, `make validation` operator interface.

- [ ] **Step 1: Write failing operational file tests**

Parse both YAML files and assert shared model/speaker/seed/audio settings, different selection mode/manifest/output paths, cache `/workspace/huggingface_cache`, and executable shell scripts containing `set -Eeuo pipefail`. Parse Makefile text to assert validation is not a dependency of setup/prepare/test/pilot.

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/test_operational_files.py -v`

Expected: FAIL because operational files are absent.

- [ ] **Step 3: Add exact configs and fail-fast scripts**

Use output base `audio/qwen3_aiden_seed42`; pilot manifest `data/pilot_100.jsonl`; validation manifest `data/validation_manifest.jsonl`. Bootstrap installs system packages, uv/Python 3.12, locked project plus TTS extra, exports HF cache variables via `.env.runpod`, checks CUDA/GPU/bf16 and versions, and attempts optional FlashAttention with `MAX_JOBS=4` while preserving SDPA fallback. Wrapper scripts `cd` to repository root and use `.venv/bin/python -m ...`.

- [ ] **Step 4: Make scripts executable and test syntax**

Run: `chmod +x scripts/*.sh && bash -n scripts/*.sh && uv run pytest tests/test_operational_files.py -v && make test`

Expected: shell syntax and tests PASS; no dataset/model download occurs.

- [ ] **Step 5: Generate and verify lockfile**

Run: `uv lock && uv sync --extra dev && uv run python -c "import numpy, soundfile, yaml; print('runtime imports ok')"`

Expected: lock succeeds and prints `runtime imports ok`.

- [ ] **Step 6: Commit**

```bash
git add configs scripts Makefile tests/test_operational_files.py uv.lock
git commit -m "ops: add RunPod setup and execution commands"
```

### Task 11: Documentation, GPU Marker, and Full Verification

**Files:**
- Create: `README.md`
- Create: `AGENTS.md`
- Create: `tests/test_gpu_integration.py`
- Modify: `docs/PROJECT_SPEC.md`
- Modify: `docs/IMPLEMENTATION_PLAN.md`

**Interfaces:**
- Produces complete RunPod operator documentation and an opt-in real-model check.

- [ ] **Step 1: Add an opt-in GPU integration test**

Mark with `@pytest.mark.gpu`; skip unless CUDA is available and `RUN_QWEN_GPU_TEST=1`; instantiate the configured backend, assert Aiden/English validation, synthesize one short question, assert returned 24,000Hz and non-empty finite waveform, then pass it through validated WAV storage.

- [ ] **Step 2: Write README and contributor rules**

Document project purpose/scope, why `unfiltered.nocontext`, actual schema/count/revisions, uv rationale, RunPod GPU recommendation, storage estimation formula, HF authentication, install/prepare/pilot/report/validation/sharding/resume commands, output tree, metadata fields, OOM/failure response, backup commands before Pod termination, SDPA fallback, and explicit excluded features. `AGENTS.md` requires TDD, CPU default tests, no evidence or QA evaluation, no automatic validation run, and documentation/config synchronization for model/speaker changes.

- [ ] **Step 3: Run static and CPU verification**

Run: `uv run ruff check . && uv run pytest -m "not gpu" -v && make test && git diff --check`

Expected: Ruff exits 0, all CPU tests including mock end-to-end PASS, Make exits 0, diff check exits 0.

- [ ] **Step 4: Verify CLI help and dry-run**

Run: `uv run python -m triviaqa_tts.cli.prepare_dataset --help && uv run python -m triviaqa_tts.cli.synthesize_questions --help && uv run python -m triviaqa_tts.cli.report_pilot --help`

Expected: each command exits 0 and lists documented options. After a local fixture manifest is supplied by the smoke test, dry-run must exit 0 without importing qwen-tts.

- [ ] **Step 5: Record GPU test status accurately**

On CPU-only development environment run `uv run pytest -m gpu -v`; expected result is SKIPPED with the CUDA/opt-in reason. On RunPod run `RUN_QWEN_GPU_TEST=1 uv run pytest -m gpu -v`; expected result is one real-model PASS or a reported environment failure that is not represented as a pass.

- [ ] **Step 6: Update plan checkboxes and commit**

Mark only actually completed checklist items, then:

```bash
git add README.md AGENTS.md tests/test_gpu_integration.py docs
git commit -m "docs: complete RunPod operating guide"
```

## Final Acceptance Run

- [ ] Run `uv run pytest -m "not gpu" -v` and record exact passed/skipped counts.
- [ ] Run `uv run ruff check .` and `git diff --check`.
- [ ] Run `git status --short --branch` and confirm only intentional changes remain.
- [ ] Confirm `rg -n "Moshi|EPAD|CosyVoice|ASR|answer accuracy" src tests` finds no implementation, only explicit prohibition tests/docs if present.
- [ ] Report created files, dependency strategy, actual TriviaQA schema, model ID/speaker, commands/results, first RunPod command, pilot/validation commands, GPU test status, and remaining environment uncertainty.
