# Task 8 report: Synthesis transaction and mock end-to-end smoke test

## Delivered

- Added `synthesize_manifest(config, backend_factory) -> SynthesisSummary` with the
  required selection counters and one durable transaction per selected question.
- Each successful transaction now performs the required sequence: strict manifest
  validation, minimal question normalization and SHA-256, per-attempt seed reset,
  backend synthesis, atomic decoded WAV validation/publication, audio SHA-256, and
  fsync-backed metadata append.
- Success and warning metadata preserves the canonical source fields, including raw
  question, answer, aliases, and normalized aliases, while also recording exact
  `tts_text`, question/audio hashes, safe audio path, dataset/model revisions, model
  configuration, runtime environment, native waveform shape/rate, stored format, and
  all decoded audio quality metrics.
- Output WAV names use `safe_audio_stem` and a resolved parent check, preventing a
  manifest question ID from escaping the configured audio directory.
- Resume loads latest metadata, revalidates every existing WAV and identity invariant,
  and skips complete records without appending duplicate metadata. Overwrite/default
  modes continue to regenerate selected records.
- Ordinary synthesis/conversion/validation errors append a complete failure record and
  continue with the next sample. Failure-log and metadata-write errors remain fatal.
- CUDA OOM detection covers Torch exception types/chains and common CUDA OOM messages;
  it clears the CUDA cache, resets the seed, retries the same sample once, then records
  a failure and continues if the retry also fails.
- EIO and ENOSPC are detected through wrapped exception chains and re-raised. Config,
  manifest/schema, runtime metadata, and model-load validation occur outside the
  per-sample recovery boundary and are therefore fatal.
- Dry-run validates and selects the manifest but returns before constructing the
  backend or creating output directories.

## Mock coverage

- Added a deterministic 24 kHz sine-wave `MockBackend` fixture.
- The two-record end-to-end smoke test verifies safe WAV paths, mono PCM16 at 24 kHz,
  source-field preservation, normalized speech text, hashes, runtime/model identity,
  waveform details, every decoded audio metric, and the exact summary.
- A second resume invocation proves there are no backend synthesis calls and no new
  metadata bytes.
- A failing mock proves an ordinary first-sample error is logged while the following
  sample is synthesized and committed.
- Dedicated mock cases cover one CUDA OOM retry/cache clear and dry-run backend
  laziness.

## TDD evidence

- RED: `uv run pytest tests/test_smoke.py -v` failed during collection because
  `SynthesisSummary` and `synthesize_manifest` did not yet exist.
- GREEN: the same focused smoke command passed all 4 tests after the transaction was
  implemented.

The local `apply_patch` helper could not create its sandbox namespace in this runtime,
so the task brief's `git apply --recount` fallback was used for source/report changes.

## Verification

- `uv run pytest tests/test_smoke.py tests/test_resume.py tests/test_audio_validation.py -v`
  — 78 passed.
- `uv run ruff check .` — all checks passed.
- `uv run pytest -v` — 137 passed.
- `git diff --check` — passed before report creation; rerun at commit gate.
