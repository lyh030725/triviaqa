# Contributor Rules

- 기능이나 버그 수정은 먼저 실패하는 테스트를 추가하고, 최소 구현 후 관련 테스트와 전체 CPU
  suite를 실행한다.
- 기본 검증은 network/GPU가 필요 없는 `uv run pytest -m "not gpu"`이다. 실제 모델 테스트는
  `RUN_QWEN_GPU_TEST=1`과 CUDA가 모두 있을 때만 실행한다.
- TriviaQA evidence 처리, 음성 QA 추론, ASR, Moshi/EPAD, answer-accuracy 평가는 추가하지 않는다.
- setup, prepare, test, pilot에서 전체 validation 합성을 자동 시작하지 않는다. 사용자의 명시적
  validation 명령만 허용한다.
- model ID/revision, speaker, language, seed, sample rate 또는 CUDA dependency를 변경하면
  `configs/*.yaml`, `pyproject.toml`/`uv.lock`, README와 설계 문서를 함께 갱신한다.
- 출력은 atomic WAV와 append-only JSONL의 resume 불변조건을 유지하고, 기존 산출물을 무단으로
  삭제하거나 덮어쓰지 않는다.
