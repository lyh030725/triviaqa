# TriviaQA 질문 TTS 데이터셋

TriviaQA validation 질문을 영어 음성으로 합성하는 재현 가능한 파이프라인이다.
질문과 정답 alias는 보존하지만 evidence 문서는 사용하지 않는다. 생성한 WAV는 Moshi의
closed-book 음성 QA 평가에도 사용할 수 있다.

## 구성

- Dataset: `mandarjoshi/trivia_qa`, `unfiltered.nocontext`, validation 11,313개
- TTS: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`, speaker `Aiden`
- Audio: 24kHz mono PCM 16-bit WAV
- Seed: 42
- Python 3.12, PyTorch/TorchAudio 2.11.0 cu128

실행 시 dataset과 model의 revision을 commit SHA로 고정하고 manifest와 생성 metadata에
기록한다.

## 설치

CUDA와 bf16을 지원하는 NVIDIA GPU가 필요하다. 시스템에 `ffmpeg`, `libsndfile`, `sox`,
`uv`를 설치한 뒤 다음을 실행한다.

```bash
uv venv --python 3.12
uv sync --locked --extra tts
```

Hugging Face 인증이 필요한 환경에서는 `hf auth login` 또는 `HF_TOKEN`을 사용한다.

## 데이터 생성

```bash
# validation manifest와 고정된 pilot 100개 준비
./scripts/prepare_dataset.sh

# pilot 합성 및 품질 리포트 생성
./scripts/synthesize_pilot.sh

# pilot 확인 후 전체 validation 합성
./scripts/synthesize_validation.sh
```

전체 합성은 명시적으로 `synthesize_validation.sh`를 실행할 때만 시작한다. 중단 후 같은
명령을 실행하면 정상 WAV와 metadata가 있는 항목을 건너뛴다.

범위와 shard는 CLI 옵션으로 나눌 수 있다.

```bash
.venv/bin/python -m triviaqa_tts.cli.synthesize_questions \
  --config configs/validation.yaml \
  --num-shards 4 \
  --shard-index 1 \
  --resume
```

선택 결과만 확인하려면 `--dry-run`, 기존 항목을 다시 만들려면 별도 실행에서
`--overwrite`를 사용한다.

## Moshi 평가

공개 Moshi checkpoint로 합성된 질문 500개의 exact match와 alias containment를 평가한다.
각 질문 앞에 5초 무음을 넣고 응답은 최대 20초 생성한다.

```bash
uv pip install --no-deps \
  'git+https://github.com/kyutai-labs/moshi.git@e6a55d2722a65870ef52a6c9f6ecfc0e90f38362#subdirectory=moshi'
uv pip install 'sentencepiece>=0.2,<0.3'
./scripts/evaluate_moshi.sh
```

평가는 `question_id` 기준으로 재개되며 결과는
`logs/moshi_eval_preroll_5s_max20s/`에 저장된다.

## 주요 산출물

```text
data/
  validation_manifest.jsonl
  pilot_100.jsonl
audio/qwen3_aiden_seed42/
  pilot/*.wav
  validation/*.wav
logs/
  pilot/
  validation/
  moshi_eval_preroll_5s_max20s/
    results.jsonl
    answers.json
    correct_answers.json
    summary.json
```

최종 Moshi 평가의 raw results, 전체 답변, 정답 응답, summary는 재현 가능한 평가 기록으로
Git에 함께 추적한다.

## 검증

```bash
make test
uv run ruff check .
uv run python -m triviaqa_tts.cli.synthesize_questions --help
```

GPU 테스트는 모델 다운로드와 VRAM을 사용하므로 별도로 실행한다.

```bash
RUN_QWEN_GPU_TEST=1 uv run pytest -m gpu -v
```

## 참고 자료

- [TriviaQA](https://nlp.cs.washington.edu/triviaqa/)
- [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
- [Moshi paper](https://arxiv.org/abs/2410.00037)
- [Official Moshi implementation](https://github.com/kyutai-labs/moshi)
