# TriviaQA 질문 TTS 데이터셋

TriviaQA validation 질문을 `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`의 영어 화자
`Aiden`으로 합성하는 재현 가능한 RunPod 파이프라인이다. 질문과 정답 alias는 보존하지만 evidence
문서는 읽거나 복사하지 않는다. 음성 QA, ASR, Moshi/EPAD, 정답 정확도 평가는 범위 밖이다.

## 고정 입력과 환경

- Dataset: `mandarjoshi/trivia_qa`, `unfiltered.nocontext`, `validation` 11,313개
- 확인한 dataset revision: `0f7faf33a3908546c6fd5b73a660e0f8ff173c2f`
- 사용 schema: `question`, `question_id`, `answer.value`, `answer.aliases`,
  `answer.normalized_aliases`
- 확인한 model revision: `0c0e3051f131929182e2c023b9537f8b1c68adfe`
- 출력: 24,000Hz, mono, PCM 16-bit WAV; seed 42
- Python 3.12, `qwen-tts==0.1.1`, PyTorch/TorchAudio 2.11.0 cu128

`unfiltered.nocontext`는 질문 TTS에 불필요한 Wikipedia/Web evidence를 내려받고 처리하는 범위를
최소화한다. 실행 때 Hub의 revision을 immutable SHA로 resolve하여 manifest와 생성 metadata에
기록한다. `uv.lock`은 Python/CUDA 의존성을 동일하게 재설치하기 위해 사용한다.

bf16을 지원하는 Ampere/Ada/Hopper 계열 NVIDIA GPU를 권장한다. 필요한 저장 공간은 대략
`질문 수 × 평균 초 × 24,000 samples/s × 2 bytes`이며 WAV header, 모델/cache, 임시 파일과
여유 공간을 별도로 잡는다. 예를 들어 평균 8초인 11,313개 PCM WAV는 약 4.35GB이고 모델 및
Hub cache 공간은 여기에 포함되지 않는다.

## RunPod에서 실행

저장 가능한 `/workspace` 아래에서 저장소를 받은 뒤 첫 명령은 다음과 같다.

```bash
cd /workspace/triviaqa-tts
./scripts/bootstrap_runpod.sh
source .env.runpod
```

비공개/제한 저장소 인증이 필요한 환경에서는 먼저 `hf auth login` 또는 `HF_TOKEN`을 설정한다.
bootstrap은 잠긴 cu128 환경과 시스템 패키지를 설치하고 CUDA/bf16을 검사한다. FlashAttention
설치가 실패해도 `attention: auto`는 SDPA로 안전하게 전환된다. 빌드를 생략하려면
`INSTALL_FLASH_ATTN=0 ./scripts/bootstrap_runpod.sh`를 사용한다.

Dataset manifest와 고정 seed pilot 100개를 만든다.

```bash
./scripts/prepare_dataset.sh
./scripts/synthesize_pilot.sh
```

두 번째 명령은 pilot 합성 후 품질 JSON/CSV와 고정 10개 청취 목록도 만든다. 청취 결과와
`logs/pilot/pilot_audio_report.json`의 warning/failure를 확인한 뒤에만 전체 validation을
명시적으로 시작한다.

```bash
./scripts/synthesize_validation.sh
```

설치, prepare, test 또는 pilot은 전체 validation 합성을 자동으로 실행하지 않는다.

## 선택, 분할, 재개

합성 CLI는 inclusive `--start-index`, exclusive `--end-index`를 먼저 적용하고, 전역 ordinal의
modulo로 shard를 고른 뒤 `--limit`을 적용한다. 예를 들어 네 Pod의 두 번째 shard는 다음과 같다.

```bash
.venv/bin/python -m triviaqa_tts.cli.synthesize_questions \
  --config configs/validation.yaml --num-shards 4 --shard-index 1 --resume
```

실행 전 선택만 확인하려면 `--dry-run`을 사용한다. 정상 WAV와 metadata hash/config가 모두 맞는
항목만 `--resume`이 건너뛴다. 재생성이 필요하면 별도 실행에서 `--overwrite`를 사용하며 두 옵션은
동시에 쓸 수 없다. 각 shard는 서로 다른 `paths.audio_dir`와 `paths.logs_dir`를 가진 config 사본을
사용한 뒤 산출물을 합치는 방식을 권장한다.

## 산출물

```text
data/
  validation_manifest.jsonl
  validation_manifest.meta.json
  pilot_100.jsonl
  pilot_question_ids.txt
audio/qwen3_aiden_seed42/
  pilot/*.wav
  validation/*.wav
logs/
  pilot/{metadata.jsonl,failures.jsonl,pilot_audio_report.json,pilot_audio_report.csv}
  validation/{metadata.jsonl,failures.jsonl}
```

Manifest에는 dataset/config/split/revision, 원 index, question ID/text, answer value와 alias들이 있다.
성공 metadata에는 여기에 실제 TTS text/hash, model revision, speaker/language/seed/dtype/attention,
환경 버전, WAV 경로/hash, native/final sample rate, channel/subtype, duration, clipping/zero/silence
지표와 warning이 추가된다. 실패 로그에는 예외 유형/메시지와 CUDA OOM 재시도 여부가 기록된다.

일반 샘플 실패는 기록하고 계속하며 CUDA OOM은 cache를 비운 뒤 한 번 재시도한다. 반복 OOM이면
질문 범위나 shard를 줄이고 더 큰 GPU를 사용한다. 모델 크기를 자동 변경하지 않는다. disk full,
manifest/schema, 모델 load, metadata write 오류는 전체 실행을 중단한다. FlashAttention 문제는 config의
`tts.attention: sdpa`로 고정해 재시도할 수 있다.

Pod를 종료하기 전 durable volume이나 원격 저장소로 최소한 `data`, `audio`, `logs`, config를
복사한다. 예:

```bash
tar -C /workspace/triviaqa-tts -czf /workspace/triviaqa-tts-results.tgz data audio logs configs
# 또는 설정된 원격 저장소로
rclone copy /workspace/triviaqa-tts/ remote:triviaqa-tts/ --include '/data/**' --include '/audio/**' --include '/logs/**' --include '/configs/**'
```

## 검증

```bash
make test
uv run ruff check .
uv run python -m triviaqa_tts.cli.synthesize_questions --help
uv run pytest -m gpu -v                         # 기본: skip
RUN_QWEN_GPU_TEST=1 uv run pytest -m gpu -v    # CUDA RunPod: 실제 1문장 합성
```

GPU 테스트는 모델 다운로드와 VRAM을 사용하므로 opt-in이다. Dataset 또는 모델의 `main`이 바뀔 수
있다는 점과 GPU kernel의 bitwise 비결정성은 남는 환경 불확실성이며, 실제 SHA와 각 WAV hash로
추적한다.
