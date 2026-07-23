# TriviaQA 질문 TTS 데이터셋

TriviaQA validation 질문을 `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`의 영어 화자
`Aiden`으로 합성하는 재현 가능한 RunPod 파이프라인이다. 질문과 정답 alias는 보존하지만 evidence
문서는 읽거나 복사하지 않는다. 합성 WAV는 공개 Moshi bf16 checkpoint의 closed-book 음성 QA
baseline에도 사용한다.

## 고정 입력과 환경

- Dataset: `mandarjoshi/trivia_qa`, `unfiltered.nocontext`, `validation` 11,313개
- 확인한 dataset revision: `0f7faf33a3908546c6fd5b73a660e0f8ff173c2f`
- 사용 schema: `question`, `question_id`, `answer.value`, `answer.aliases`,
  `answer.normalized_aliases`
- 확인한 model revision: `0c0e3051f131929182e2c023b9537f8b1c68adfe`
- 출력: 24,000Hz, mono, PCM 16-bit WAV; seed 42
- Python 3.12, `qwen-tts==0.1.1`, PyTorch/TorchAudio 2.11.0 cu128
- 4분 이상으로 선별된 표본은 1,639. 중복 오디오를 제외하면 1,635개
- 데이터 오디오를 context_path로 전달: audiomarathon-moshi-ringkv/src/audiomarathon_ringkv/evaluate.py:233
- 실제 입력 순서: 5초 무음 → context 오디오 → 0.25초 무음 → 질문/선택지 TTS: audiomarathon-moshi-ringkv/src/
    audiomarathon_ringkv/evaluate.py:84

- 다만 RingKV가 기본 240초만 보존하므로, 전체 입력이 길면 context의 앞부분부터 제거
- lm_kwargs_overrides={"context": 3000}의 context는 별도 텍스트가 아니라 캐시 길이 설정
  ,audiomarathon-moshi-ringkv/src/audiomarathon_ringkv/evaluate.py:52

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

## Moshi 평가

이 평가는 논문 Table 8의 완전 재현이 아니다. 저자들이 TriviaQA configuration/split, TTS 엔진,
incipit 목록, 생성 종료 조건, 채점 코드를 공개하지 않았기 때문이다. 공개 자원만 고정한 baseline이며,
후속 full-cache/compression 비교도 같은 protocol을 사용해야 한다.

현재 protocol:

- 입력: `unfiltered.nocontext` validation에서 합성에 성공해
  `logs/pilot/metadata.jsonl`에 기록된 순서의 500개 mono 24kHz WAV
- TTS: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`, Aiden, seed 42
- 모델: `kyutai/moshiko-pytorch-bf16`, bf16; 실행 결과에 실제 Hugging Face revision 기록
- 생성: reset 후 silent user frame 5초를 넣어 공개 대화형 checkpoint의 시작 인사를 기다린 뒤
  질문 WAV를 Mimi/Moshi user stream에 넣고 질문 종료 직후 text stream에 model의
  `end_of_text_padding_id`인 EPAD 1개 강제. 이후 silent user frames로 응답 생성
- sampling: Moshi 공개 `LMGen` 기본값, 질문마다 seed 4242와 streaming state reset
- 종료: 최대 20초. text token이 나온 뒤 2초 동안 새 token이 없으면 조기 종료
- 예측: Moshi Inner Monologue의 PAD/EPAD/EOS 제외 text
- 채점: TriviaQA 공식 normalization 후 모든 alias 대상 exact match와 token-bounded
  alias containment. 실패도 500개 분모에 포함

5초 pre-roll은 공개 모델용 우회책이며 논문의 미공개 random incipit 또는 base checkpoint를
재현하지 않는다. 20초를 넘는 응답이 보이면
`--max-response-seconds`를 늘려 전 항목을 같은 값으로 다시 평가한다.

Moshi는 기존 CUDA/Torch 설치를 재사용한다. 검증한 Moshi source commit은
`e6a55d2722a65870ef52a6c9f6ecfc0e90f38362` (`moshi==0.2.13`)다. upstream package가 아직
Torch 2.11을 dependency 범위에 포함하지 않아 dependency 해결만 건너뛴다. 현재 RTX 4090 smoke
평가에서 동작을 확인했다.

```bash
uv pip install --python .venv/bin/python --no-deps \
  'git+https://github.com/kyutai-labs/moshi.git@e6a55d2722a65870ef52a6c9f6ecfc0e90f38362#subdirectory=moshi'
uv pip install --python .venv/bin/python 'sentencepiece>=0.2,<0.3'
./scripts/evaluate_moshi.sh
```

중단 후 같은 명령을 실행하면 `question_id` 기준으로 완료 항목을 건너뛴다. 산출물:

```text
logs/moshi_eval_preroll_5s_max20s/results.jsonl  # 항목별 prediction, score, model/audio revision, 실행 시간
logs/moshi_eval_preroll_5s_max20s/summary.json  # EM, alias-containment accuracy, 성공/실패 수
logs/moshi_eval_preroll_5s_max20s/answers.json  # 질문, 정답, alias, Moshi 원문 답변
logs/moshi_eval_preroll_5s_max20s/correct_answers.json  # alias containment 정답만
```

2026-07-22 현재 5초 pre-roll·20초 상한 500개 결과:

- checkpoint revision: `2bfc9ae6e89079a5cc7ed2a68436010d91a3d289`
- 성공/실패: 500/0
- exact match: 1/500, 0.2%
- alias containment: 83/500, 16.6%
- 종료: text idle 499개, 20초 상한 1개
- 기존 5초 pre-roll·8초 상한과 정확도 동일
- 8초 대비 답변 text 변경 44개, 정답/오답 전환 0개
- 0초 pre-roll baseline 대비 containment: 59개에서 83개, +24개(+4.8%p)

근거: [Moshi 논문](https://arxiv.org/abs/2410.00037),
[공식 PyTorch 구현](https://github.com/kyutai-labs/moshi),
[Spoken QA 공식 Issue #87](https://github.com/kyutai-labs/moshi/issues/87),
[TriviaQA 공식 채점 코드](https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py).

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
