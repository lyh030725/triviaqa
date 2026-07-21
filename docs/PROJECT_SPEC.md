# TriviaQA 질문 TTS 데이터셋 생성 프로젝트 명세

## 1. 문서 상태

- 프로젝트: `triviaqa-tts`
- 작성일: 2026-07-21
- 상태: 구현 전 승인된 설계
- 기본 실행 위치: `/workspace/triviaqa-tts`
- 현재 개발 작업공간: `/workspace/triviaqa_tts/triviaqa-tts`

이 프로젝트는 TriviaQA 질문을 Qwen3-TTS로 합성하고, 검증된 WAV와 정답 및 생성 정보를 저장한다. 음성 QA 추론, Moshi 평가, EPAD 주입, ASR 기반 전사 평가, 정답 정확도 평가는 범위에 포함하지 않는다.

## 2. 공식 자료 조사 결과

### 2.1 TriviaQA

공식 Hugging Face dataset repository와 Datasets Server API를 2026-07-21에 확인했다.

- dataset: `mandarjoshi/trivia_qa`
- configuration: `unfiltered.nocontext`
- split: `validation`
- 확인한 repository SHA: `0f7faf33a3908546c6fd5b73a660e0f8ff173c2f`
- validation example 수: `11,313`
- configuration은 실제로 존재하며 Parquet builder를 사용한다.
- `unfiltered.nocontext`의 context 필드는 schema에는 남아 있지만 validation row에서 빈 sequence로 제공된다. 본 프로젝트는 `entity_pages`, `search_results`, Wikipedia evidence, Web evidence를 manifest에 복사하거나 TTS 입력에 사용하지 않는다.

실제 top-level schema는 다음과 같다.

| 필드 | 실제 형식 | 사용 여부 |
| --- | --- | --- |
| `question` | string | 보존 및 TTS 입력 |
| `question_id` | string | 보존 및 파일 식별 |
| `question_source` | string | 미사용 |
| `entity_pages` | sequence 구조 | 미사용 |
| `search_results` | sequence 구조 | 미사용 |
| `answer` | struct | 필요한 하위 필드 보존 |

실제 `answer` struct는 `aliases: sequence[string]`, `normalized_aliases: sequence[string]`, `matched_wiki_entity_name: string`, `normalized_matched_wiki_entity_name: string`, `normalized_value: string`, `type: string`, `value: string`을 포함한다. 본 프로젝트는 `value`, `aliases`, `normalized_aliases`를 보존한다.

실제 validation 첫 example은 `question_id="tc_2"`, `question="Who was the man behind The Chipmunks?"`, `answer.value="David Seville"`이며 alias 배열은 `['David Seville']`이다. 구현은 이 예시에 의존하지 않고 시작 시 schema contract를 검사한다.

자료:

- [Hugging Face TriviaQA dataset](https://huggingface.co/datasets/mandarjoshi/trivia_qa)
- [Dataset repository revision](https://huggingface.co/datasets/mandarjoshi/trivia_qa/commit/0f7faf33a3908546c6fd5b73a660e0f8ff173c2f)
- [Hugging Face Datasets Server API](https://datasets-server.huggingface.co/info?dataset=mandarjoshi%2Ftrivia_qa)

### 2.2 Qwen3-TTS

공식 QwenLM repository, 공식 Hugging Face model card와 model config를 2026-07-21에 확인했다.

- model ID: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- 확인한 model SHA: `0c0e3051f131929182e2c023b9537f8b1c68adfe`
- package: `qwen-tts`
- 확인한 package version: `0.1.1`
- 공식 import: `from qwen_tts import Qwen3TTSModel`
- 공식 loader: `Qwen3TTSModel.from_pretrained(...)`
- 공식 generation API: `generate_custom_voice(text=..., language=..., speaker=..., instruct=...)`
- 반환 형식: `(list[numpy.ndarray], sample_rate)`
- 선택 화자: `Aiden`
- 공식 설명: 맑은 중역대를 가진 밝은 미국 남성 음성, native language English
- 모델이 지원하는 다른 영어 native speaker: `Ryan`
- model tokenizer config의 native input/output sample rate: `24,000Hz`
- `speech_tokenizer/preprocessor_config.json`의 sampling rate: `24,000Hz`

CustomVoice의 공식 speaker 목록은 Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee다. 구현은 모델을 불러온 직후 `get_supported_speakers()`와 `get_supported_languages()`로 `Aiden` 및 `English`를 다시 검증한다.

공식 설치 문서는 깨끗한 Python 3.12 환경에서 `pip install -U qwen-tts`를 권장한다. 공식 package metadata는 `transformers==4.57.3`, `accelerate==1.12.0`을 고정하며 PyTorch 자체 버전은 고정하지 않는다. FlashAttention 2는 선택 권장이며 공식 FlashAttention 기준은 Linux, CUDA 12.0 이상, PyTorch 2.2 이상, Ampere/Ada/Hopper GPU와 fp16 또는 bf16이다.

자료:

- [Qwen3-TTS official repository](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen3-TTS official model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
- [Qwen3-TTS package metadata](https://github.com/QwenLM/Qwen3-TTS/blob/main/pyproject.toml)
- [FlashAttention official repository](https://github.com/Dao-AILab/flash-attention)
- [PyTorch official version matrix](https://pytorch.org/get-started/previous-versions/)

## 3. 목표와 비목표

### 3.1 목표

1. revision을 식별한 TriviaQA validation split으로 deterministic manifest를 만든다.
2. 고정 seed random sampling으로 pilot 100개를 만든다.
3. Qwen3-TTS 1.7B CustomVoice와 Aiden으로 질문을 하나씩 합성한다.
4. 최종 오디오는 24kHz mono PCM 16-bit WAV로 저장하고 다시 읽어 검증한다.
5. 정답, alias, 입력 문자열, 환경, revision, 오디오 hash를 metadata에 보존한다.
6. 중단 후 신뢰성 있게 resume하고 여러 Pod가 겹치지 않게 shard 처리한다.
7. GPU 없이 핵심 로직과 mock end-to-end 흐름을 검증할 수 있게 한다.

### 3.2 비목표

- CosyVoice3 또는 다른 TTS backend 구현
- TriviaQA evidence document 처리
- Moshi 또는 음성 QA 모델 추론
- EPAD 주입
- ASR transcription 또는 발음 자동 평가
- 정답 정확도 평가
- 웹 UI, 작업 큐, 데이터베이스 기반 orchestration

## 4. 프로젝트 구조

요청된 구조를 유지하고 test helper와 pilot report CLI만 필요한 위치에 추가한다.

```text
triviaqa-tts/
├── AGENTS.md
├── README.md
├── .gitignore
├── pyproject.toml
├── uv.lock
├── Makefile
├── configs/
│   ├── pilot.yaml
│   └── validation.yaml
├── scripts/
│   ├── bootstrap_runpod.sh
│   ├── prepare_dataset.sh
│   ├── synthesize_pilot.sh
│   └── synthesize_validation.sh
├── src/triviaqa_tts/
│   ├── __init__.py
│   ├── config.py
│   ├── reproducibility.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── prepare.py
│   │   └── manifest.py
│   ├── tts/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── qwen3_backend.py
│   │   └── synthesize.py
│   ├── audio/
│   │   ├── __init__.py
│   │   └── validation.py
│   └── cli/
│       ├── prepare_dataset.py
│       ├── synthesize_questions.py
│       └── report_pilot.py
├── tests/
│   ├── conftest.py
│   ├── test_manifest.py
│   ├── test_audio_validation.py
│   ├── test_resume.py
│   ├── test_reproducibility.py
│   ├── test_selection.py
│   └── test_smoke.py
├── data/
├── audio/
├── logs/
└── docs/
    ├── PROJECT_SPEC.md
    └── IMPLEMENTATION_PLAN.md
```

`report_pilot.py`, `test_selection.py`, `test_smoke.py`, `conftest.py`, `uv.lock`은 요구된 동작을 분리해 검증하고 재현 가능한 설치를 제공하기 위한 추가다.

## 5. 파일별 책임

| 파일 | 책임 |
| --- | --- |
| `config.py` | YAML parsing, typed configuration, path resolution, CLI override 검증 |
| `reproducibility.py` | 모든 RNG seed 설정, deterministic 설정, 환경 및 package version 수집 |
| `data/manifest.py` | schema contract, example 변환, canonical JSONL, hash, 안전한 JSONL 읽기 |
| `data/prepare.py` | revision 확인, dataset load, 전체 manifest 및 pilot 산출물 생성 |
| `tts/base.py` | 요청된 `TTSBackend` Protocol과 mock-friendly contract |
| `tts/qwen3_backend.py` | 지연 import, 공식 모델 loading, speaker/language 검증, 단일 합성 |
| `tts/synthesize.py` | 선택/샤딩, resume, 합성 transaction, 실패 처리, metadata append |
| `audio/validation.py` | mono 변환, 필요 시 resampling, PCM16 저장, 재읽기 검증, 품질 지표 |
| dataset CLI | manifest와 pilot 준비 진입점 |
| synthesis CLI | config/override 병합과 합성 진입점 |
| report CLI | pilot metadata 집계 및 JSON/CSV/청취 목록 생성 |
| shell scripts | RunPod 설치 및 반복 명령의 fail-fast wrapper |
| tests | GPU 독립 unit/mock smoke와 별도 GPU integration 검증 |

## 6. Dataset 준비 설계

### 6.1 Revision과 schema

config의 `dataset.revision`이 null이면 Hub에서 현재 commit SHA를 resolve한 뒤 그 SHA를 `load_dataset(..., revision=resolved_sha)`에 전달한다. 사용자가 revision을 지정하면 해당 revision을 SHA로 resolve하거나 로드 결과의 cache metadata와 Hub API를 대조한다. 실제 SHA를 manifest 각 record와 sidecar metadata에 기록한다.

로드 후 다음 contract를 검사한다.

- split 길이가 dataset info의 validation example 수와 일치
- `question`, `question_id`, `answer` 존재
- `answer.value`, `answer.aliases`, `answer.normalized_aliases` 존재
- 선언된 scalar/list 형식과 실제 값 형식 일치

schema 불일치는 전체 실행을 즉시 중단한다.

### 6.2 Manifest

`data/validation_manifest.jsonl`의 각 line은 UTF-8, `ensure_ascii=false`, compact JSON, 고정 key 순서, LF newline로 작성한다. record는 다음 필드를 가진다.

- `dataset`, `configuration`, `split`, `dataset_revision`
- `original_index`, `question_id`, `question`
- `answer_value`, `aliases`, `normalized_aliases`

manifest에는 생성 시각을 넣지 않는다. 생성 시각, example 수, manifest SHA-256, 중복 question ID 수, 빈 question 수, 빈 answer 수는 `data/validation_manifest.meta.json`에 기록한다. metadata의 timestamp가 달라져도 manifest hash는 동일하다. 임시 파일에 전체 manifest를 작성하고 hash를 계산한 후 atomic rename한다.

### 6.3 Pilot selection

- size: 100
- seed: 20260721
- 알고리즘: `random.Random(seed).sample(range(N), size)` 후 원본 순서 보존을 위해 index 정렬
- 결과: `data/pilot_100.jsonl`, `data/pilot_question_ids.txt`

Python의 sampling 알고리즘과 버전 차이 위험을 제거하기 위해 선택 index 목록과 selection algorithm version을 sidecar metadata에 기록한다. 같은 구현, manifest hash, seed에서 결과는 동일하다.

## 7. 합성 설계

### 7.1 Backend contract

`TTSBackend`는 `model_id`, `native_sample_rate`, `synthesize(text, seed) -> (waveform, sample_rate)`만 노출한다. Qwen backend는 model을 한 번 load하고 질문을 한 개씩 생성한다. model revision을 snapshot SHA로 resolve하고 실행 metadata에 기록한다.

Qwen backend는 다음 공식 API를 사용한다.

```python
model = Qwen3TTSModel.from_pretrained(
    model_id,
    revision=model_revision,
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation=resolved_attention,
)
wavs, sample_rate = model.generate_custom_voice(
    text=text,
    language="English",
    speaker="Aiden",
    instruct="",
)
```

FlashAttention 2 import와 GPU 호환성이 확인되면 사용하고, 아니면 `sdpa`를 사용한다. config 또는 CLI에서 알 수 없는 speaker/language를 전달하면 generation 전에 실패한다.

### 7.2 질문 전처리

처리 순서는 HTML entity decode, 앞뒤 공백 제거, 모든 연속 whitespace를 한 칸으로 축약, 종결 문장부호가 전혀 없으면 `?` 추가다. 원문 `question`과 실제 `tts_text`를 모두 저장한다. 요약, 재작성, 정답 삽입, 고유명사 교체, 발음 힌트 삽입은 금지한다.

### 7.3 파일명

question ID는 안전한 ASCII 영숫자, `-`, `_`, `.`만 남기고 나머지는 `_`로 바꾸며 path traversal과 reserved name을 차단한다. 항상 원래 question ID SHA-256의 짧은 suffix를 붙여 정규화 충돌도 제거한다. 최종 규칙과 길이 제한은 unit test로 고정한다.

### 7.4 Selection과 sharding

처리 순서는 manifest order를 기준으로 한다.

1. `start-index` inclusive와 `end-index` exclusive 적용
2. deterministic shard: 선택된 record의 전역 ordinal에 대해 `ordinal % num_shards == shard_index`
3. `limit` 적용

`0 <= shard_index < num_shards`, 양수 `num_shards`, 유효한 index range를 강제한다. shard는 서로 겹치지 않고 합치면 range 전체가 된다.

## 8. 오디오 저장과 검증

Qwen 공식 tokenizer는 24kHz를 출력한다. 실제 API 반환 sample rate도 매 sample 확인한다. 다른 rate가 반환되면 `soxr` high-quality mode로 24kHz로 변환한다. 다채널 입력은 명시적으로 평균 mono 변환하되 원래 shape를 로그에 남긴다.

최종 저장 형식:

- WAV
- 24,000Hz
- 1 channel
- PCM 16-bit (`PCM_16`)

WAV는 최종 경로와 같은 filesystem의 임시 파일에 저장한 뒤 다시 읽는다. 다음 필수 조건을 모두 통과해야 성공 파일로 atomic rename한다.

- 존재하고 크기가 0보다 큼
- soundfile decoding 성공
- 1 channel, 24kHz
- sample 수와 duration이 0보다 큼
- NaN/Inf 없음
- sample 값이 유효 범위 안에 있음

품질 지표는 peak amplitude, clipped sample count, zero-valued sample ratio, duration, leading silence, trailing silence다. config threshold를 넘는 길이, clipping, silence는 WAV를 삭제하지 않고 `warning` status와 warning reason 배열로 기록한다. 필수 검증 실패는 success로 기록하지 않는다.

기본 기준:

```yaml
audio_validation:
  minimum_duration_seconds: 0.3
  maximum_duration_seconds: 60.0
  clipping_threshold: 0.999
  silence_threshold: 0.01
  maximum_leading_silence_seconds: 2.0
  maximum_trailing_silence_seconds: 3.0
```

## 9. Atomicity, metadata, resume

각 sample transaction은 normalize/hash, seed reset, synthesize, convert, temp write, decode/validate, audio hash, atomic rename, metadata append 순서다. metadata line은 단일 JSON object로 serialize해 한 번에 append하고 flush 및 `fsync`한다.

success/warning metadata에는 요구된 필드와 함께 `normalized_aliases`, dataset/model revision, 환경 정보, warning reasons, zero ratio, silence metrics, generation configuration을 포함한다. 일반 sample 오류는 failures JSONL에도 기록한다.

JSONL reader는 완전한 newline으로 끝나지 않는 마지막 line의 JSON parse 실패만 안전하게 무시하고 경고한다. 완결된 line 또는 중간 line이 손상되면 즉시 실패한다. 같은 question ID에 여러 record가 있으면 마지막 유효 record를 현재 상태로 사용한다.

Resume 완료 조건:

- WAV 존재
- 최신 metadata status가 `success` 또는 `warning`
- 실제 audio SHA-256 일치
- WAV 필수 재검증 통과
- question SHA-256 일치
- model ID와 resolved revision 일치
- speaker, language, seed, generation configuration 일치
- stored sample rate/channel/subtype 일치

하나라도 다르면 재생성한다. `--overwrite`는 모든 선택 sample을 재생성하고, `--resume`과 동시에 사용할 수 없다.

일반 sample 오류는 기록 후 계속한다. CUDA OOM은 cache를 비우고 같은 sample을 한 번만 재시도한 뒤 실패로 기록하고 계속한다. model load, config/schema, disk full, metadata write 문제는 결과 일관성에 영향을 주므로 전체 실행을 중단한다.

## 10. 재현성

각 질문 합성 직전에 같은 seed 42를 Python `random`, NumPy, PyTorch CPU, 모든 CUDA device에 설정한다. 가능한 deterministic PyTorch 설정과 cuDNN flags도 적용하되 지원하지 않는 Qwen/FlashAttention 연산 때문에 bitwise determinism을 강제하다 실패하지는 않는다.

모든 WAV의 SHA-256을 기록하며 실행 metadata에는 Python, NumPy, PyTorch, CUDA runtime, GPU, qwen-tts, transformers, model/dataset revision, dtype, attention implementation과 generation kwargs를 기록한다.

## 11. RunPod 설치

Python 3.12와 `uv`가 관리하는 project `.venv`를 사용한다. Conda보다 환경 생성과 lockfile 재현이 단순하고, 일반 `venv`보다 dependency resolution과 재설치가 빠르기 때문이다.

`bootstrap_runpod.sh`는 fail-fast로 다음을 수행한다.

1. Linux와 `/workspace` 확인
2. `ffmpeg`, `libsndfile1`, build tools, `git`, `curl` 등 설치
3. `uv` 설치 또는 확인
4. Python 3.12와 `.venv` 준비
5. lock된 dependency 및 editable project 설치
6. `/workspace/huggingface_cache` 생성과 HF cache 환경 안내 파일 작성
7. CUDA, GPU 이름, bf16, disk space 확인
8. PyTorch/CUDA/qwen-tts/transformers/soundfile 버전 출력
9. 선택적으로 FlashAttention 2 설치/검증; 실패하면 경고하고 SDPA 사용

Qwen package가 PyTorch를 고정하지 않으므로 bootstrap은 RunPod CUDA image와 lockfile의 torch/torchaudio 조합을 검증한다. 기본 지원선은 CUDA 12.x와 PyTorch 2.2 이상이다. 실제 잠금 버전은 구현 시점의 공식 PyTorch CUDA wheel matrix와 Qwen dependency를 함께 통과한 조합으로 `pyproject.toml`, `uv.lock`, README에 동일하게 기록한다.

## 12. CLI와 Make targets

지원 CLI:

- `python -m triviaqa_tts.cli.prepare_dataset --config ...`
- `python -m triviaqa_tts.cli.synthesize_questions --config ...`
- `python -m triviaqa_tts.cli.report_pilot --config configs/pilot.yaml`

합성 CLI는 `--limit`, `--start-index`, `--end-index`, `--shard-index`, `--num-shards`, `--resume`, `--overwrite`, `--dry-run`을 지원한다. dry-run은 모델을 load하지 않고 선택 결과, 출력 경로, resume 후보를 출력한다.

Make targets:

- `make setup`: bootstrap 실행
- `make prepare`: 전체 manifest와 pilot 생성
- `make pilot`: pilot만 합성
- `make test`: CPU unit/mock smoke test
- `make validation`: 전체 validation 명시적 합성

설치나 prepare 후 validation 합성이 자동 시작되지 않는다.

## 13. Pilot 보고서

pilot metadata를 읽어 `logs/pilot_audio_report.json`과 `.csv`를 생성한다. 성공, 실패, warning, 총/평균/최소/최대 duration, clipping/short/leading silence/trailing silence 수를 포함한다. pilot selection seed와 별도의 고정 report seed로 10개를 뽑아 경로와 질문을 stdout 및 JSON에 기록한다. ASR은 사용하지 않는다.

## 14. 테스트 전략과 인수 기준

CPU test는 example 변환, alias 보존, pilot 재현성, manifest hash, filename, normalization, NaN/Inf/clipping, resume 조건, 손상된 마지막 JSONL, shard disjointness/coverage를 검증한다. Mock backend smoke test는 작은 waveform에서 WAV, validation, metadata, resume, report까지 실행한다.

`gpu` marker test만 실제 Qwen model download와 단일 짧은 synthesis를 허용한다. 기본 test suite는 network와 GPU를 요구하지 않는다.

완료 판정은 사용자 요구사항의 14개 조건과 일치한다. 특히 CPU unit test와 mock end-to-end test가 실제로 통과해야 하며, GPU가 없으면 GPU integration test 미실행 사실을 명시한다.

## 15. 알려진 위험과 대응

| 위험 또는 불확실성 | 대응 |
| --- | --- |
| Hub의 `main` revision 변경 | 실행 시작 시 SHA resolve 후 모든 load와 metadata를 SHA에 고정 |
| `datasets`가 Parquet schema 표현을 바꿈 | 필요한 nested contract만 엄격히 검사하고 실제 값도 검증 |
| qwen-tts/PyTorch/CUDA 조합 변화 | lockfile, bootstrap compatibility check, 버전 metadata |
| FlashAttention build 실패 | 선택 dependency로 취급하고 SDPA fallback |
| GPU가 bf16 미지원 | 명확한 사전검사; config가 허용하면 fp16 사용, 기본은 bf16 |
| CUDA sampling의 bitwise 비결정성 | 질문별 RNG reset과 WAV SHA-256 기록 |
| JSONL append 중 Pod 종료 | 마지막 partial line 무시, WAV atomic rename, strict resume |
| question ID path/충돌 | sanitize와 원 ID hash suffix |
| disk full 또는 inode 부족 | 시작 전 확인, write failure fatal, README에 용량 산정과 백업 안내 |
| OOM | 단일 sample 처리, 한 번 재시도, 실패 기록, 0.6B 자동 전환 금지 |
| 매우 긴 질문이 60초 초과 | 생성물 보존 및 warning; 질문 내용 자동 변경 금지 |

## 16. 승인된 주요 결정

- Qwen3-TTS `1.7B`를 기본값으로 사용한다.
- 영어 화자는 `Aiden`을 사용한다.
- 단일 sample streaming pipeline과 deterministic multi-Pod sharding을 사용한다.
- manifest와 시간 가변 sidecar metadata를 분리한다.
- 24kHz native 출력을 확인하되 API 반환값을 신뢰의 최종 근거로 다시 검사한다.
- `uv`와 Python 3.12 project environment를 사용한다.
- JSONL과 파일 기반 resume를 유지하며 별도 DB나 queue를 도입하지 않는다.
