from __future__ import annotations

import json
import random
import re
import string
import time
from itertools import chain, repeat
from pathlib import Path

import numpy as np

from triviaqa_tts.data.manifest import iter_jsonl

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = set(string.punctuation + "‘’´`")


def normalize_answer(text: str) -> str:
    """Apply TriviaQA's official exact-match normalization."""
    text = text.lower().replace("_", " ")
    text = "".join(character if character not in _PUNCTUATION else " " for character in text)
    return " ".join(_ARTICLES.sub(" ", text).split())


def score_answer(prediction: str, aliases: list[str]) -> dict[str, bool | str | None]:
    normalized_prediction = normalize_answer(prediction)
    normalized_aliases = [(alias, normalize_answer(alias)) for alias in aliases]
    exact = next(
        (alias for alias, normalized in normalized_aliases if normalized == normalized_prediction),
        None,
    )
    padded_prediction = f" {normalized_prediction} "
    contained = next(
        (
            alias
            for alias, normalized in normalized_aliases
            if normalized and f" {normalized} " in padded_prediction
        ),
        None,
    )
    return {
        "exact_match": exact is not None,
        "alias_containment": contained is not None,
        "matched_alias": exact or contained,
    }


def load_audio_records(metadata_path: Path, limit: int) -> list[dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for record in iter_jsonl(metadata_path):
        if record.get("status") != "success":
            continue
        question_id = str(record["question_id"])
        records[question_id] = record
    selected = list(records.values())[:limit]
    if len(selected) < limit:
        raise ValueError(f"requested {limit} audio records, found {len(selected)}")
    missing = [
        str(record["audio_path"])
        for record in selected
        if not Path(str(record["audio_path"])).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing audio file: {missing[0]}")
    return selected


class MoshiAnswerer:
    """Load Moshi once and answer 24 kHz mono WAV questions one at a time."""

    def __init__(
        self,
        model_id: str,
        revision: str | None,
        seed: int,
        preroll_seconds: float,
        max_response_seconds: float,
        idle_stop_seconds: float,
        minimum_response_seconds: float,
    ) -> None:
        import torch
        from moshi.models import LMGen, loaders

        if not torch.cuda.is_available():
            raise RuntimeError("Moshi evaluation requires CUDA")
        self.torch = torch
        self.seed = seed
        self.preroll_seconds = preroll_seconds
        self.max_response_seconds = max_response_seconds
        self.idle_stop_seconds = idle_stop_seconds
        self.minimum_response_seconds = minimum_response_seconds
        self.checkpoint = loaders.CheckpointInfo.from_hf_repo(model_id, revision=revision)
        self.mimi = self.checkpoint.get_mimi(device="cuda")
        self.tokenizer = self.checkpoint.get_text_tokenizer()
        self.lm = self.checkpoint.get_moshi(device="cuda", dtype=torch.bfloat16).eval()
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.generated_tokens: list[int] = []
        self.force_epad = False
        self.collecting = False
        self.response_step = 0
        self.last_content_step: int | None = None
        self.saw_eos = False
        self.lm_gen = LMGen(
            self.lm,
            on_text_hook=self._on_text,
            **self.checkpoint.lm_gen_config,
        )
        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

    @property
    def model_revision(self) -> str:
        return Path(self.checkpoint.moshi_weights).parent.name

    def _on_text(self, token) -> None:
        token_id = int(token[0].item())
        if self.force_epad:
            token.fill_(self.lm.end_of_text_padding_id)
            self.force_epad = False
            self.collecting = True
            return
        if not self.collecting:
            return
        self.generated_tokens.append(token_id)
        if token_id not in (self.lm.text_padding_token_id, self.lm.end_of_text_padding_id):
            self.last_content_step = self.response_step
        if token_id == self.tokenizer.eos_id():
            self.saw_eos = True

    def answer(self, audio_path: Path) -> dict[str, object]:
        import soundfile as sf

        started_at = time.monotonic()
        waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        if sample_rate != self.mimi.sample_rate or waveform.shape[1] != 1:
            raise ValueError(
                f"expected mono {self.mimi.sample_rate} Hz WAV, got {waveform.shape[1]} channel(s) at {sample_rate} Hz"
            )
        self._seed_all()
        self.mimi.reset_streaming()
        self.lm_gen.reset_streaming()
        self.generated_tokens = []
        self.force_epad = False
        self.collecting = False
        self.last_content_step = None
        self.saw_eos = False

        samples = self.torch.from_numpy(waveform[:, 0]).to(device="cuda")[None, None]
        remainder = samples.shape[-1] % self.frame_size
        if remainder:
            samples = self.torch.nn.functional.pad(samples, (0, self.frame_size - remainder))

        silence = self.torch.zeros((1, 1, self.frame_size), device="cuda")
        preroll_steps = round(self.preroll_seconds * self.mimi.frame_rate)
        input_frames = chain(
            repeat(silence, preroll_steps),
            samples.split(self.frame_size, dim=-1),
        )
        first_frame = True
        for frame in input_frames:
            codes = self.mimi.encode(frame)
            if first_frame:
                self.lm_gen.step(codes)
                first_frame = False
            self.lm_gen.step(codes)

        max_steps = round(self.max_response_seconds * self.mimi.frame_rate)
        minimum_steps = round(self.minimum_response_seconds * self.mimi.frame_rate)
        idle_steps = round(self.idle_stop_seconds * self.mimi.frame_rate)
        self.force_epad = True
        stop_reason = "max_response_seconds"
        for step in range(max_steps):
            self.response_step = step
            self.lm_gen.step(self.mimi.encode(silence))
            if self.saw_eos:
                stop_reason = "eos"
                break
            if (
                step >= minimum_steps
                and self.last_content_step is not None
                and step - self.last_content_step >= idle_steps
            ):
                stop_reason = "text_idle"
                break

        special_ids = {
            self.lm.text_padding_token_id,
            self.lm.end_of_text_padding_id,
            self.tokenizer.eos_id(),
        }
        content_ids = [token for token in self.generated_tokens if token not in special_ids]
        return {
            "prediction": self.tokenizer.decode_ids(content_ids).strip(),
            "generated_text_token_count": len(content_ids),
            "response_steps": self.response_step + 1,
            "stop_reason": stop_reason,
            "elapsed_seconds": round(time.monotonic() - started_at, 3),
        }

    def _seed_all(self) -> None:
        self.torch.manual_seed(self.seed)
        self.torch.cuda.manual_seed_all(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)


def append_result(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        stream.write("\n")
        stream.flush()


def write_summary(results_path: Path, summary_path: Path, expected_count: int) -> dict[str, object]:
    results = list(iter_jsonl(results_path))
    successes = [result for result in results if result.get("status") == "success"]
    exact = sum(bool(result.get("exact_match")) for result in successes)
    containment = sum(bool(result.get("alias_containment")) for result in successes)
    summary = {
        "expected_count": expected_count,
        "completed_count": len(results),
        "success_count": len(successes),
        "failure_count": len(results) - len(successes),
        "exact_match_count": exact,
        "exact_match_accuracy": exact / expected_count,
        "alias_containment_count": containment,
        "alias_containment_accuracy": containment / expected_count,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(summary_path)
    return summary


def write_answers(results_path: Path, answers_path: Path) -> None:
    answers = [
        {
            "question_id": result["question_id"],
            "question": result["question"],
            "reference_answer": result["answer_value"],
            "aliases": result["aliases"],
            "moshi_answer": result.get("prediction"),
            "status": result["status"],
        }
        for result in iter_jsonl(results_path)
    ]
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = answers_path.with_suffix(answers_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(answers, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(answers_path)
