from __future__ import annotations

import argparse
import json
from pathlib import Path

from triviaqa_tts.data.manifest import iter_jsonl
from triviaqa_tts.moshi.evaluation import (
    MoshiAnswerer,
    append_result,
    load_audio_records,
    score_answer,
    write_answers,
    write_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Moshi on synthesized TriviaQA audio")
    parser.add_argument("--metadata", type=Path, default=Path("logs/pilot/metadata.jsonl"))
    output_dir = Path("logs/moshi_eval_preroll_5s_max20s")
    parser.add_argument("--output", type=Path, default=output_dir / "results.jsonl")
    parser.add_argument("--summary", type=Path, default=output_dir / "summary.json")
    parser.add_argument("--answers", type=Path, default=output_dir / "answers.json")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--model-id", default="kyutai/moshiko-pytorch-bf16")
    parser.add_argument("--revision")
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--preroll-seconds", type=float, default=5.0)
    # ponytail: hard cap prevents endless PAD/rambling generation.
    parser.add_argument("--max-response-seconds", type=float, default=20.0)
    parser.add_argument("--idle-stop-seconds", type=float, default=2.0)
    parser.add_argument("--minimum-response-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.max_response_seconds <= 0 or args.idle_stop_seconds <= 0:
        parser.error("response and idle limits must be positive")
    if args.preroll_seconds < 0:
        parser.error("preroll seconds must be non-negative")
    if not 0 <= args.minimum_response_seconds <= args.max_response_seconds:
        parser.error("minimum response must be between zero and maximum response")

    records = load_audio_records(args.metadata, args.limit)
    completed = (
        {str(record["question_id"]) for record in iter_jsonl(args.output)}
        if args.output.exists()
        else set()
    )
    pending = [record for record in records if str(record["question_id"]) not in completed]
    if pending:
        answerer = MoshiAnswerer(
            args.model_id,
            args.revision,
            args.seed,
            args.preroll_seconds,
            args.max_response_seconds,
            args.idle_stop_seconds,
            args.minimum_response_seconds,
        )
        for index, record in enumerate(pending, start=len(completed) + 1):
            result: dict[str, object] = {
                "question_id": record["question_id"],
                "original_index": record["original_index"],
                "question": record["question"],
                "answer_value": record["answer_value"],
                "aliases": record["aliases"],
                "audio_path": record["audio_path"],
                "audio_sha256": record["audio_sha256"],
                "moshi_model_id": args.model_id,
                "moshi_model_revision": answerer.model_revision,
                "moshi_seed": args.seed,
                "preroll_seconds": args.preroll_seconds,
                "max_response_seconds": args.max_response_seconds,
                "idle_stop_seconds": args.idle_stop_seconds,
                "minimum_response_seconds": args.minimum_response_seconds,
                "forced_epad": True,
            }
            try:
                result.update(answerer.answer(Path(str(record["audio_path"]))))
                result.update(score_answer(str(result["prediction"]), list(record["aliases"])))
                result["status"] = "success"
            except Exception as error:
                result.update(
                    status="failure", error_type=type(error).__name__, error_message=str(error)
                )
            append_result(args.output, result)
            print(f"{index}/{args.limit} {result['question_id']} {result['status']}", flush=True)

    summary = write_summary(args.output, args.summary, args.limit)
    write_answers(args.output, args.answers)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
