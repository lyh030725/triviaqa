import json

from triviaqa_tts.moshi.evaluation import normalize_answer, score_answer, write_answers


def test_normalize_answer_matches_triviaqa_rules():
    assert normalize_answer("The_Cat's-Name") == "cat s name"


def test_score_answer_checks_exact_and_token_bounded_containment():
    assert score_answer("I think it was Rudolf Hess.", ["Hess, Rudolf", "Rudolf Hess"]) == {
        "exact_match": False,
        "alias_containment": True,
        "matched_alias": "Rudolf Hess",
    }
    assert score_answer("Russia", ["US"])["alias_containment"] is False
    assert score_answer("The Snickers", ["Snickers"])["exact_match"] is True


def test_write_answers_creates_separate_readable_json(tmp_path):
    results = tmp_path / "results.jsonl"
    results.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "question": "Who?",
                "answer_value": "Ada",
                "aliases": ["Ada Lovelace"],
                "prediction": "Ada Lovelace",
                "status": "success",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    answers = tmp_path / "answers.json"

    write_answers(results, answers)

    assert json.loads(answers.read_text(encoding="utf-8")) == [
        {
            "question_id": "q1",
            "question": "Who?",
            "reference_answer": "Ada",
            "aliases": ["Ada Lovelace"],
            "moshi_answer": "Ada Lovelace",
            "status": "success",
        }
    ]
