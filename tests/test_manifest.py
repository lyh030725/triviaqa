from __future__ import annotations

import hashlib
import warnings

import pytest

from triviaqa_tts.data.manifest import (
    ManifestFormatError,
    example_to_record,
    iter_jsonl,
    normalize_question,
    safe_audio_stem,
    sha256_file,
    write_canonical_jsonl,
)


def test_example_to_record_preserves_answer_lists():
    row = {
        "question_id": "tc/2",
        "question": "  Who  &amp; why ",
        "answer": {
            "value": "Answer",
            "aliases": ["A", "Answer"],
            "normalized_aliases": ["a", "answer"],
        },
    }

    record = example_to_record(row, original_index=7, dataset_revision="abc")

    assert record == {
        "dataset": "mandarjoshi/trivia_qa",
        "configuration": "unfiltered.nocontext",
        "split": "validation",
        "dataset_revision": "abc",
        "original_index": 7,
        "question_id": "tc/2",
        "question": "  Who  &amp; why ",
        "answer_value": "Answer",
        "aliases": ["A", "Answer"],
        "normalized_aliases": ["a", "answer"],
    }


def test_normalize_question_is_minimal():
    assert normalize_question("  Who  &amp; why ") == "Who & why?"
    assert normalize_question("Already complete!") == "Already complete!"


def test_canonical_jsonl_is_identical_utf8_lf_and_hashable(tmp_path):
    records = [
        {
            "dataset": "mandarjoshi/trivia_qa",
            "configuration": "unfiltered.nocontext",
            "split": "validation",
            "dataset_revision": "abc",
            "original_index": 0,
            "question_id": "é/1",
            "question": "Café?",
            "answer_value": "Café",
            "aliases": ["Café"],
            "normalized_aliases": ["café"],
        }
    ]
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    first_hash = write_canonical_jsonl(records, first)
    second_hash = write_canonical_jsonl(records, second)

    expected_bytes = (
        b'{"dataset":"mandarjoshi/trivia_qa","configuration":"unfiltered.nocontext",'
        b'"split":"validation","dataset_revision":"abc","original_index":0,'
        b'"question_id":"\xc3\xa9/1","question":"Caf\xc3\xa9?",'
        b'"answer_value":"Caf\xc3\xa9","aliases":["Caf\xc3\xa9"],'
        b'"normalized_aliases":["caf\xc3\xa9"]}\n'
    )
    assert first.read_bytes() == expected_bytes
    assert second.read_bytes() == expected_bytes
    assert first_hash == second_hash == hashlib.sha256(expected_bytes).hexdigest()
    assert sha256_file(first) == first_hash


def test_safe_audio_stem_uses_sanitized_prefix_and_question_id_hash():
    question_id = "trivia / question: 42"

    stem = safe_audio_stem(question_id)

    assert stem == f"trivia_question_42_{hashlib.sha256(question_id.encode()).hexdigest()[:12]}"


def test_iter_jsonl_warns_and_ignores_only_invalid_unterminated_final_line(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_bytes(b'{"question_id":"one"}\n{"question_id":')

    with pytest.warns(UserWarning, match="incomplete final JSONL record"):
        assert list(iter_jsonl(path)) == [{"question_id": "one"}]


@pytest.mark.parametrize(
    ("contents", "allow_partial_last_line"),
    [
        (b'{"question_id":}\n', True),
        (b'{"question_id":}\n{"question_id":"two"}\n', True),
        (b'{"question_id":"one"}\n{"question_id":', False),
    ],
)
def test_iter_jsonl_rejects_complete_or_disallowed_partial_corruption(
    tmp_path, contents, allow_partial_last_line
):
    path = tmp_path / "manifest.jsonl"
    path.write_bytes(contents)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ManifestFormatError):
            list(iter_jsonl(path, allow_partial_last_line=allow_partial_last_line))
