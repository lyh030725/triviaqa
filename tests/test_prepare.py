from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Features, Sequence, Value

from triviaqa_tts.config import AppConfig
from triviaqa_tts.data.manifest import iter_jsonl, sha256_file
from triviaqa_tts.data.prepare import prepare_dataset, validate_triviaqa_schema


def triviaqa_features() -> Features:
    return Features(
        {
            "question": Value("string"),
            "question_id": Value("string"),
            "question_source": Value("string"),
            "entity_pages": Sequence({"title": Value("string")}),
            "search_results": Sequence({"title": Value("string")}),
            "answer": {
                "aliases": Sequence(Value("string")),
                "normalized_aliases": Sequence(Value("string")),
                "matched_wiki_entity_name": Value("string"),
                "normalized_matched_wiki_entity_name": Value("string"),
                "normalized_value": Value("string"),
                "type": Value("string"),
                "value": Value("string"),
            },
        }
    )


class FakeDataset:
    def __init__(self, rows, *, features=None, advertised_count=None):
        self._rows = rows
        self.features = features or triviaqa_features()
        count = len(rows) if advertised_count is None else advertised_count
        self.info = SimpleNamespace(splits={"validation": SimpleNamespace(num_examples=count)})

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)


class FakeHubApi:
    def __init__(self, sha="f" * 40):
        self.sha = sha
        self.calls = []

    def dataset_info(self, repo_id, *, revision):
        self.calls.append((repo_id, revision))
        return SimpleNamespace(sha=self.sha)


def make_config(tmp_path: Path, *, pilot_size=2) -> AppConfig:
    config = AppConfig()
    return replace(
        config,
        dataset=replace(config.dataset, revision="release-tag"),
        selection=replace(
            config.selection,
            manifest_path=tmp_path / "validation_manifest.jsonl",
            pilot_size=pilot_size,
            pilot_seed=17,
        ),
        paths=replace(config.paths, data_dir=tmp_path, cache_dir=tmp_path / "cache"),
    )


def valid_rows():
    return [
        {
            "question_id": "one",
            "question": "First question?",
            "answer": {
                "value": "First",
                "aliases": ["First", "1st"],
                "normalized_aliases": ["first", "1st"],
            },
        },
        {
            "question_id": "duplicate",
            "question": "   ",
            "answer": {
                "value": "Second",
                "aliases": ["Second"],
                "normalized_aliases": ["second"],
            },
        },
        {
            "question_id": "duplicate",
            "question": "Third question?",
            "answer": {"value": "", "aliases": [""], "normalized_aliases": [""]},
        },
    ]


def test_prepare_pins_resolved_sha_and_writes_complete_atomic_outputs(tmp_path):
    config = make_config(tmp_path)
    dataset = FakeDataset(valid_rows())
    loader_calls = []

    def loader(path, name, **kwargs):
        loader_calls.append((path, name, kwargs))
        return dataset

    hub_api = FakeHubApi()

    result = prepare_dataset(config, dataset_loader=loader, hub_api=hub_api)

    assert hub_api.calls == [("mandarjoshi/trivia_qa", "release-tag")]
    assert loader_calls == [
        (
            "mandarjoshi/trivia_qa",
            "unfiltered.nocontext",
            {
                "split": "validation",
                "revision": "f" * 40,
                "cache_dir": str(tmp_path / "cache"),
            },
        )
    ]
    records = list(iter_jsonl(result.manifest_path, allow_partial_last_line=False))
    assert [record["original_index"] for record in records] == [0, 1, 2]
    assert {record["dataset_revision"] for record in records} == {"f" * 40}
    assert records[0]["aliases"] == ["First", "1st"]

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["dataset_revision"] == "f" * 40
    assert metadata["example_count"] == 3
    assert metadata["manifest_sha256"] == sha256_file(result.manifest_path)
    assert metadata["duplicate_question_id_count"] == 1
    assert metadata["empty_question_count"] == 1
    assert metadata["empty_answer_count"] == 1
    assert metadata["pilot_indices"] == result.pilot_indices
    assert metadata["selection_algorithm_version"]

    pilot_records = list(iter_jsonl(result.pilot_manifest_path, allow_partial_last_line=False))
    assert pilot_records == [records[index] for index in result.pilot_indices]
    assert result.pilot_ids_path.read_text(encoding="utf-8").splitlines() == [
        str(record["question_id"]) for record in pilot_records
    ]
    assert not list(tmp_path.glob("*.tmp"))


def test_missing_nested_schema_fails_before_writing_files(tmp_path):
    features = triviaqa_features()
    del features["answer"]["normalized_aliases"]
    dataset = FakeDataset(valid_rows(), features=features)
    config = make_config(tmp_path)

    with pytest.raises(ValueError, match=r"answer\.normalized_aliases"):
        prepare_dataset(config, dataset_loader=lambda *args, **kwargs: dataset, hub_api=FakeHubApi())

    assert not config.selection.manifest_path.exists()
    assert not (tmp_path / "validation_manifest.meta.json").exists()
    assert not (tmp_path / "pilot_2.jsonl").exists()
    assert not (tmp_path / "pilot_question_ids.txt").exists()


def test_runtime_schema_mismatch_fails_before_writing_files(tmp_path):
    rows = valid_rows()
    rows[1]["answer"]["aliases"] = "not-a-list"
    dataset = FakeDataset(rows)
    config = make_config(tmp_path)

    with pytest.raises(ValueError, match=r"row 1.*answer\.aliases"):
        prepare_dataset(config, dataset_loader=lambda *args, **kwargs: dataset, hub_api=FakeHubApi())

    assert not config.selection.manifest_path.exists()


def test_validate_schema_rejects_wrong_declared_scalar_type():
    features = triviaqa_features()
    features["question"] = Value("int64")

    with pytest.raises(ValueError, match="question.*string"):
        validate_triviaqa_schema(FakeDataset(valid_rows(), features=features))
