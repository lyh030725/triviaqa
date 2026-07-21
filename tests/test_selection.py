from __future__ import annotations

import pytest

from triviaqa_tts.data.prepare import sample_pilot_indices
from triviaqa_tts.tts.synthesize import select_records


def test_pilot_sampling_is_reproducible_and_not_prefix():
    first = sample_pilot_indices(11_313, 100, 20_260_721)
    second = sample_pilot_indices(11_313, 100, 20_260_721)

    assert first == second == sorted(first)
    assert first != list(range(100))
    assert len(set(first)) == 100


@pytest.mark.parametrize(("count", "size"), [(0, 1), (3, 0), (3, 4)])
def test_pilot_sampling_rejects_invalid_sizes(count, size):
    with pytest.raises(ValueError):
        sample_pilot_indices(count, size, seed=42)


def test_shards_are_disjoint_and_cover_the_selected_global_range():
    records = [{"original_index": index, "question_id": f"q{index}"} for index in range(20)]

    shards = [
        select_records(
            records,
            start_index=3,
            end_index=17,
            shard_index=shard_index,
            num_shards=4,
            limit=None,
        )
        for shard_index in range(4)
    ]

    shard_ids = [{record["question_id"] for record in shard} for shard in shards]
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(shard_ids)
        for right in shard_ids[index + 1 :]
    )
    assert set().union(*shard_ids) == {f"q{index}" for index in range(3, 17)}
    assert [[record["original_index"] for record in shard] for shard in shards] == [
        [4, 8, 12, 16],
        [5, 9, 13],
        [6, 10, 14],
        [3, 7, 11, 15],
    ]


def test_limit_is_applied_after_range_and_global_ordinal_sharding():
    records = [{"original_index": index} for index in range(12)]

    selected = select_records(
        records,
        start_index=2,
        end_index=11,
        shard_index=1,
        num_shards=3,
        limit=2,
    )

    assert [record["original_index"] for record in selected] == [4, 7]
