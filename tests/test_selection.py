from __future__ import annotations

import pytest

from triviaqa_tts.data.prepare import sample_pilot_indices


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
