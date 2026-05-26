import numpy as np
import pytest

from bayes_chunk.segmentation import find_boundaries_by_top_k


def test_empty_surprisal_returns_no_boundaries():
    assert find_boundaries_by_top_k([]) == []


def test_boundary_stride_must_be_positive():
    with pytest.raises(ValueError):
        find_boundaries_by_top_k([1.0, 2.0], boundary_stride=0)


def test_token_zero_replaces_lowest_selected_boundary():
    values = np.zeros(100)
    values[10] = 5.0
    values[70] = 4.0
    values[90] = 3.0

    assert find_boundaries_by_top_k(values, boundary_stride=40) == [0, 10, 70]


def test_token_zero_is_not_duplicated_when_already_selected():
    values = np.zeros(100)
    values[0] = 10.0
    values[10] = 5.0
    values[70] = 4.0

    assert find_boundaries_by_top_k(values, boundary_stride=40) == [0, 10, 70]


def test_all_positions_returned_when_k_exceeds_length():
    assert find_boundaries_by_top_k([0.1, 0.2, 0.3], boundary_stride=1) == [0, 1, 2]
