import torch

from bayes_chunk.segmentation import BayesSegmenter, FixedSegmenter


def test_segment_token_ids_preserves_all_tokens():
    segmenter = BayesSegmenter(include_last_token_boundary=False)
    segments = segmenter.segment_token_ids(list(range(10)), [0, 3, 7])

    assert [token for segment in segments for token in segment.token_ids] == list(range(10))
    assert [(segment.start, segment.end) for segment in segments] == [(0, 3), (3, 7), (7, 10)]


def test_segment_token_ids_inserts_zero_boundary():
    segmenter = BayesSegmenter(include_last_token_boundary=False)
    segments = segmenter.segment_token_ids(list(range(6)), [3])

    assert [(segment.start, segment.end) for segment in segments] == [(0, 3), (3, 6)]


def test_include_last_token_boundary_splits_tail_token():
    segmenter = BayesSegmenter(include_last_token_boundary=True)
    segments = segmenter.segment_token_ids(list(range(5)), [0, 2])

    assert [(segment.start, segment.end) for segment in segments] == [(0, 2), (2, 4), (4, 5)]


def test_fixed_segmenter_matches_window_overlap():
    segmenter = FixedSegmenter(window_size=4, overlap=1)
    segments = segmenter.segment_target_ids(torch.arange(10))

    assert [(segment.start, segment.end) for segment in segments] == [(0, 4), (3, 7), (6, 10), (9, 10)]
    assert [segment.token_ids for segment in segments] == [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9], [9]]
