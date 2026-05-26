from bayes_chunk.config import SegmentationConfig

from .segmenter import BayesSegmenter, FixedSegmenter


def create_segmenter(config: SegmentationConfig):
    if config.type == "bayes":
        return BayesSegmenter(
            boundary_stride=config.boundary_stride,
            min_segment_length=config.min_segment_length,
            max_segment_length=config.max_segment_length,
            include_last_token_boundary=config.include_last_token_boundary,
            verbose=config.verbose,
        )
    if config.type == "fixed":
        return FixedSegmenter(window_size=config.window_size, overlap=config.overlap)
    raise ValueError("segmentation.type must be either 'fixed' or 'bayes'")
