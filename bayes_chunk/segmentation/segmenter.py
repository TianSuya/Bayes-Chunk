from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .boundaries import find_boundaries_by_top_k
from .surprisal import compute_surprisal
from .types import Segment, SegmentationResult


@dataclass
class BayesSegmenter:
    boundary_stride: int = 40
    min_segment_length: Optional[int] = None
    max_segment_length: Optional[int] = None
    include_last_token_boundary: bool = True
    verbose: bool = False

    def boundaries_from_surprisal(self, surprisal_values: Iterable[float]) -> List[int]:
        return find_boundaries_by_top_k(
            surprisal_values,
            boundary_stride=self.boundary_stride,
        )

    def segment_target_ids(
        self,
        target_ids: torch.Tensor,
        *,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prompt: str,
        target: str,
        sep: str = " ",
    ) -> List[Segment]:
        surprisal, _, _ = compute_surprisal(model, tokenizer, prompt, target, sep=sep)
        boundaries = self.boundaries_from_surprisal(surprisal)
        return self.segment_token_ids(target_ids.detach().cpu().tolist(), boundaries, tokenizer=None)

    def segment_token_ids(
        self,
        token_ids: Iterable[int],
        boundaries: Iterable[int],
        tokenizer: Optional[AutoTokenizer] = None,
    ) -> List[Segment]:
        ids = list(token_ids)
        n_tokens = len(ids)
        boundary_positions = sorted({idx for idx in boundaries if 0 <= idx <= n_tokens})
        if not boundary_positions or boundary_positions[0] != 0:
            boundary_positions = [0] + boundary_positions

        if self.include_last_token_boundary and n_tokens > 0 and (n_tokens - 1) not in boundary_positions:
            boundary_positions.append(n_tokens - 1)
            boundary_positions = sorted(set(boundary_positions))

        raw_segments: List[Segment] = []
        for i, start in enumerate(boundary_positions):
            end = boundary_positions[i + 1] if i + 1 < len(boundary_positions) else n_tokens
            if start < end:
                seg_ids = ids[start:end]
                text = tokenizer.decode(seg_ids) if tokenizer is not None else None
                raw_segments.append(Segment(start=start, end=end, token_ids=seg_ids, text=text))

        segments = self._merge_short_segments(raw_segments, tokenizer)
        return self._split_long_segments(segments, tokenizer)

    def segment(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prompt: str,
        target: str,
        sep: str = " ",
    ) -> SegmentationResult:
        surprisal, tokens, token_ids = compute_surprisal(model, tokenizer, prompt, target, sep=sep)
        boundaries = self.boundaries_from_surprisal(surprisal)
        segments = self.segment_token_ids(token_ids, boundaries, tokenizer)
        return SegmentationResult(
            boundaries=[segment.start for segment in segments],
            segments=segments,
            surprisal=surprisal,
            tokens=tokens,
            token_ids=token_ids,
        )

    def segment_tensor(
        self,
        target_ids: torch.Tensor,
        boundaries: Iterable[int],
    ) -> List[torch.Tensor]:
        ids = target_ids.detach().cpu().tolist()
        return [
            target_ids[segment.start : segment.end]
            for segment in self.segment_token_ids(ids, boundaries)
        ]

    def _merge_short_segments(
        self,
        segments: List[Segment],
        tokenizer: Optional[AutoTokenizer],
    ) -> List[Segment]:
        if not self.min_segment_length or self.min_segment_length <= 1:
            return segments
        merged: List[Segment] = []
        for segment in segments:
            if merged and segment.length < self.min_segment_length:
                prev = merged.pop()
                token_ids = prev.token_ids + segment.token_ids
                text = tokenizer.decode(token_ids) if tokenizer is not None else None
                merged.append(Segment(prev.start, segment.end, token_ids, text))
            else:
                merged.append(segment)
        return merged

    def _split_long_segments(
        self,
        segments: List[Segment],
        tokenizer: Optional[AutoTokenizer],
    ) -> List[Segment]:
        if not self.max_segment_length or self.max_segment_length <= 0:
            return segments
        split_segments: List[Segment] = []
        for segment in segments:
            for offset in range(0, segment.length, self.max_segment_length):
                token_ids = segment.token_ids[offset : offset + self.max_segment_length]
                start = segment.start + offset
                end = start + len(token_ids)
                text = tokenizer.decode(token_ids) if tokenizer is not None else None
                split_segments.append(Segment(start, end, token_ids, text))
        return split_segments


@dataclass
class FixedSegmenter:
    window_size: int
    overlap: int = 0

    def __post_init__(self):
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.overlap < 0:
            raise ValueError("overlap must be non-negative")
        if self.overlap >= self.window_size:
            raise ValueError("overlap must be smaller than window_size")

    @classmethod
    def from_hparams(cls, hparams) -> "FixedSegmenter":
        return cls(window_size=hparams.window_size, overlap=getattr(hparams, "overlap", 0))

    def segment_target_ids(
        self,
        target_ids: torch.Tensor,
        **_,
    ) -> List[Segment]:
        ids = target_ids.detach().cpu().tolist()
        segments: List[Segment] = []
        start = 0
        stride = self.window_size - self.overlap
        while start < len(ids):
            end = min(start + self.window_size, len(ids))
            segments.append(Segment(start=start, end=end, token_ids=ids[start:end]))
            start += stride
        return segments

    def segment(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prompt: str,
        target: str,
        sep: str = " ",
    ) -> SegmentationResult:
        del model, prompt, sep
        token_ids = tokenizer(target, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        segments = self.segment_target_ids(token_ids)
        return SegmentationResult(
            boundaries=[segment.start for segment in segments],
            segments=segments,
            token_ids=token_ids.detach().cpu().tolist(),
        )
