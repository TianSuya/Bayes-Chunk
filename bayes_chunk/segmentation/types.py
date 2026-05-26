from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    token_ids: List[int]
    text: Optional[str] = None

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class SegmentationResult:
    boundaries: List[int]
    segments: List[Segment]
    surprisal: Optional[np.ndarray] = field(default=None, repr=False)
    tokens: Optional[List[str]] = None
    token_ids: Optional[List[int]] = None
