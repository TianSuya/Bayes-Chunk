import math
from typing import Iterable, List, Optional

import numpy as np


def find_boundaries_by_top_k(
    surprisal_values: Iterable[float],
    boundary_stride: int = 40,
) -> List[int]:
    """Select high-surprisal boundaries while always preserving token 0.

    The number of returned boundaries is ``ceil(len(surprisal_values) / boundary_stride)``.
    When token 0 is not selected by surprisal, it replaces the selected non-zero boundary
    with the lowest surprisal, making the first-token boundary deterministic.
    """
    values = np.asarray(list(surprisal_values), dtype=float)
    n_tokens = len(values)
    if n_tokens == 0:
        return []
    if boundary_stride <= 0:
        raise ValueError("boundary_stride must be positive")

    k = max(1, math.ceil(n_tokens / boundary_stride))
    if k >= n_tokens:
        return list(range(n_tokens))

    selected = np.argsort(values)[-k:].tolist()
    selected_set = set(selected)

    if 0 not in selected_set:
        replace_idx: Optional[int] = min(selected_set, key=lambda idx: values[idx])
        selected_set.remove(replace_idx)
        selected_set.add(0)

    return sorted(selected_set)
