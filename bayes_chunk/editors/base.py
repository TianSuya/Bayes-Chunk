from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from bayes_chunk.algorithms.registry import get_algorithm_spec


@dataclass
class EditRequest:
    prompt: str
    target: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_legacy_record(self) -> Dict[str, Any]:
        record = dict(self.metadata)
        record.setdefault("question", self.prompt)
        record.setdefault("answer", self.target)
        return record


class Editor:
    def __init__(self, name: str, hparams: Any):
        self.name = name
        self.hparams = hparams
        self.spec = get_algorithm_spec(name)

    @classmethod
    def from_hparams_file(cls, name: str, hparams_path: Union[str, Path]) -> "Editor":
        spec = get_algorithm_spec(name)
        hparams = spec.hparams_cls.from_json(hparams_path)
        return cls(name, hparams)

    def edit(
        self,
        model,
        tokenizer,
        requests: EditRequest | Dict[str, Any] | List[EditRequest | Dict[str, Any]],
        *,
        ex_data: Optional[List[str]] = None,
        projector: Optional[torch.Tensor] = None,
        segmenter: Optional[Any] = None,
    ) -> Dict[str, torch.Tensor]:
        batch = self._normalize_requests(requests)
        kwargs: Dict[str, Any] = {}
        if self.spec.requires_ex_data:
            kwargs["ex_data"] = ex_data or []
        if self.spec.accepts_projector:
            kwargs["P"] = projector
        if self.spec.accepts_segmenter:
            kwargs["segmenter"] = segmenter
        return self.spec.apply_fn(model, tokenizer, self.hparams, batch, **kwargs)

    @staticmethod
    def _normalize_requests(
        requests: EditRequest | Dict[str, Any] | List[EditRequest | Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        items = requests if isinstance(requests, list) else [requests]
        normalized = []
        for item in items:
            if isinstance(item, EditRequest):
                normalized.append(item.to_legacy_record())
            else:
                normalized.append(dict(item))
        return normalized
