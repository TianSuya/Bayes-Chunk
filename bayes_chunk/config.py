from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union
import json

import yaml


@dataclass
class ModelConfig:
    name_or_path: str = ""
    tokenizer_name_or_path: Optional[str] = None
    torch_dtype: Optional[str] = None
    device: Optional[str] = None
    device_map: Optional[str] = None
    max_memory: Optional[Dict[str, str]] = None


@dataclass
class SegmentationConfig:
    type: str = "fixed"
    boundary_stride: int = 40
    window_size: int = 40
    overlap: int = 0
    min_segment_length: Optional[int] = None
    max_segment_length: Optional[int] = None
    include_last_token_boundary: bool = True
    verbose: bool = False


@dataclass
class AlgorithmConfig:
    name: str = "MEMIT_ARE"
    hparams_path: Optional[str] = None
    hparams: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    dataset: str = "editevery"
    data_path: Optional[str] = None
    limit: Optional[int] = None
    output_path: str = "outputs/results.json"
    metrics_output_path: Optional[str] = None
    max_new_tokens: int = 512
    temperature: float = 0.001


@dataclass
class BayesChunkConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BayesChunkConfig":
        return cls(
            model=ModelConfig(**data.get("model", {})),
            algorithm=AlgorithmConfig(**data.get("algorithm", {})),
            segmentation=SegmentationConfig(**data.get("segmentation", {})),
            evaluation=EvaluationConfig(**data.get("evaluation", {})),
        )


def load_config(path: Union[str, Path]) -> BayesChunkConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(handle) or {}
        else:
            data = json.load(handle)
    return BayesChunkConfig.from_dict(data)
