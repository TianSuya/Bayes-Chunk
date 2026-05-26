from dataclasses import dataclass
from typing import Callable, Dict, Type


@dataclass(frozen=True)
class AlgorithmSpec:
    hparams_cls: Type
    apply_fn: Callable
    requires_ex_data: bool = False
    accepts_projector: bool = False
    accepts_segmenter: bool = False


def _registry() -> Dict[str, AlgorithmSpec]:
    from bayes_chunk.algorithms.alphaedit import AlphaEditHyperParams, apply_AlphaEdit_to_model
    from bayes_chunk.algorithms.alphaedit_are import (
        AlphaEditAREHyperParams,
        apply_AlphaEdit_ARE_to_model,
    )
    from bayes_chunk.algorithms.memit import MEMITHyperParams, apply_memit_to_model
    from bayes_chunk.algorithms.memit_are import MEMITAREHyperParams, apply_memit_ARE_to_model
    from bayes_chunk.algorithms.unke import unkeHyperParams, apply_unke_to_model
    from bayes_chunk.algorithms.unke_are import unkeAREHyperParams, apply_unke_ARE_to_model

    return {
        "MEMIT": AlgorithmSpec(MEMITHyperParams, apply_memit_to_model),
        "MEMIT_ARE": AlgorithmSpec(
            MEMITAREHyperParams,
            apply_memit_ARE_to_model,
            accepts_segmenter=True,
        ),
        "AlphaEdit": AlgorithmSpec(
            AlphaEditHyperParams,
            apply_AlphaEdit_to_model,
            accepts_projector=True,
        ),
        "AlphaEdit_ARE": AlgorithmSpec(
            AlphaEditAREHyperParams,
            apply_AlphaEdit_ARE_to_model,
            accepts_projector=True,
            accepts_segmenter=True,
        ),
        "unke": AlgorithmSpec(unkeHyperParams, apply_unke_to_model, requires_ex_data=True),
        "unke_ARE": AlgorithmSpec(
            unkeAREHyperParams,
            apply_unke_ARE_to_model,
            requires_ex_data=True,
            accepts_segmenter=True,
        ),
    }


def get_algorithm_spec(name: str) -> AlgorithmSpec:
    registry = _registry()
    if name not in registry:
        raise ValueError(f"Unknown algorithm '{name}'. Available algorithms: {', '.join(sorted(registry))}")
    return registry[name]


def list_algorithms():
    return sorted(_registry())


def create_editor(name: str, hparams_path):
    from bayes_chunk.editors import Editor

    return Editor.from_hparams_file(name, hparams_path)
