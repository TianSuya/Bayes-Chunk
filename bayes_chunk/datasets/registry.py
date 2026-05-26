from typing import Dict, Type


def _registry() -> Dict[str, Type]:
    from bayes_chunk.datasets import CounterFactDataset, EditeveryDataset, FakeDataset, MQUAKEDataset, QWQDataset, UnKEDataset

    return {
        "cf": CounterFactDataset,
        "counterfact": CounterFactDataset,
        "editevery": EditeveryDataset,
        "fake": FakeDataset,
        "mquake": MQUAKEDataset,
        "qwq": QWQDataset,
        "unke": UnKEDataset,
    }


def list_datasets():
    return sorted(_registry())


def get_dataset_class(name: str):
    registry = _registry()
    if name not in registry:
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {', '.join(sorted(registry))}")
    return registry[name]
