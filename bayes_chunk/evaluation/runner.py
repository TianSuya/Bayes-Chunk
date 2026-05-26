from pathlib import Path
from typing import Any, Dict, List
import json

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from bayes_chunk.config import BayesChunkConfig
from bayes_chunk.datasets import get_dataset_class
from bayes_chunk.editors import Editor
from bayes_chunk.evaluation.metrics import evaluate_records
from bayes_chunk.segmentation import create_segmenter
from bayes_chunk.utils.globals import DATA_DIR
from bayes_chunk.utils import nethook


def _dtype_from_name(name: str | None):
    if name is None:
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch dtype '{name}'")
    return mapping[name]


def load_model_and_tokenizer(config: BayesChunkConfig):
    model_kwargs: Dict[str, Any] = {}
    dtype = _dtype_from_name(config.model.torch_dtype)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if config.model.device_map is not None:
        model_kwargs["device_map"] = config.model.device_map
    if config.model.max_memory is not None:
        model_kwargs["max_memory"] = config.model.max_memory

    model = AutoModelForCausalLM.from_pretrained(config.model.name_or_path, **model_kwargs)
    if config.model.device and config.model.device_map is None:
        model = model.to(config.model.device)
    tokenizer_name = config.model.tokenizer_name_or_path or config.model.name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def run_evaluation(config: BayesChunkConfig) -> List[Dict[str, Any]]:
    model, tokenizer = load_model_and_tokenizer(config)
    if not config.algorithm.hparams_path:
        raise ValueError("evaluation requires algorithm.hparams_path")
    editor = Editor.from_hparams_file(config.algorithm.name, config.algorithm.hparams_path)
    segmenter = create_segmenter(config.segmentation)
    dataset_cls = get_dataset_class(config.evaluation.dataset)
    dataset = dataset_cls(
        config.evaluation.data_path or DATA_DIR,
        model_name=editor.hparams.model_name,
        size=config.evaluation.limit,
    )

    results: List[Dict[str, Any]] = []
    for index in tqdm(range(len(dataset))):
        item = dataset[index]
        record = dict(item)
        weights_copy = editor.edit(model, tokenizer, record, segmenter=segmenter)
        generated = _generate(model, tokenizer, record["question"], config)
        record["original_prediction"] = generated
        results.append(record)
        with torch.no_grad():
            for key, value in weights_copy.items():
                nethook.get_parameter(model, key)[...] = value.to(next(model.parameters()).device)

    output_path = Path(config.evaluation.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    if config.evaluation.metrics_output_path:
        metrics = evaluate_records(results)
        metrics_path = Path(config.evaluation.metrics_output_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def _generate(model, tokenizer, prompt: str, config: BayesChunkConfig) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=config.evaluation.temperature,
            max_new_tokens=config.evaluation.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0][len(inputs["input_ids"][0]) :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)
