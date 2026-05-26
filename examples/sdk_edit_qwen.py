import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

# This example intentionally uses the public Bayes-Chunk SDK instead of calling
# algorithm internals. It is the recommended pattern when you want to embed
# Bayes-Chunk in your own experiment scripts.
from bayes_chunk import BayesSegmenter, create_editor
from bayes_chunk.config import load_config
from bayes_chunk.datasets import get_dataset_class
from bayes_chunk.evaluation import evaluate_records, load_model_and_tokenizer
from bayes_chunk.utils import nethook
from bayes_chunk.utils.globals import DATA_DIR


def generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float) -> str:
    """Generate the edited model's answer for one evaluation prompt."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    do_sample = temperature > 0
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0][len(inputs["input_ids"][0]) :]
    return tokenizer.decode(generated_ids, skip_special_tokens=False)


def restore_weights(model, weights_copy):
    """Restore original parameters after each edit.

    MEMIT-style editors return a copy of the parameters they changed. Restoring
    them keeps each dataset record independent, which is usually what you want
    for editing benchmarks.
    """
    device = next(model.parameters()).device
    with torch.no_grad():
        for key, value in weights_copy.items():
            nethook.get_parameter(model, key)[...] = value.to(device)


def main() -> int:
    parser = argparse.ArgumentParser(description="SDK example: edit Qwen on EditEvery with Bayes segmentation.")
    parser.add_argument("--config", default="bayes_chunk/configs/qwen25_memit_are_bayes_editevery30.yaml")
    parser.add_argument("--dataset", default="editevery", choices=["editevery", "qwq"])
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output", default="outputs/sdk_qwen_editevery30_results.json")
    parser.add_argument("--metrics-output", default="outputs/sdk_qwen_editevery30_metrics.json")
    args = parser.parse_args()

    # The config owns model path, algorithm hparams, segmentation options, and
    # generation settings. Override CLI flags only for fields that are useful
    # when trying the example quickly, such as dataset and limit.
    config = load_config(args.config)

    # Load the actual Hugging Face model/tokenizer. For the provided Qwen config
    # this expects a local Qwen2.5-7B-Instruct checkpoint path.
    model, tokenizer = load_model_and_tokenizer(config)

    # Build an editor from the algorithm registry. The same code can load
    # MEMIT_ARE, AlphaEdit_ARE, or future registered algorithms by changing YAML.
    editor = create_editor(config.algorithm.name, config.algorithm.hparams_path)

    # Bayes segmentation is injected into the editor instead of being hard-coded
    # in the algorithm. This is the main SDK extension point.
    segmenter = BayesSegmenter(
        boundary_stride=config.segmentation.boundary_stride,
        include_last_token_boundary=config.segmentation.include_last_token_boundary,
        min_segment_length=config.segmentation.min_segment_length,
        max_segment_length=config.segmentation.max_segment_length,
        verbose=config.segmentation.verbose,
    )

    # Dataset classes normalize records into the shape expected by editors:
    # question/prompt text plus the desired answer/target text.
    dataset_cls = get_dataset_class(args.dataset)
    dataset = dataset_cls(
        config.evaluation.data_path or DATA_DIR,
        model_name=editor.hparams.model_name,
        size=args.limit,
    )

    results = []
    for index in tqdm(range(len(dataset)), desc="Editing"):
        record = dict(dataset[index])

        # Apply one edit, evaluate the edited model, then restore model weights.
        # This mirrors standard per-record knowledge editing evaluation.
        weights_copy = editor.edit(model, tokenizer, record, segmenter=segmenter)
        record["original_prediction"] = generate(
            model,
            tokenizer,
            record["question"],
            max_new_tokens=config.evaluation.max_new_tokens,
            temperature=config.evaluation.temperature,
        )
        results.append(record)
        restore_weights(model, weights_copy)

    # Store full per-record outputs so metrics can be recomputed later without
    # rerunning expensive model editing.
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # Compute lightweight text metrics from the package evaluation module.
    metrics = evaluate_records(results)
    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
