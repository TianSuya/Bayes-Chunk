<p align="center">
  <img src="assets/bayes-chunk.png" alt="Bayes-Chunk" width="70%">
</p>

<h1 align="center">Bayes-Chunk</h1>

<p align="center">
  <b>Plug-and-play Bayes segmentation for long-form knowledge editing.</b>
</p>

<p align="center">
  Official implementation of <b>AnyEdit++: Adaptive Long-Form Knowledge Editing via Bayesian Surprise</b>.
</p>

<p align="center">
  <a href="https://tiansuya.github.io/Bayes-Chunk/"><img alt="Website" src="https://img.shields.io/badge/Website-GitHub%20Pages-111111"></a>
  <a href="https://openreview.net/pdf?id=W6qfbvysDh"><img alt="Paper" src="https://img.shields.io/badge/Paper-ICML%202026-34a853"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-3776ab">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-Ready-ee4c2c">
  <img alt="SDK" src="https://img.shields.io/badge/Usage-SDK%20%7C%20CLI-7b61ff">
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#highlights">Highlights</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="https://tiansuya.github.io/Bayes-Chunk/">Project Page</a> ·
  <a href="#examples">Examples</a> ·
  <a href="#configuration">Configuration</a>
</p>

<p align="center">
  <b>English</b> | <a href="README.zh-CN.md">简体中文</a>
</p>

---

## Overview

**Bayes-Chunk** is a research-oriented toolkit for long-form knowledge editing.
It decouples **editing algorithms** from **answer segmentation**, making Bayes
segmentation a reusable module that can be injected into different editing
methods such as MEMIT, AlphaEdit, and UnKE.

> Core idea: keep the editing algorithm stable, make segmentation adaptive.

## Highlights

| Capability | Description |
| --- | --- |
| **Plug-and-play segmentation** | Use Bayes or fixed segmentation through the same editor interface. |
| **Algorithm registry** | Built-in support for `MEMIT`, `MEMIT_ARE`, `AlphaEdit`, `AlphaEdit_ARE`, `UnKE`, and `UnKE_ARE`. |
| **Long-form datasets** | Dataset loaders for EditEvery, QwQ, MQuAKE, CounterFact-style data, and related benchmarks. |
| **SDK + CLI workflows** | Use the Python API in custom experiments or run reproducible YAML-driven CLI jobs. |
| **Config-first experiments** | Model path, editor, segmentation, dataset, output path, and metrics are all controlled by YAML. |

### Design

```text
                    +-----------------------------+
                    |        Bayes-Chunk          |
                    +-----------------------------+
                         /                   \
                        /                     \
        +-------------------------+   +--------------------------+
        |   Editing Algorithms    |   |      Segmentation        |
        | MEMIT / AlphaEdit / ... |   | Bayes / Fixed / Future   |
        +-------------------------+   +--------------------------+
                        \                     /
                         \                   /
                    +-----------------------------+
                    |   Dataset Evaluation Flow   |
                    | edit -> generate -> metric  |
                    +-----------------------------+
```


## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/TianSuya/Bayes-Chunk.git
cd Bayes-Chunk
pip install -e ".[dev,eval]"
```

Runtime-only installation:

```bash
pip install -e .
```

Recommended environment:

| Dependency | Recommendation |
| --- | --- |
| Python | `>=3.9` |
| PyTorch | CUDA build for local GPU editing |
| Transformers | Compatible with your target model checkpoint |
| GPU memory | Depends on model size and covariance/statistics cache usage |

## Quick Start

Validate a config without loading a large model:

```bash
python -m bayes_chunk.cli evaluate \
  --config bayes_chunk/configs/qwen25_memit_are_bayes_editevery30.yaml \
  --dry-run
```

Run 30 EditEvery edits with Qwen and Bayes segmentation:

```bash
GPU_ID=0 PYTHON_BIN=/path/to/python bash examples/cli_edit_qwen_editevery30.sh
```

Run the same workflow through the SDK:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/sdk_edit_qwen.py \
  --config bayes_chunk/configs/qwen25_memit_are_bayes_editevery30.yaml \
  --dataset editevery \
  --limit 30
```

Both workflows write:

```text
outputs/cli_qwen_editevery30_results.json
outputs/cli_qwen_editevery30_metrics.json
```

## Examples

Bayes-Chunk provides two first-class usage styles.

### Python SDK

Use this style when you are writing a custom experiment script, integrating
Bayes-Chunk into another pipeline, or programmatically swapping algorithms.

```python
from bayes_chunk import BayesSegmenter, create_editor
from bayes_chunk.config import load_config
from bayes_chunk.evaluation import load_model_and_tokenizer

config = load_config("bayes_chunk/configs/qwen25_memit_are_bayes_editevery30.yaml")
model, tokenizer = load_model_and_tokenizer(config)

editor = create_editor(config.algorithm.name, config.algorithm.hparams_path)
segmenter = BayesSegmenter(boundary_stride=40)

record = {
    "question": "Question text",
    "answer": "Target answer text",
}

weights_copy = editor.edit(model, tokenizer, record, segmenter=segmenter)
```

Full file: [`examples/sdk_edit_qwen.py`](examples/sdk_edit_qwen.py)

### Command Line

Use this style when you want reproducible experiments driven by YAML configs.

```bash
python -m bayes_chunk.cli list-algorithms

python -m bayes_chunk.cli segment \
  --config bayes_chunk/configs/qwen25_memit_are_bayes.yaml \
  --prompt "Question text" \
  --target "Target answer text"

python -m bayes_chunk.cli evaluate \
  --config bayes_chunk/configs/qwen25_memit_are_bayes_editevery30.yaml
```

Full file: [`examples/cli_edit_qwen_editevery30.sh`](examples/cli_edit_qwen_editevery30.sh)

## Configuration

Experiments are controlled by YAML files under
[`bayes_chunk/configs`](bayes_chunk/configs).

```yaml
model:
  name_or_path: /path/to/Qwen2.5-7B-Instruct
  tokenizer_name_or_path: /path/to/Qwen2.5-7B-Instruct
  torch_dtype: bfloat16
  device: cuda

algorithm:
  name: MEMIT_ARE
  hparams_path: hparams/MEMIT_ARE/Qwen2.5-7B-Instruct.json

segmentation:
  type: bayes
  boundary_stride: 40
  include_last_token_boundary: true

evaluation:
  dataset: editevery
  data_path: data
  limit: 30
  output_path: outputs/cli_qwen_editevery30_results.json
  metrics_output_path: outputs/cli_qwen_editevery30_metrics.json
  max_new_tokens: 512
  temperature: 0.001
```

### Swap Segmentation

Bayes segmentation:

```yaml
segmentation:
  type: bayes
  boundary_stride: 40
  include_last_token_boundary: true
```

Fixed-window segmentation:

```yaml
segmentation:
  type: fixed
  window_size: 40
  overlap: 0
```

The editor call stays the same. Only the segmentation module changes.

## Data And Model Files

Dataset files are expected under `data/` unless `evaluation.data_path` points
elsewhere:

```text
data/editevery.json
data/qwq.json
```

Large model checkpoints and covariance statistics are not bundled. Point
`model.name_or_path` to a local Hugging Face checkpoint, and keep statistics
caches in the layout expected by the MEMIT utilities:

```text
data/stats/<model-name>/wikipedia_stats/
```

## Evaluation

`bayes_chunk.evaluation` provides lightweight text metrics:

| Metric | Meaning |
| --- | --- |
| `bleu` | n-gram overlap with smoothing |
| `rouge_l` | longest-common-subsequence F1 |
| `token_f1` | token-level overlap F1 |
| `exact_match` | normalized exact string match |

Recompute metrics from saved results:

```python
from bayes_chunk.evaluation import evaluate_results_file

metrics = evaluate_results_file(
    "outputs/cli_qwen_editevery30_results.json",
    "outputs/cli_qwen_editevery30_metrics.json",
)
```

## Repository Layout

```text
bayes_chunk/
  algorithms/     # Editing algorithms and registry
  segmentation/   # Fixed and Bayes segmenters
  datasets/       # Dataset loaders and registry
  evaluation/     # Model loading, generation, and metrics
  editors/        # Common editor interface
  utils/          # Shared model and statistics utilities
  configs/        # Example YAML configs
examples/         # SDK and CLI usage examples
hparams/          # Algorithm hyperparameter files
assets/           # Logo and project assets
tests/            # Unit and CLI smoke tests
```

## Development

Run tests:

```bash
pytest -q
```

Run syntax checks:

```bash
python -m py_compile $(find bayes_chunk examples tests -name "*.py")
```

Development guidelines:

- keep segmentation outside algorithm internals;
- add new editing methods through `bayes_chunk.algorithms`;
- add new segmentation methods through `bayes_chunk.segmentation`;
- keep examples runnable and configs explicit.
