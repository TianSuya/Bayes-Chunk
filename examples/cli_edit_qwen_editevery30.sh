#!/usr/bin/env bash
set -euo pipefail

# CLI example: run 30 EditEvery edits with Qwen2.5-7B-Instruct and Bayes
# segmentation. The YAML config contains the model path, algorithm hparams,
# dataset path, output path, and metrics output path.

# Select the visible GPU. Override at runtime, for example:
#   GPU_ID=0 bash examples/cli_edit_qwen_editevery30.sh
GPU_ID="${GPU_ID:-1}"

# Use a specific Python interpreter or conda environment if needed:
#   PYTHON_BIN=/path/to/env/bin/python bash examples/cli_edit_qwen_editevery30.sh
PYTHON_BIN="${PYTHON_BIN:-python}"

# Swap this config to test another model, algorithm, or segmentation strategy.
CONFIG="${CONFIG:-bayes_chunk/configs/qwen25_memit_are_bayes_editevery30.yaml}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# The evaluate command performs editing, generation, result writing, and metric
# writing according to the config.
"${PYTHON_BIN}" -m bayes_chunk.cli evaluate --config "${CONFIG}"
