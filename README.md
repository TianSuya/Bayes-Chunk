# AnyEdit++

This repository contains the implementation of **AnyEdit++**, a framework for efficient and reliable model editing. We provide implementations for our proposed method **MEMIT_ARE_bayes** along with several baseline methods including AlphaEdit, MEMIT, and UnKE.

## Environment Setup

To set up the environment, please install the required dependencies:

```bash
pip install torch transformers numpy
# Add other dependencies as needed
```

## Data Preparation

Ensure your datasets are placed in the `data/` directory. The project supports the following datasets:
- `unke` (UnKE)
- `cf` (CounterFact)
- `mquake` (MQuAKE)
- `qwq` (QWQ)
- `editevery`

## Running the Methods

We provide a convenient shell script `run.sh` to execute the editing experiments. You can configure the parameters inside the script or pass them as command-line arguments.

### Running MEMIT_ARE_bayes (Proposed Method)

To run our proposed method `MEMIT_ARE_bayes`, use the following command:

```bash
./run.sh \
    --alg_name=MEMIT_ARE_bayes \
    --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
    --hparams_fname=Llama3-8B-Instruct.json \
    --ds_name=unke \
    --dataset_size_limit=1000 \
    --num_edits=1
```

**Parameters:**
- `--alg_name`: The algorithm to use (`MEMIT_ARE_bayes`, `MEMIT_ARE`, `MEMIT`, `AlphaEdit`, `AlphaEdit_ARE`, `unke`, `unke_ARE`).
- `--model_name`: The path or name of the model to edit.
- `--hparams_fname`: The hyperparameter file located in `hparams/<alg_name>/`.
- `--ds_name`: The dataset name.
- `--dataset_size_limit`: Number of samples to use.
- `--num_edits`: Number of sequential edits (batch size).

### Running Baseline Methods

#### AlphaEdit

```bash
./run.sh \
    --alg_name=AlphaEdit \
    --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
    --hparams_fname=Llama3-8B-Instruct.json \
    --ds_name=unke
```

#### MEMIT

```bash
./run.sh \
    --alg_name=MEMIT \
    --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
    --hparams_fname=Llama3-8B-Instruct.json \
    --ds_name=cf
```

#### UnKE

```bash
./run.sh \
    --alg_name=unke \
    --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
    --hparams_fname=Llama3-8B-Instruct.json \
    --ds_name=unke
```

### Logs

The execution logs will be saved in the `experiments/logs/` directory.
