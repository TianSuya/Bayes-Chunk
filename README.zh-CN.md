<p align="center">
  <img src="assets/bayes-chunk.png" alt="Bayes-Chunk" width="70%">
</p>

<h1 align="center">Bayes-Chunk</h1>

<p align="center">
  <b>面向长文本知识编辑的即插即用 Bayes 分段库。</b>
</p>

<p align="center">
  论文 <b>AnyEdit++: Adaptive Long-Form Knowledge Editing via Bayesian Surprise</b> 的官方代码实现。
</p>

<p align="center">
  <a href="https://tiansuya.github.io/Bayes-Chunk/"><img alt="Website" src="https://img.shields.io/badge/Website-GitHub%20Pages-111111"></a>
  <a href="https://openreview.net/pdf?id=W6qfbvysDh"><img alt="Paper" src="https://img.shields.io/badge/Paper-ICML%202026-34a853"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-3776ab">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-Ready-ee4c2c">
  <img alt="SDK" src="https://img.shields.io/badge/Usage-SDK%20%7C%20CLI-7b61ff">
</p>

<p align="center">
  <a href="#项目概览">项目概览</a> ·
  <a href="#核心特性">核心特性</a> ·
  <a href="#安装">安装</a> ·
  <a href="#快速开始">快速开始</a> ·
  <a href="https://tiansuya.github.io/Bayes-Chunk/">项目主页</a> ·
  <a href="#使用示例">使用示例</a> ·
  <a href="#配置文件">配置文件</a>
</p>

<p align="center">
  <a href="README.md">English</a> | <b>简体中文</b>
</p>

---

## 项目概览

**Bayes-Chunk** 是一个面向长文本知识编辑的研究型工具库。它将
**编辑算法** 与 **答案分段策略** 解耦，使 Bayes 分段可以作为可复用模块注入到
MEMIT、AlphaEdit、UnKE 等不同编辑方法中。

> 核心思路：保持编辑算法稳定，让分段策略自适应。

## 核心特性

| 能力 | 说明 |
| --- | --- |
| **即插即用分段** | 通过同一个 editor 接口切换 Bayes 分段或固定分段。 |
| **算法注册机制** | 内置支持 `MEMIT`、`MEMIT_ARE`、`AlphaEdit`、`AlphaEdit_ARE`、`UnKE`、`UnKE_ARE`。 |
| **长文本数据集** | 提供 EditEvery、QwQ、MQuAKE、CounterFact 风格数据等加载器。 |
| **SDK + CLI** | 既可以在 Python 实验脚本中调用，也可以通过 YAML 配置运行可复现实验。 |
| **配置优先** | 模型路径、编辑算法、分段策略、数据集、输出路径和指标均由 YAML 控制。 |

### 设计结构

```text
                    +-----------------------------+
                    |        Bayes-Chunk          |
                    +-----------------------------+
                         /                   \
                        /                     \
        +-------------------------+   +--------------------------+
        |     编辑算法模块        |   |        分段模块          |
        | MEMIT / AlphaEdit / ... |   | Bayes / Fixed / Future   |
        +-------------------------+   +--------------------------+
                        \                     /
                         \                   /
                    +-----------------------------+
                    |       数据集评估流程        |
                    | edit -> generate -> metric  |
                    +-----------------------------+
```

Bayes 边界选择始终包含 token 位置 `0`。旧版本中“从前 N 个 token 中强制选择
一个边界”的超参数已经移除，这样可以稳定第一个编辑片段，同时简化配置。

## 安装

克隆仓库并以 editable 模式安装：

```bash
git clone https://github.com/TianSuya/Bayes-Chunk.git
cd Bayes-Chunk
pip install -e ".[dev,eval]"
```

如果只需要运行库：

```bash
pip install -e .
```

推荐环境：

| 依赖 | 建议 |
| --- | --- |
| Python | `>=3.9` |
| PyTorch | 使用支持本地 GPU 的 CUDA 版本 |
| Transformers | 与目标模型 checkpoint 兼容 |
| GPU 显存 | 取决于模型规模和 covariance/statistics cache 使用情况 |

## 快速开始

不加载大模型，仅验证配置：

```bash
python -m bayes_chunk.cli evaluate \
  --config bayes_chunk/configs/qwen25_memit_are_bayes_editevery30.yaml \
  --dry-run
```

使用 Qwen 和 Bayes 分段运行 30 条 EditEvery 编辑：

```bash
GPU_ID=0 PYTHON_BIN=/path/to/python bash examples/cli_edit_qwen_editevery30.sh
```

通过 SDK 运行同样流程：

```bash
CUDA_VISIBLE_DEVICES=0 python examples/sdk_edit_qwen.py \
  --config bayes_chunk/configs/qwen25_memit_are_bayes_editevery30.yaml \
  --dataset editevery \
  --limit 30
```

两个流程都会写出：

```text
outputs/cli_qwen_editevery30_results.json
outputs/cli_qwen_editevery30_metrics.json
```

## 使用示例

Bayes-Chunk 提供两种主要使用方式。

### Python SDK

当你需要写自定义实验脚本、接入已有 pipeline 或动态切换算法时，建议使用 SDK。

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

完整文件见 [`examples/sdk_edit_qwen.py`](examples/sdk_edit_qwen.py)。

### 命令行

当你希望通过 YAML 配置运行可复现实验时，建议使用 CLI。

```bash
python -m bayes_chunk.cli list-algorithms

python -m bayes_chunk.cli segment \
  --config bayes_chunk/configs/qwen25_memit_are_bayes.yaml \
  --prompt "Question text" \
  --target "Target answer text"

python -m bayes_chunk.cli evaluate \
  --config bayes_chunk/configs/qwen25_memit_are_bayes_editevery30.yaml
```

完整文件见 [`examples/cli_edit_qwen_editevery30.sh`](examples/cli_edit_qwen_editevery30.sh)。

## 配置文件

实验配置位于 [`bayes_chunk/configs`](bayes_chunk/configs)。

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

### 切换分段策略

Bayes 分段：

```yaml
segmentation:
  type: bayes
  boundary_stride: 40
  include_last_token_boundary: true
```

固定窗口分段：

```yaml
segmentation:
  type: fixed
  window_size: 40
  overlap: 0
```

编辑器调用方式保持不变，只需要替换分段模块。

## 数据和模型文件

默认情况下，数据文件放在 `data/` 下，除非通过 `evaluation.data_path` 指定其他目录：

```text
data/editevery.json
data/qwq.json
```

大模型 checkpoint 和 covariance statistics 不会随仓库打包。请将
`model.name_or_path` 指向本地 Hugging Face checkpoint，并按照 MEMIT 工具期望的
结构放置 statistics cache：

```text
data/stats/<model-name>/wikipedia_stats/
```

## 评估

`bayes_chunk.evaluation` 提供轻量文本指标：

| 指标 | 含义 |
| --- | --- |
| `bleu` | 带平滑的 n-gram 重叠 |
| `rouge_l` | 最长公共子序列 F1 |
| `token_f1` | token 级重叠 F1 |
| `exact_match` | 归一化后的精确匹配 |

从已保存结果中重新计算指标：

```python
from bayes_chunk.evaluation import evaluate_results_file

metrics = evaluate_results_file(
    "outputs/cli_qwen_editevery30_results.json",
    "outputs/cli_qwen_editevery30_metrics.json",
)
```

## 目录结构

```text
bayes_chunk/
  algorithms/     # 编辑算法和注册表
  segmentation/   # 固定分段和 Bayes 分段
  datasets/       # 数据集加载器和注册表
  evaluation/     # 模型加载、生成和指标
  editors/        # 通用 editor 接口
  utils/          # 模型与 statistics 工具
  configs/        # 示例 YAML 配置
examples/         # SDK 和 CLI 使用示例
hparams/          # 算法超参数文件
assets/           # Logo 和项目资源
tests/            # 单元测试和 CLI smoke tests
```

## 开发

运行测试：

```bash
pytest -q
```

运行语法检查：

```bash
python -m py_compile $(find bayes_chunk examples tests -name "*.py")
```

开发约定：

- 分段逻辑应保持在算法内部之外；
- 新编辑方法通过 `bayes_chunk.algorithms` 注册；
- 新分段方法通过 `bayes_chunk.segmentation` 注册；
- examples 应保持可运行，configs 应保持明确。
