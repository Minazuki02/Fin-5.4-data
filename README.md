# Fin-R1-Data

**Financial Reasoning Training Data Construction Pipeline**
**金融推理训练数据构建流水线**

This repository contains the complete data construction pipeline for building training data for a financial reasoning model, inspired by [Fin-R1](https://arxiv.org/abs/2503.07832). Since Fin-R1's code and training data are not open-sourced, this project replicates the data construction methodology from scratch using 7 open-source financial datasets, GPT-5.4 reasoning chain distillation, iterative prompt optimization, and two-stage quality filtering.

本仓库包含金融推理模型训练数据的完整构建流水线。由于 Fin-R1 的代码和训练数据未开源，本项目使用 7 个开源金融数据集从零复现其数据构建方法，采用 GPT-5.4 推理链蒸馏、逐数据集 Prompt 迭代优化，以及两阶段质量过滤。

---

## Pipeline Architecture / 流水线架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA CONSTRUCTION PIPELINE                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  7 Open Datasets ──► Schema Normalization ──► Train/Val/Test Split  │
│  (HuggingFace,       (unified fields,        (8:1:1 for datasets   │
│   ModelScope,         table/multi-turn         missing splits)      │
│   GitHub)             special handling)                              │
│                                                                     │
│       ┌─────────────────────────────────────────────┐               │
│       │         PROMPT ITERATION LOOP               │               │
│       │                                             │               │
│       │  ┌──► 500-sample pilot distill (GPT-5.4)   │               │
│       │  │         │                                │               │
│       │  │    compare with ground-truth             │               │
│       │  │         │                                │               │
│       │  │    badcase analysis & categorization     │               │
│       │  │         │                                │               │
│       │  │    optimize prompt (per dataset)         │               │
│       │  │         │                                │               │
│       │  └────── iterate until convergence ◄────────│               │
│       └─────────────────────────────────────────────┘               │
│                          │                                          │
│                   Full distillation                                 │
│                          │                                          │
│              Two-Stage Quality Filter                               │
│              ├── Stage 1: Answer correctness (rule + LLM judge)     │
│              └── Stage 2: Reasoning quality (LLM judge)             │
│                          │                                          │
│              Domain-weighted data mixing                            │
│                          │                                          │
│              Final SFT dataset (42,612 train / 5,380 val)           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Datasets / 数据集来源

| Dataset | Type / 类型 | Language | Samples | Source |
|---------|------------|----------|---------|--------|
| [FinQA](https://huggingface.co/datasets/ibm-research/finqa) | Financial numerical reasoning / 金融数值推理 | EN | 8,281 | IBM Research |
| [ConvFinQA](https://huggingface.co/datasets/AdaptLLM/ConvFinQA) | Multi-turn financial QA / 多轮金融问答 | EN | 12,594 | AdaptLLM |
| [TAT-QA](https://huggingface.co/datasets/next-tat/TAT-QA) | Table + text QA / 表格文本混合问答 | EN | 16,558 | TAT-QA Team |
| [FinanceIQ](https://huggingface.co/datasets/Duxiaoman-DI/FinanceIQ) | Financial exam MCQ / 金融考试选择题 | ZH | 7,173 | Duxiaoman-DI |
| [FinCUGE-Instruction](https://huggingface.co/datasets/Maciel/FinCUGE-Instruction) | Chinese financial NLP tasks / 中文金融 NLP 任务 | ZH | 138,304 | FinCUGE |
| [finance-alpaca](https://huggingface.co/datasets/gbharti/finance-alpaca) | Financial instruction data / 金融指令数据 | EN | 68,912 | gbharti |
| [financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank) | Sentiment classification / 情感分类 | EN | 14,780 | Takala |

> **Note:** FinCorpus was excluded due to its large volume (~200GB). / FinCorpus 因体积过大（约 200GB）未纳入。

---

## Schema Normalization / Schema 统一

All datasets are converted to a unified JSONL schema:

所有数据集统一为以下 JSONL schema：

```json
{
  "id": "unique_id",
  "source": "dataset_name",
  "split": "train|val|test",
  "task_type": "reasoning|mcq|instruction|sentiment",
  "task_domain": "financial_reasoning|financial_knowledge|...",
  "language": "en|zh",
  "question": "...",
  "context": "table or passage (flattened for tables)",
  "options": ["A. ...", "B. ..."],
  "reference_answer": "ground truth",
  "model_reasoning": "(filled by distillation)",
  "model_answer": "(filled by distillation)",
  "metadata": { "raw_sample": {...} }
}
```

**Split strategy / 划分策略:**
- Datasets with existing train/val/test splits: preserved as-is (FinQA, TAT-QA)
- Datasets with only train+eval or train-only: merged and re-split at **8:1:1** ratio with seed=42
  - 仅有 train+eval 或仅 train 的数据集：合并后按 8:1:1 比例重新划分

**Special handling / 特殊处理:**
- **ConvFinQA**: Multi-turn conversations flattened with dialogue history preserved in metadata / 多轮对话展平，对话历史保留在 metadata 中
- **TAT-QA**: Tables converted to markdown-style text with scale and unit metadata / 表格转为 markdown 格式文本，保留 scale 和 unit 元数据
- **FinCUGE-Instruction**: Multiple sub-tasks (sentiment, NER, relation extraction, event extraction, etc.) handled with task-type routing / 多种子任务通过 task_type 路由处理

---

## Distillation & Prompt Iteration / 蒸馏与 Prompt 迭代

### Methodology / 方法论

The core methodology is **per-dataset iterative prompt optimization**:

核心方法是**逐数据集迭代 Prompt 优化**：

1. **Pilot run**: Sample 500 records per dataset, distill reasoning chains via GPT-5.4
2. **Evaluation**: Compare distilled answers against ground-truth, compute strict accuracy
3. **Badcase analysis**: Categorize all errors (rounding, sign flip, abstention, evidence misread, etc.)
4. **Prompt optimization**: Design targeted prompt improvements based on error categories
5. **Re-iterate**: Run another 500-sample pilot, compare accuracy, repeat until convergence

### Accuracy Progression / 正确率迭代进展

| Dataset | Metric | Pilot v1 | Pilot v2 | Pilot v3 | Pilot v4 | Best | Notes / 备注 |
|---------|--------|---------|---------|---------|---------|------|------|
| **FinQA** | strict accuracy | 45.6% (baseline) | 44.4% (specialized) | — | — | v2 | Strict accuracy slightly dropped but substantive errors reduced 112→71 / 严格正确率略降但实质性错误从 112 降至 71 |
| **ConvFinQA** | strict accuracy | 42.2% | — | — | — | v1 | 216/500 were rounding/format only; repair pipeline added / 216 条仅为格式问题，增加了修复流水线 |
| **TAT-QA** | strict accuracy | 52.0% | 64.8% | — | — | v2 | +12.8% from prompt optimization, full run reached 63.8% / Prompt 优化提升 12.8%，全量运行达 63.8% |
| **FinanceIQ** | strict accuracy | 86.8% | 86.0% | **87.4%** | 86.0% | **v3** | v4 overengineered, v3 was simpler and better / v4 过度优化，简洁的 v3 效果最好 |
| **FinCUGE-Instruction** | answer keep rate | 64.0% | 66.2% | 62.4% (router) | — | **v2** | Sub-task FinRE: 26%→47% by simplifying prompt / 子任务 FinRE：简化 Prompt 后从 26% 升至 47% |
| **financial_phrasebank** | strict accuracy | **86.4%** | 85.6% | — | — | **v1** | Simpler v1 outperformed v2 / 更简洁的 v1 优于 v2 |
| **finance-alpaca** | answer keep rate | multi-version iteration | | | | v2 | Multiple parser and prompt iterations / 多版本迭代 |

### Key Insight / 关键发现

> **Simpler prompts sometimes outperform complex ones.** The FinCUGE-Instruction FinRE sub-task accuracy jumped from 26% to 47% simply by simplifying the prompt. Financial Phrasebank v1 (simpler) beat v2 (more detailed). FinanceIQ v3 (streamlined) outperformed v4 (over-specified).
>
> **简洁的 Prompt 有时优于复杂的 Prompt。** FinCUGE-Instruction 的 FinRE 子任务仅通过简化 Prompt 就从 26% 提升到 47%。Financial Phrasebank 的简洁版 v1 优于详细版 v2。FinanceIQ 的精简版 v3 优于过度指定的 v4。

### Badcase Categories / 错误类型分类

The project implements detailed error categorization per dataset type:

| Category | Description | Affected Datasets |
|----------|-------------|-------------------|
| `rounding_or_format` | Numeric format mismatch (e.g., "11.6" vs "$11.6 million") | FinQA, ConvFinQA, TAT-QA |
| `sign_error` | Correct magnitude, wrong sign | FinQA, ConvFinQA, TAT-QA |
| `calculation_error` | Wrong numeric result | FinQA, TAT-QA |
| `evidence_misread` | Read wrong value from table/passage | TAT-QA |
| `percentage_or_unit_error` | % vs decimal, million vs thousand | TAT-QA |
| `reasoning_answer_inconsistent` | Reasoning conclusion differs from final answer | TAT-QA, FinanceIQ |
| `abstain_error` | Model refuses to answer | All |
| `wrong_option_selection` | Selects wrong MCQ option | FinanceIQ |
| `entity/relation/event_error` | NLP extraction errors | FinCUGE-Instruction |
| `neutral_polarity_confusion` | Sentiment boundary errors | financial_phrasebank |

---

## Quality Filtering / 质量过滤

A **two-stage quality filter** is applied after full distillation:

蒸馏完成后应用**两阶段质量过滤**：

### Stage 1: Answer Correctness / 答案正确性
- **Rule-based scoring** for MCQ and judgment tasks (exact option match)
- **LLM judge fallback** (GPT-5.4) for open-ended answers where rules cannot resolve
- Binary output: `answer_score ∈ {0, 1}`

### Stage 2: Reasoning Quality / 推理质量
- Only applied to samples that pass Stage 1
- LLM judge evaluates: internal consistency, terminology, step count, logical flow, domain relevance, instruction alignment
- Binary output: `reasoning_score ∈ {0, 1}`

Final training data = samples passing **both** stages.

---

## Data Mixing / 数据混合

Final dataset composition follows domain-weighted sampling (from `configs/sampling_config.yaml`):

最终数据集按领域加权采样：

| Domain / 领域 | Weight | Datasets | Train Samples |
|------|--------|----------|---------------|
| Financial Reasoning / 金融推理 | 40% | FinQA (30%), ConvFinQA (30%), TAT-QA (40%) | 24,139 |
| Financial Instruction / 金融指令 | 30% | FinCUGE-Instruction (55%), finance-alpaca (45%) | 11,000 |
| Financial Knowledge / 金融知识 | 25% | FinanceIQ (100%) | 4,973 |
| Market Sentiment / 市场情绪 | 5% | financial_phrasebank (100%) | 2,500 |
| **Total** | **100%** | | **42,612 train / 5,380 val** |

---

## Final Output / 最终产出

Two formats are provided under `train_data/`:

`train_data/` 下提供两种格式：

- **`final_mixture_exact/`** — Original format with all fields (question, context, options, reference_answer, model_reasoning, model_answer, metadata)
- **`swift_final_mixture_exact/`** — [ms-swift](https://github.com/modelscope/ms-swift) SFT messages format (`system` / `user` / `assistant` roles with `<reasoning>` / `<final_answer>` tags)

---

## Project Structure / 项目结构

```
Fin-R1-data/
├── configs/                          # Configuration files / 配置文件
│   ├── distill.yaml                  # Distillation API & generation config
│   ├── distill_finance_alpaca_v1.yaml
│   ├── distill_finance_alpaca_v2.yaml
│   └── sampling_config.yaml          # Domain weights & mixing ratios
│
├── prompts/                          # All prompt templates / 所有 Prompt 模板
│   ├── finqa_reasoning_prompt.txt    # FinQA-specific prompt
│   ├── convfinqa_reasoning_prompt.txt
│   ├── tatqa_reasoning_prompt.txt
│   ├── financeiq_mcq_prompt_v1~v4_*.txt  # FinanceIQ v1-v4 iterations
│   ├── fincuge_instruction_prompt_*.txt  # FinCUGE sub-task prompts
│   ├── financial_phrasebank_sentiment_prompt_v1~v2.txt
│   ├── finance_alpaca_instruction_prompt_v1~v2.txt
│   ├── *_answer_judge_prompt.txt     # Answer scoring judge prompts
│   ├── reasoning_quality_*_prompt.txt # Reasoning quality judge prompts
│   └── ...
│
├── scripts/                          # Pipeline scripts / 流水线脚本
│   ├── download_open_datasets.py     # Download all 7 datasets
│   ├── verify_downloads.py           # Verify download integrity
│   ├── run_distill.py                # Run GPT-5.4 distillation
│   ├── run_convfinqa_repair.py       # Repair ConvFinQA format issues
│   ├── export_*_badcases.py          # Export badcases by category
│   ├── analyze_*_pilot.py            # Analyze pilot run results
│   ├── run_*_answer_score.py         # Per-dataset answer scoring
│   ├── run_two_stage_quality_filter.py  # Two-stage filter
│   ├── run_reasoning_quality_score.py   # Reasoning quality scoring
│   ├── sample_jsonl_records.py       # Deterministic sampling
│   ├── merge_jsonl_shards.py         # Merge sharded JSONL files
│   ├── publish_*_train_data.py       # Publish filtered data
│   ├── prepare_swift_sft_dataset.py  # Convert to ms-swift SFT format
│   └── run_finqa_convfinqa_benchmark.py  # Benchmark evaluation
│
├── src/                              # Core library / 核心库
│   ├── distill/                      # Distillation engine
│   │   ├── pipeline.py               # Main distillation pipeline
│   │   ├── provider.py               # OpenAI-compatible API provider
│   │   ├── prompt_builder.py         # Prompt rendering
│   │   ├── parser.py                 # Response parsing
│   │   └── cache.py                  # Request caching
│   ├── evaluation/                   # Evaluation utilities
│   │   ├── execution_eval.py         # Execution-based evaluation
│   │   └── normalizers.py            # Answer normalization
│   ├── eval/                         # Benchmark evaluation
│   │   ├── convfinqa_matcher.py      # ConvFinQA answer matching
│   │   └── llm_benchmark.py          # LLM benchmark runner
│   ├── postprocess/                  # Post-processing
│   │   └── convfinqa_postprocess.py  # ConvFinQA answer normalization
│   └── utils/
│       ├── io.py                     # JSONL I/O utilities
│       └── logging_utils.py          # Logging configuration
│
├── train_data/                       # Final training data / 最终训练数据
│   ├── final_mixture_exact/          # Original format (per-source + merged)
│   │   ├── train.jsonl               # 42,612 samples
│   │   ├── val.jsonl                 # 5,380 samples
│   │   └── manifest.json             # Mixture manifest
│   └── swift_final_mixture_exact/    # ms-swift SFT format
│       ├── train.jsonl
│       ├── val.jsonl
│       └── summary.json
│
├── requirements.txt
└── README.md
```

---

## Quick Start / 快速开始

### Use the pre-built training data / 直接使用预构建数据

The final training data is included in this repository:

```bash
# Original format
train_data/final_mixture_exact/train.jsonl   # 42,612 samples
train_data/final_mixture_exact/val.jsonl     # 5,380 samples

# ms-swift SFT format (ready for fine-tuning)
train_data/swift_final_mixture_exact/train.jsonl
train_data/swift_final_mixture_exact/val.jsonl
```

### Reproduce from scratch / 从零复现

```bash
# 1. Install dependencies / 安装依赖
pip install -r requirements.txt

# 2. Download raw datasets / 下载原始数据集
python scripts/download_open_datasets.py

# 3. Set API credentials for distillation / 设置蒸馏 API 凭证
export API_BASE="https://your-api-endpoint/v1"
export API_KEY="your-api-key"
export MODEL_NAME="gpt-5.4"

# 4. Run distillation (example: FinQA) / 运行蒸馏
python scripts/run_distill.py --sources finqa --config configs/distill.yaml

# 5. Evaluate and filter / 评估与过滤
python scripts/run_two_stage_quality_filter.py --input data/distilled/finqa/train.jsonl

# 6. Convert to SFT format / 转换为 SFT 格式
python scripts/prepare_swift_sft_dataset.py
```

---

## Acknowledgments / 致谢

- [Fin-R1](https://arxiv.org/abs/2503.07832) for the methodology inspiration / 方法论灵感
- All open-source dataset authors listed above / 上述所有开源数据集作者
- [ms-swift](https://github.com/modelscope/ms-swift) for the SFT training framework / SFT 训练框架

---

## License

This repository contains pipeline code and derived training data. The raw source datasets retain their original licenses. Please refer to each dataset's license before use.

本仓库包含流水线代码和衍生训练数据。原始数据集保留其原有许可证，使用前请参阅各数据集的许可证。
