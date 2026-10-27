# Advanced15K V5 — Dataset Selection, Training, and Evaluation

## Overview

Advanced15K V5 is a quality-filtered, diversity-aware subset of 15,000 examples selected from the AceReason-1.1-SFT 100K dataset.

The selected dataset was used to fine-tune Qwen2.5-3B-Instruct using supervised fine-tuning (SFT) with LoRA.

## Dataset Selection

The V5 selection process uses the following stages:

1. Exclude code examples, examples without boxed answers, and examples exceeding the configured length or quality thresholds.
2. Remove duplicate instructions.
3. Filter examples with high densities of self-doubt markers such as "Wait", "Hmm", and "Alternatively".
4. Represent instructions using TF-IDF with up to 30,000 features.
5. Reduce dimensionality using Truncated SVD with 128 components.
6. Cluster examples using MiniBatch K-Means with 100 clusters.
7. Allocate selection quotas proportional to the square root of cluster sizes.
8. Apply length-aware sampling within each cluster.
9. Select and shuffle 15,000 examples using random seed 42.

### Selection Statistics

| Metric | Value |
|---|---:|
| Source dataset | AceReason-1.1-SFT 100K |
| After basic quality filters | 28,683 |
| After deduplication | 25,892 |
| After overthinking filter | 22,008 |
| Final selected examples | 15,000 |
| Maximum selection length | 6,044 tokens |
| Median selected length | 3,824 tokens |
| Mean selected length | 3,789 tokens |
| Total selected tokens | 56.8 million |
| Clusters represented | 100/100 |
| Random seed | 42 |

The full selection statistics are available in `audit/advanced15k_v5_report.json`.

## Training Configuration

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen2.5-3B-Instruct |
| Fine-tuning method | SFT with LoRA |
| LoRA rank | 64 |
| LoRA alpha | 128 |
| LoRA dropout | 0.05 |
| Learning rate | 1e-4 |
| Epochs | 2 |
| Per-device batch size | 1 |
| Gradient accumulation | 16 |
| Effective batch size | 16 (single GPU) |
| Training cutoff length | 6,144 tokens |
| Learning-rate scheduler | Cosine |
| Warmup ratio | 0.05 |
| Precision | BF16 |
| Maximum gradient norm | 1.0 |

Training configuration: `LLaMA-Factory/examples/train_lora/advanced15k_v5.yaml`

## Evaluation Configuration

The 32K Pass@8 evaluation uses:

- Temperature: 0.6
- Top-p: 0.95
- Samples per problem: 8
- Maximum generation tokens: 32,768
- System instruction: "Please reason step by step, and put your final answer within \boxed{}."

The 16K single-sample evaluation files use temperature 0.0 and one sample per problem. These are separate evaluations and should not be interpreted as Pass@8 results.

## Advanced15K V5 — 32K Pass@8 Results

| Benchmark | Correct / Total | Pass@8 Accuracy |
|---|---:|---:|
| AIME 2024 | 7/30 | 23.33% |
| AMC 2023 | 25/40 | 62.50% |
| CN Math 2024 | 10/30 | 33.33% |
| GPQA-Diamond | 145/198 | 73.23% |
| Kaoyan | 122/199 | 61.31% |
| Minerva Math | 127/272 | 46.69% |
| MATH-500 | Pending verification | — |
| OlympiadBench | Pending verification | — |

The reported accuracy is Pass@8: a problem is counted as solved when at least one of eight generated responses is correct.

These results were recorded from earlier evaluation runs. The raw evaluation outputs are retained on the Falcon cluster rather than committed to GitHub.

## Reproducibility

Dataset selection: `scripts/select_advanced15k_v5.py`

Dataset-selection SLURM script: `scripts/select_advanced15k_v5.slurm`

Dataset registration: `scripts/register_v5.py`

Training SLURM script: `scripts/train_advanced15k_v5.slurm`

Training configuration: `LLaMA-Factory/examples/train_lora/advanced15k_v5.yaml`

Model merging configuration: `LLaMA-Factory/examples/train_lora/advanced15k_v5_merge.yaml`

Benchmark evaluation scripts: `eval/eval_advanced15k_v5_*.sh`

The complete training dataset and model weights are hosted separately on Hugging Face and are not included in this Git repository.

## Notes

- Selection length (6,044 tokens) and training cutoff length (6,144 tokens) are separate settings.
- The 16K and 32K evaluations use different generation settings and should be reported separately.
- MATH-500 and OlympiadBench 32K results must be updated after their final evaluation status and scores are verified.
