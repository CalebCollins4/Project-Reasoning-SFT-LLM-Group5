# Reasoning SFT for Qwen2.5-3B-Instruct

**Virginia Tech · ECE 6514 Trustworthy Machine Learning · Fall 2026**  
**Team:** Caleb Collins, Jidapa Chalermkit, Juhi Vikas Sawant

## Project overview

We study how selecting 15,000 supervised fine-tuning examples from the AceReason-1.1-SFT dataset affects the mathematical and general reasoning performance of **Qwen2.5-3B-Instruct**. We compare a random 15K subset with more targeted data-selection strategies, using the original model as a baseline.

The experiments use **LLaMA-Factory** for supervised fine-tuning and evaluate reasoning on eight benchmarks.

## Data-selection strategies

| Strategy | Description |
|---|---|
| Baseline | Original Qwen2.5-3B-Instruct, without project fine-tuning |
| Random15K | 15,000 randomly selected training examples |
| Advanced15K V5 | Difficulty-informed selection of 15,000 examples |
| Advanced15K (diversity-based) | Selection designed to increase coverage and diversity of reasoning examples |

## Evaluation results

**Pass@8 accuracy (%):** a problem counts as solved if at least one of eight generated answers is correct. The table combines results in the midterm report with subsequently verified V5 results, including the completed MATH-500 run (422/500).

| Benchmark | Baseline | Random15K | Advanced15K V5 | Advanced15K (diversity) |
|---|---:|---:|---:|---:|
| AIME 2024 | 16.67 | 10.00 | **23.33** | 16.67 |
| MATH-500 | 86.40 | 83.20 | **84.40** | 88.20 |
| CN Math 2024 | 36.67 | 13.33 | **33.33** | 36.67 |
| Kaoyan | 53.77 | 58.29 | **61.31** | 63.82 |
| AMC 2023 | 70.00 | 65.00 | **62.50** | 80.00 |
| Minerva Math | 47.43 | 42.65 | **46.69** | 45.32 |
| OlympiadBench | 49.93 | — | Pending | 55.41 |
| GPQA-Diamond | 73.23 | 40.91 | **73.23** | 69.19 |

**Note:** V5 values are from the latest verified 32,768-token evaluation runs. The other columns reflect the midterm report; the reported configurations or evaluation status may differ. A dash indicates an unavailable result, not a zero score.

## Training and evaluation setup

| Setting | Advanced15K V5 |
|---|---|
| Base model | Qwen2.5-3B-Instruct |
| Training dataset | 15,000 selected examples (`advanced15k_v5.json`) |
| Fine-tuning | LoRA via LLaMA-Factory |
| LoRA rank / alpha / dropout | 64 / 128 / 0.05 |
| Training epochs | 2 |
| Training precision | BF16 |
| Evaluation temperature / top-p | 0.6 / 0.95 |
| Samples per question | 8 |
| Maximum generated tokens | 32,768 |
| Compute | Virginia Tech ARC Falcon |

## Repository and artifacts

- **Original project documentation:** See the repository's root `README.md` for setup and upstream instructions.
- **Experiment scripts and configurations:** See the `scripts/`, `eval/`, and `LLaMA-Factory/` directories, where available in this fork.
- **Team report:** *TML Midterm Report* (Fall 2026).
- **Hugging Face organization:** [TML-Project1](https://huggingface.co/TML-Project1) — hosts the team’s datasets, fine-tuned models, and other published artifacts. Large model weights and training datasets are not included in this Git repository.

## Key takeaway

Data selection meaningfully changes performance across benchmarks. **Advanced15K V5** improves over the random subset on several tasks, including AIME, CN Math, Kaoyan, Minerva Math, and GPQA-Diamond, while the diversity-based strategy performs best on several others. No single selection strategy dominates every benchmark.
