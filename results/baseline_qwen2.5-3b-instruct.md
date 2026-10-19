# Baseline: Qwen2.5-3B-Instruct (before fine-tuning)

All 8 benchmarks in `eval/data`, evaluated with the repo's own `eval/eval_single.sh` (not modified) on 2026-10-18 at repo commit `638989b`.

## Scores

`Acc` is pass@8 (a problem counts if any of the 8 samples is correct), the metric the README leaderboard uses. `Pass@1` is the mean accuracy over the 8 samples; `eval/README.md` asks for this one too.

| Benchmark | Problems | Acc (pass@8) | Pass@1 | README leaderboard baseline row (Fall 2025, pass@8) |
|---|---:|---:|---:|---:|
| aime (AIME 2024) | 30 | 0.1667 | 0.0833 | 0.200 |
| math (MATH-500) | 500 | 0.8640 | 0.6570 | 0.844 |
| cn_math_2024 | 30 | 0.3667 | 0.1167 | 0.233 |
| kaoyan | 199 | 0.5377 | 0.2180 | 0.513 |
| amc | 40 | 0.7000 | 0.4188 | 0.700 |
| minerva | 272 | 0.4743 | 0.3015 | 0.338 |
| olympiadbench | 675 | 0.4993 | 0.2811 | 0.495 |
| gpqa | 198 | 0.7323 | 0.3037 | 0.742 |
| **mean of 8** | | **0.5426** | **0.2975** | 0.508 |

The leaderboard AVG also includes AIME25, which is not in `eval/data`, so the means here cover these 8 benchmarks only. `math` has 500 problems in `eval/data/math/test.jsonl` (the README table says 499).

## Settings

Defaults of `eval_single.sh` / `sbatch eval.sh`: `--temperature 0.6 --top_p 0.95 --n_sampling 8 --k 1 --max_tokens 32768 --seed 0 --prompt_type qwen-instruct --surround_with_messages --split test`, system prompt `Please reason step by step, and put your final answer within \boxed{}.`

## Environment of this run

This run was done locally, not on ARC, because `eval/requirements.txt` pins `vllm<=0.6.1`, which does not support an RTX 5070 Ti.

- 1x NVIDIA RTX 5070 Ti (16 GB), Docker Desktop (WSL2), image `vllm/vllm-openai:v0.32.0`
- vllm 0.32.0, torch 2.13.0+cu130, transformers 5.19.0, sympy 1.14.0, antlr4-python3-runtime 4.9.3, latex2sympy2 1.9.1
- `gpu_memory_utilization` 0.85 (ARC default is 0.96); total wall time about 4.2 h

vLLM sampling is not bit-identical across versions, so a rerun on ARC will not reproduce these numbers exactly.

## Notes

- **Small benchmarks are noisy.** One problem is worth 0.033 on `aime` / `cn_math_2024` and 0.025 on `amc`. `cn_math_2024` is 4 problems above the Fall-2025 row, which is within the noise of a 30-problem set.
- **Grader tolerance.** `numeric_equal` in `eval/utils/grader.py` uses `isclose(..., abs_tol=1e-4)` (the relative-tolerance line above it is commented out), and `symbolic_equal` falls back to that check, so any two numbers within 1e-4 of each other compare equal, which makes all tiny values interchangeable. For example, gold `5.3e-9` accepts `3.4 \times 10^{-9}`. `math_equal` also accepts reference/100 and reference*100. Re-grading the saved answers with `rel_tol=1e-4` and without the /100 and *100 rule, using the repo's own `check_is_correct`, changes only two benchmarks by more than 0.001: minerva goes from 0.4743 to 0.4265 Acc (Pass@1 0.3015 to 0.2505) and olympiadbench from 0.4993 to 0.4948. Minerva stays about 0.09 above the Fall-2025 row even then, and that remaining gap is unexplained. Use the same grader for the baseline and the fine-tuned models so the comparison stays consistent.
- **`eval/requirements.txt` does not install as written.** `pip install -r eval/requirements.txt` fails with `ResolutionImpossible` because `antlr4-python3-runtime==4.11.1` conflicts with `latex2sympy2==1.9.1` (reproduced in a clean venv with just those pins and `sympy==1.12`), and `latex2sympy2` cannot load with antlr 4.11.1. This run used `antlr4-python3-runtime==4.9.3` plus `pip install --no-deps latex2sympy2==1.9.1`. With antlr 4.9.3, sympy's `parse_latex` is unavailable (it needs 4.11), so the grader silently falls back to `parse_expr` and `latex2sympy2`. The logs are full of `ANTLR runtime and generated code versions disagree: 4.9.3!=4.7.2` warnings, which are harmless.
