#!/bin/bash
#SBATCH --job-name=v5_kaoyan_pass8_32k_corrected
#SBATCH --account=tml_2026
#SBATCH --partition=a100_normal_q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=v5_kaoyan_pass8_32k_corrected_%j.out
#SBATCH --error=v5_kaoyan_pass8_32k_corrected_%j.err

set -euo pipefail
export PATH="/home/juhis/.conda/envs/reasoning-eval/bin:$PATH"
module load CUDA/12.8.0
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export PATH="$CUDA_HOME/bin:$PATH"
echo "CUDA_HOME=$CUDA_HOME"
nvcc --version

cd ~/Project-Reasoning-SFT-LLM/eval

export CUDA_VISIBLE_DEVICES=0

/home/juhis/.conda/envs/reasoning-eval/bin/python eval.py \
  --model_name_or_path /home/juhis/Project-Reasoning-SFT-LLM/LLaMA-Factory/saves/qwen2.5-3b/advanced15k_v5_merged \
  --data_name kaoyan \
  --prompt_type qwen-instruct \
  --temperature 0.6 \
  --top_p 0.95 \
  --start_idx 0 \
  --n_sampling 8 \
  --k 8 \
  --split test \
  --max_tokens 32768 \
  --seed 0 \
  --surround_with_messages \
  --output_dir ./outputs/advanced15k_v5_kaoyan_pass8_32k_corrected \
  --completions_save_dir ./outputs/advanced15k_v5_kaoyan_pass8_32k_corrected/completions \
  --gpu_memory_utilization 0.85
