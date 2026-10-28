#!/bin/bash
#SBATCH --job-name=sft_random15k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --partition=a30_normal_q
#SBATCH --qos=fal_a30_normal_short
#SBATCH --account=tml_2026
#SBATCH --gres=gpu:4

# Full fine-tune of Qwen2.5-3B-Instruct on Random15K (ARC Falcon, 4x A30, ZeRO-3).
# Usage, from the repository root on Falcon: sbatch scripts/train_random15k.slurm
# Extra arguments override the YAML, e.g. sbatch scripts/train_random15k.slurm max_steps=10

module load Miniconda3
module load CUDA/12.6.0
source activate trainenv

export TRITON_CACHE_DIR=/tmp/triton_$USER
export HF_HOME=$HOME/.cache/huggingface

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/LLaMA-Factory"
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/random15k.yaml "$@"
