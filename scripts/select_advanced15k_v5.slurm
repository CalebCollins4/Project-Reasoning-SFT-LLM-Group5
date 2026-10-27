#!/bin/bash
#SBATCH --job-name=select15k_v5
#SBATCH --account=tml_2026
#SBATCH --partition=normal_q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=audit/select_v5_%j.out
cd ~/Project-Reasoning-SFT-LLM
/home/juhis/.conda/envs/reasoning-sft/bin/python scripts/select_advanced15k_v5.py 6044
