#!/bin/bash
#SBATCH --job-name=adv15k_v5
#SBATCH --account=tml_2026
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=/home/juhis/Project-Reasoning-SFT-LLM/LLaMA-Factory/adv15k_v5_%j.out
#SBATCH --error=/home/juhis/Project-Reasoning-SFT-LLM/LLaMA-Factory/adv15k_v5_%j.err

export PYTHONUNBUFFERED=1
module load CUDA/12.8.0 2>/dev/null || true
PY=/home/juhis/.conda/envs/reasoning-sft/bin/python
CLI=/home/juhis/.conda/envs/reasoning-sft/bin/llamafactory-cli
[ -x "$CLI" ] || CLI="$PY -m llamafactory.cli"
cd /home/juhis/Project-Reasoning-SFT-LLM/LLaMA-Factory
OUTLOG=/home/juhis/Project-Reasoning-SFT-LLM/LLaMA-Factory/adv15k_v5_${SLURM_JOB_ID}.out
OUTDIR=saves/qwen2.5-3b/advanced15k_v5
EPOCHS=${EPOCHS:-2}

echo "node: $(hostname)  start: $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
$PY -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; x=torch.randn(2048,2048,device='cuda'); print('CUDA OK', torch.cuda.get_device_name(0), float((x@x).sum()))" || { echo "GPU check failed"; exit 1; }

$CLI train examples/train_lora/advanced15k_v5.yaml num_train_epochs=$EPOCHS &
TRAIN_PID=$!
(
  while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 60
    if grep -q "NaN" $OUTDIR/trainer_log.jsonl 2>/dev/null || grep -qiE "'grad_norm': nan|'loss': nan" $OUTLOG 2>/dev/null; then
      echo "!!! NaN detected at $(date) - stopping training"; kill $TRAIN_PID; pkill -P $TRAIN_PID; break
    fi
  done
) &
wait $TRAIN_PID
STATUS=$?
echo "training exit status: $STATUS at $(date)"
[ $STATUS -eq 0 ] || { echo "training did not finish cleanly - skipping merge"; exit 1; }

$PY - <<'PYEOF2'
import sys, torch
from safetensors.torch import load_file
sd = load_file("saves/qwen2.5-3b/advanced15k_v5/adapter_model.safetensors")
bad = sum(1 for v in sd.values() if not torch.isfinite(v.float()).all())
print("adapter tensors:", len(sd), "non-finite:", bad)
sys.exit(1 if bad else 0)
PYEOF2
[ $? -eq 0 ] || { echo "adapter has NaN/Inf - not merging"; exit 1; }

$CLI export examples/train_lora/advanced15k_v5_merge.yaml
echo "done: $(date)"
ls -lh saves/qwen2.5-3b/advanced15k_v5_merged | head
