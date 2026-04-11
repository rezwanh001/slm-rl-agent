#!/bin/bash
# Re-run PPO for all 15 configs with conservative hyperparameters + reward whitening.
# SFT and reward models are already trained — skip them.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/cpami-gpu-2xa6000/miniconda/envs/slm-rl/bin/python3
OUT="$REPO/outputs"
DATA="$REPO/data"
LOGS="$REPO/logs"
mkdir -p "$LOGS"

declare -A LORA_R=(
  [pythia-70m]=8
  [pythia-160m]=16
  [pythia-410m]=32
  [smollm2-135m]=16
  [smollm2-360m]=32
)

DATASETS=(tinystories cnn_dailymail wikitext)

run_ppo() {
  local MODEL=$1 DS=$2 GPU=$3
  local PPO_DIR="$OUT/$MODEL/$DS/ppo"
  local LORA=${LORA_R[$MODEL]}
  [[ -d "$PPO_DIR/final" ]] && { echo "[SKIP PPO] $MODEL/$DS"; return 0; }
  echo "[PPO] $MODEL/$DS (GPU $GPU)"
  CUDA_VISIBLE_DEVICES=$GPU $PY "$REPO/scripts/train_ppo.py" \
    --policy_model "$OUT/$MODEL/$DS/sft/final" \
    --reward_model "$OUT/$MODEL/$DS/reward_model/final" \
    --dataset_path "$DATA/$DS/sft_train.json" \
    --output_dir "$PPO_DIR" \
    --num_steps 250 \
    --batch_size 32 \
    --mini_batch_size 4 \
    --num_ppo_epochs 2 \
    --learning_rate 5e-6 \
    --kl_penalty 0.2 \
    --target_kl 6.0 \
    --clip_range 0.2 \
    --gamma 1.0 \
    --gae_lambda 0.95 \
    --max_new_tokens 96 \
    --temperature 0.9 \
    --top_p 0.95 \
    --lora_r "$LORA" \
    --save_steps 1000 \
    --logging_steps 25 \
    --seed 42 \
    > "$LOGS/${MODEL}_${DS}_ppo.log" 2>&1
  echo "[PPO DONE] $MODEL/$DS"
}

(
  for DS in "${DATASETS[@]}"; do run_ppo pythia-70m   "$DS" 0; done
  for DS in "${DATASETS[@]}"; do run_ppo pythia-410m  "$DS" 0; done
) &
PID0=$!

(
  for DS in "${DATASETS[@]}"; do run_ppo pythia-160m  "$DS" 1; done
  for DS in "${DATASETS[@]}"; do run_ppo smollm2-135m "$DS" 1; done
  for DS in "${DATASETS[@]}"; do run_ppo smollm2-360m "$DS" 1; done
) &
PID1=$!

wait $PID0 $PID1
echo "All PPO runs complete."
