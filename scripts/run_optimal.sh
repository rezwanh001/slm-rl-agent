#!/bin/bash
# =============================================================================
# run_optimal.sh — Full RLHF Pipeline with Optimized Hyperparameters
# =============================================================================
# Hardware: 2× NVIDIA RTX A6000 (48 GB each)
# GPU 0: pythia-70m, pythia-410m
# GPU 1: pythia-160m, smollm2-135m, smollm2-360m
#
# Usage:
#   # Activate environment first:
#   conda activate slm-rl
#
#   # Run both GPUs in parallel:
#   bash scripts/run_optimal.sh 2>&1 | tee logs/run_optimal.log
#
#   # Or run a single model manually (see functions at bottom):
#   run_model pythia-70m tinystories 0
# =============================================================================

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="$REPO_ROOT/scripts"
DATA="$REPO_ROOT/data"
OUT="$REPO_ROOT/outputs"
LOGS="$REPO_ROOT/logs"
mkdir -p "$LOGS"

PYTHON=$(which python3)

# ── Model registry ─────────────────────────────────────────────────────────────
declare -A HF_MODEL=(
  [pythia-70m]="EleutherAI/pythia-70m-deduped"
  [pythia-160m]="EleutherAI/pythia-160m-deduped"
  [pythia-410m]="EleutherAI/pythia-410m-deduped"
  [smollm2-135m]="HuggingFaceTB/SmolLM2-135M"
  [smollm2-360m]="HuggingFaceTB/SmolLM2-360M"
)

# LoRA rank: higher rank = more capacity for larger models
declare -A LORA_R=(
  [pythia-70m]=8
  [pythia-160m]=16
  [pythia-410m]=32
  [smollm2-135m]=16
  [smollm2-360m]=32
)

DATASETS=(tinystories cnn_dailymail wikitext)

# =============================================================================
# stage_sft  <model_key> <dataset> <gpu_id>
# Optimal settings:
#   - 5 epochs: small models need more passes to converge on domain text
#   - batch 8 + grad_accum 4 = effective batch 32
#   - seq_len 512: captures full context without padding waste
#   - warmup 0.06: standard 6% warmup
#   - NEFTune noise 5: regularization for better generalization
# =============================================================================
stage_sft() {
  local MODEL=$1 DATASET=$2 GPU=$3
  local SFT_DIR="$OUT/$MODEL/$DATASET/sft"
  local LORA=${LORA_R[$MODEL]}

  [[ -d "$SFT_DIR/final" ]] && { echo "  [SKIP SFT] $MODEL/$DATASET already done"; return 0; }

  echo "  [SFT] $MODEL x $DATASET (GPU $GPU)"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$SCRIPTS/train_sft.py" \
    --model_name "${HF_MODEL[$MODEL]}" \
    --dataset_path "$DATA/$DATASET/sft_train.json" \
    --eval_dataset_path "$DATA/$DATASET/sft_eval.json" \
    --output_dir "$SFT_DIR" \
    --num_epochs 5 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.06 \
    --max_seq_length 512 \
    --lora_r "$LORA" \
    --lora_alpha $((LORA * 2)) \
    --neftune_noise_alpha 5.0 \
    --save_steps 500 \
    --logging_steps 50 \
    --seed 42 \
    2>&1 | tee "$LOGS/${MODEL}_${DATASET}_sft.log"
  echo "  [SFT DONE] $MODEL x $DATASET"
}

# =============================================================================
# stage_reward  <model_key> <dataset> <gpu_id>
# Optimal settings:
#   - 2 epochs: 1 epoch often underfits; 2 gives better calibration
#   - batch 8 + grad_accum 2 = effective batch 16
#   - seq_len 512: matches SFT
#   - lr 1e-5: conservative to avoid catastrophic forgetting of SFT init
# =============================================================================
stage_reward() {
  local MODEL=$1 DATASET=$2 GPU=$3
  local RM_DIR="$OUT/$MODEL/$DATASET/reward_model"
  local LORA=${LORA_R[$MODEL]}

  [[ -d "$RM_DIR/final" ]] && { echo "  [SKIP RM] $MODEL/$DATASET already done"; return 0; }

  echo "  [REWARD] $MODEL x $DATASET (GPU $GPU)"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$SCRIPTS/train_reward.py" \
    --base_model "$OUT/$MODEL/$DATASET/sft/final" \
    --dataset_path "$DATA/$DATASET/preference_train.json" \
    --eval_dataset_path "$DATA/$DATASET/preference_eval.json" \
    --output_dir "$RM_DIR" \
    --num_epochs 2 \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --max_seq_length 512 \
    --lora_r "$LORA" \
    --lora_alpha $((LORA * 2)) \
    --save_steps 200 \
    --logging_steps 20 \
    --seed 42 \
    2>&1 | tee "$LOGS/${MODEL}_${DATASET}_reward.log"
  echo "  [REWARD DONE] $MODEL x $DATASET"
}

# =============================================================================
# stage_ppo  <model_key> <dataset> <gpu_id>
# PPO settings — tuned after observing collapse on aggressive hyperparameters:
#   - 500 steps: sufficient with reward whitening + adaptive KL
#   - batch 32, mini_batch 4: small mini-batch = more updates per rollout
#   - num_ppo_epochs 2: 2 gradient passes → stable + efficient
#   - kl_penalty 0.2: strong KL anchoring prevents reward hacking
#   - target_kl 6.0: adaptive KL target (TRL default) — don't over-constrain
#   - lr 5e-6: half the standard LR → safer updates for small models
#   - max_new_tokens 96: shorter → cleaner reward signal, less drift
#   - temperature 0.9, top_p 0.95: standard exploration
# =============================================================================
stage_ppo() {
  local MODEL=$1 DATASET=$2 GPU=$3
  local PPO_DIR="$OUT/$MODEL/$DATASET/ppo"
  local LORA=${LORA_R[$MODEL]}

  [[ -d "$PPO_DIR/final" ]] && { echo "  [SKIP PPO] $MODEL/$DATASET already done"; return 0; }

  echo "  [PPO] $MODEL x $DATASET (GPU $GPU)"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$SCRIPTS/train_ppo.py" \
    --policy_model "$OUT/$MODEL/$DATASET/sft/final" \
    --reward_model "$OUT/$MODEL/$DATASET/reward_model/final" \
    --dataset_path "$DATA/$DATASET/sft_train.json" \
    --output_dir "$PPO_DIR" \
    --num_steps 500 \
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
    --save_steps 100 \
    --logging_steps 10 \
    --seed 42 \
    2>&1 | tee "$LOGS/${MODEL}_${DATASET}_ppo.log"
  echo "  [PPO DONE] $MODEL x $DATASET"
}

# =============================================================================
# stage_eval  <model_key> <dataset> <stage> <gpu_id>
# =============================================================================
stage_eval() {
  local MODEL=$1 DATASET=$2 STAGE=$3 GPU=$4
  local EVAL_DIR="$OUT/$MODEL/$DATASET/eval_$STAGE"

  [[ -f "$EVAL_DIR/evaluation_results.json" ]] && { echo "  [SKIP EVAL] $MODEL/$DATASET/$STAGE"; return 0; }

  echo "  [EVAL $STAGE] $MODEL x $DATASET (GPU $GPU)"
  CUDA_VISIBLE_DEVICES=$GPU $PYTHON "$SCRIPTS/evaluate.py" \
    --model_path "$OUT/$MODEL/$DATASET/$STAGE/final" \
    --eval_dataset "$DATA/$DATASET/sft_eval.json" \
    --reward_model_path "$OUT/$MODEL/$DATASET/reward_model/final" \
    --output_dir "$EVAL_DIR" \
    --max_samples 200 \
    --max_new_tokens 128 \
    --temperature 0.8 \
    --batch_size 16 \
    2>&1 | tee "$LOGS/${MODEL}_${DATASET}_eval_${STAGE}.log"
  echo "  [EVAL $STAGE DONE] $MODEL x $DATASET"
}

# =============================================================================
# run_model  <model_key> <dataset> <gpu_id>
# Full pipeline for one model × dataset pair
# =============================================================================
run_model() {
  local MODEL=$1 DATASET=$2 GPU=$3
  echo "============================================================"
  echo "  START: $MODEL x $DATASET → GPU $GPU"
  echo "============================================================"
  stage_sft    "$MODEL" "$DATASET" "$GPU"
  stage_reward "$MODEL" "$DATASET" "$GPU"
  stage_ppo    "$MODEL" "$DATASET" "$GPU"
  stage_eval   "$MODEL" "$DATASET" sft "$GPU"
  stage_eval   "$MODEL" "$DATASET" ppo "$GPU"
  echo "  ALL DONE: $MODEL x $DATASET"
}

# =============================================================================
# Main: run GPU 0 and GPU 1 in parallel
# GPU 0: pythia-70m (all datasets) then pythia-410m (all datasets)
# GPU 1: pythia-160m then smollm2-135m then smollm2-360m
# =============================================================================
echo "Starting full pipeline — $(date)"
echo "Logs → $LOGS/"

(
  echo "[GPU 0] pythia-70m + pythia-410m"
  for DS in "${DATASETS[@]}"; do run_model pythia-70m  "$DS" 0; done
  for DS in "${DATASETS[@]}"; do run_model pythia-410m "$DS" 0; done
  echo "[GPU 0] DONE"
) &
GPU0_PID=$!

(
  echo "[GPU 1] pythia-160m + smollm2-135m + smollm2-360m"
  for DS in "${DATASETS[@]}"; do run_model pythia-160m   "$DS" 1; done
  for DS in "${DATASETS[@]}"; do run_model smollm2-135m  "$DS" 1; done
  for DS in "${DATASETS[@]}"; do run_model smollm2-360m  "$DS" 1; done
  echo "[GPU 1] DONE"
) &
GPU1_PID=$!

echo "GPU 0 PID=$GPU0_PID | GPU 1 PID=$GPU1_PID"
echo "Monitor GPU usage:   watch -n5 nvidia-smi"
echo "Monitor logs:        tail -f logs/<model>_<dataset>_<stage>.log"

wait "$GPU0_PID"
echo "GPU 0 finished — $(date)"
wait "$GPU1_PID"
echo "GPU 1 finished — $(date)"

echo ""
echo "============================================================"
echo "  ALL EXPERIMENTS COMPLETE — $(date)"
echo "============================================================"

# Upload newly trained models to HuggingFace
echo "Uploading new checkpoints to HuggingFace..."
python3 /tmp/upload_single_repo.py --only-models --skip-delete 2>&1 | tee "$LOGS/hf_upload.log"
echo "Upload complete. Repo: https://huggingface.co/datasets/mr3haque/SLM-RL-Agent"
