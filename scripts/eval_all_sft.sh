#!/bin/bash
# Evaluate all SFT checkpoints in parallel across 2 GPUs
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/cpami-gpu-2xa6000/miniconda/envs/slm-rl/bin/python3
OUT="$REPO/outputs"
DATA="$REPO/data"
LOGS="$REPO/logs"
mkdir -p "$LOGS"

DATASETS=(tinystories cnn_dailymail wikitext)

eval_one() {
  local MODEL=$1 DS=$2 GPU=$3
  local OUTDIR="$OUT/$MODEL/$DS/eval_sft"
  [[ -f "$OUTDIR/evaluation_results.json" ]] && { echo "[SKIP] $MODEL/$DS"; return 0; }
  echo "[EVAL SFT] $MODEL/$DS on GPU $GPU"
  CUDA_VISIBLE_DEVICES=$GPU $PY "$REPO/scripts/evaluate.py" \
    --model_path "$OUT/$MODEL/$DS/sft/final" \
    --eval_dataset "$DATA/$DS/sft_eval.json" \
    --reward_model_path "$OUT/$MODEL/$DS/reward_model/final" \
    --output_dir "$OUTDIR" \
    --max_samples 200 \
    --max_new_tokens 128 \
    --temperature 0.8 \
    --batch_size 16 \
    > "$LOGS/${MODEL}_${DS}_eval_sft.log" 2>&1
  echo "[DONE] $MODEL/$DS"
}

(
  for DS in "${DATASETS[@]}"; do eval_one pythia-70m   "$DS" 0; done
  for DS in "${DATASETS[@]}"; do eval_one pythia-410m  "$DS" 0; done
) &
PID0=$!

(
  for DS in "${DATASETS[@]}"; do eval_one pythia-160m  "$DS" 1; done
  for DS in "${DATASETS[@]}"; do eval_one smollm2-135m "$DS" 1; done
  for DS in "${DATASETS[@]}"; do eval_one smollm2-360m "$DS" 1; done
) &
PID1=$!

wait $PID0 $PID1
echo "All SFT evaluations complete."
