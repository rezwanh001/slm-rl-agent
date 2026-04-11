#!/bin/bash
# Evaluate external SOTA instruct-tuned SLM baselines against our reward models.
# Uses the tinystories reward model per-param-class (most comparable domain).
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/cpami-gpu-2xa6000/miniconda/envs/slm-rl/bin/python3
OUT="$REPO/outputs"
DATA="$REPO/data"
LOGS="$REPO/logs"
mkdir -p "$LOGS"

# Each baseline is paired with the OUR-family model whose reward scale is
# closest in size so the cross-comparison is informative.
#   baseline_name | hf_id | our_counterpart
BASELINES=(
  "smollm2-135m-instruct|HuggingFaceTB/SmolLM2-135M-Instruct|smollm2-135m"
  "smollm2-360m-instruct|HuggingFaceTB/SmolLM2-360M-Instruct|smollm2-360m"
  "qwen25-05b-instruct|Qwen/Qwen2.5-0.5B-Instruct|smollm2-360m"
)

DATASETS=(tinystories cnn_dailymail wikitext)

eval_baseline() {
  local NAME=$1 HF=$2 OURS=$3 DS=$4 GPU=$5
  local OUTDIR="$OUT/baselines/$NAME/$DS"
  [[ -f "$OUTDIR/evaluation_results.json" ]] && { echo "[SKIP] $NAME/$DS"; return 0; }
  mkdir -p "$OUTDIR"
  echo "[BASELINE] $NAME on $DS (GPU $GPU, reward=$OURS)"
  CUDA_VISIBLE_DEVICES=$GPU $PY "$REPO/scripts/evaluate_baseline.py" \
    --model_name "$HF" \
    --eval_dataset "$DATA/$DS/sft_eval.json" \
    --reward_model_path "$OUT/$OURS/$DS/reward_model/final" \
    --our_tokenizer_path "$OUT/$OURS/$DS/sft/final" \
    --output_dir "$OUTDIR" \
    --max_samples 200 \
    --max_new_tokens 128 \
    --temperature 0.8 \
    --batch_size 8 \
    > "$LOGS/baseline_${NAME}_${DS}.log" 2>&1
  echo "[DONE] $NAME/$DS"
}

# Serialize to avoid GPU OOM when PPO also running
for entry in "${BASELINES[@]}"; do
  IFS='|' read -r NAME HF OURS <<< "$entry"
  for DS in "${DATASETS[@]}"; do
    eval_baseline "$NAME" "$HF" "$OURS" "$DS" "${1:-0}"
  done
done
echo "All baselines evaluated."
