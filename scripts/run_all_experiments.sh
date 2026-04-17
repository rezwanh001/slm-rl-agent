#!/bin/bash
# ==============================================================================
# Master Experiment Runner for SLM-RL-Agents
# Runs the full RLHF pipeline (SFT → Reward Model → PPO → Evaluation)
# for all model x dataset combinations.
#
# Usage:
#   bash scripts/run_all_experiments.sh [--models "pythia-70m pythia-160m"] [--datasets "tinystories cnn_dailymail wikitext"]
#
# Requirements: conda environment 'slm-rl' with all dependencies installed
# ==============================================================================

set -e

# Configuration
CONDA_ENV="slm-rl"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${PROJECT_DIR}/data"
OUTPUT_DIR="${PROJECT_DIR}/outputs"
RESULTS_DIR="${PROJECT_DIR}/results"
LOG_DIR="${PROJECT_DIR}/logs"

# Default models and datasets
MODELS="${MODELS:-pythia-70m pythia-160m pythia-410m smollm2-135m smollm2-360m}"
DATASETS="${DATASETS:-tinystories cnn_dailymail wikitext}"

# Training hyperparameters
SFT_EPOCHS=3
SFT_BATCH_SIZE=8
SFT_LR=2e-5
SFT_MAX_SEQ=512

RM_EPOCHS=1
RM_BATCH_SIZE=4
RM_LR=1e-5

PPO_STEPS=500
PPO_BATCH_SIZE=32
PPO_MINI_BATCH=8
PPO_LR=1e-5
PPO_KL=0.1

MAX_SAMPLES=10000

# Parse command line args
while [[ $# -gt 0 ]]; do
    case $1 in
        --models) MODELS="$2"; shift 2;;
        --datasets) DATASETS="$2"; shift 2;;
        --max-samples) MAX_SAMPLES="$2"; shift 2;;
        --sft-only) SFT_ONLY=1; shift;;
        --skip-sft) SKIP_SFT=1; shift;;
        --skip-reward) SKIP_REWARD=1; shift;;
        --skip-ppo) SKIP_PPO=1; shift;;
        --skip-eval) SKIP_EVAL=1; shift;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

# Map short names to HuggingFace model IDs
get_model_id() {
    case $1 in
        pythia-70m)    echo "EleutherAI/pythia-70m-deduped";;
        pythia-160m)   echo "EleutherAI/pythia-160m-deduped";;
        pythia-410m)   echo "EleutherAI/pythia-410m-deduped";;
        smollm2-135m)  echo "HuggingFaceTB/SmolLM2-135M";;
        smollm2-360m)  echo "HuggingFaceTB/SmolLM2-360M";;
        smollm2-1.7b)  echo "HuggingFaceTB/SmolLM2-1.7B";;
        distilgpt2)    echo "distilbert/distilgpt2";;
        *)             echo "$1";;
    esac
}

# Get LoRA rank based on model size
get_lora_r() {
    case $1 in
        pythia-70m|smollm2-135m|distilgpt2) echo 8;;
        pythia-160m|smollm2-360m) echo 16;;
        pythia-410m|smollm2-1.7b) echo 32;;
        *) echo 16;;
    esac
}

# Whether to use quantization
use_quant() {
    case $1 in
        pythia-70m|distilgpt2) echo "--no_quantization";;
        *) echo "";;
    esac
}

# Activate conda
eval "$(/home/cpami-gpu-2xa6000/miniconda/bin/conda shell.bash hook)"
conda activate ${CONDA_ENV}

# Create directories
mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

echo "============================================================"
echo "SLM-RL-Agents: Full Experiment Pipeline"
echo "============================================================"
echo "Models:   ${MODELS}"
echo "Datasets: ${DATASETS}"
echo "Output:   ${OUTPUT_DIR}"
echo "============================================================"

# Track timing
PIPELINE_START=$(date +%s)

# ==============================================================================
# Step 0: Prepare datasets (if not already done)
# ==============================================================================
if [ ! -d "${DATA_DIR}/tinystories" ] || [ ! -d "${DATA_DIR}/cnn_dailymail" ] || [ ! -d "${DATA_DIR}/wikitext" ]; then
    echo ""
    echo "[STEP 0] Preparing datasets..."
    python "${PROJECT_DIR}/scripts/prepare_all_datasets.py" \
        --output_dir "${DATA_DIR}" \
        --max_samples ${MAX_SAMPLES} \
        2>&1 | tee "${LOG_DIR}/data_preparation.log"
    echo "[STEP 0] Dataset preparation complete!"
fi

# ==============================================================================
# Run pipeline for each model x dataset combination
# ==============================================================================
EXPERIMENT_NUM=0
TOTAL_EXPERIMENTS=$(echo ${MODELS} | wc -w)
TOTAL_EXPERIMENTS=$((TOTAL_EXPERIMENTS * $(echo ${DATASETS} | wc -w)))

for MODEL_SHORT in ${MODELS}; do
    MODEL_ID=$(get_model_id ${MODEL_SHORT})
    LORA_R=$(get_lora_r ${MODEL_SHORT})
    LORA_ALPHA=$((LORA_R * 2))
    QUANT_FLAG=$(use_quant ${MODEL_SHORT})

    for DATASET in ${DATASETS}; do
        EXPERIMENT_NUM=$((EXPERIMENT_NUM + 1))
        EXP_START=$(date +%s)

        EXP_DIR="${OUTPUT_DIR}/${MODEL_SHORT}/${DATASET}"
        EXP_LOG="${LOG_DIR}/${MODEL_SHORT}_${DATASET}"
        mkdir -p "${EXP_DIR}" "${EXP_LOG}"

        echo ""
        echo "============================================================"
        echo "[${EXPERIMENT_NUM}/${TOTAL_EXPERIMENTS}] ${MODEL_SHORT} x ${DATASET}"
        echo "============================================================"

        # ------------------------------------------------------------------
        # Stage 1: SFT
        # ------------------------------------------------------------------
        if [ -z "${SKIP_SFT}" ] && [ ! -f "${EXP_DIR}/sft/final/config.json" ]; then
            echo "[Stage 1/4] SFT Training: ${MODEL_SHORT} on ${DATASET}..."
            python "${PROJECT_DIR}/scripts/train_sft.py" \
                --model_name "${MODEL_ID}" \
                --dataset_path "${DATA_DIR}/${DATASET}/sft_train.json" \
                --eval_dataset_path "${DATA_DIR}/${DATASET}/sft_eval.json" \
                --output_dir "${EXP_DIR}/sft" \
                --num_epochs ${SFT_EPOCHS} \
                --batch_size ${SFT_BATCH_SIZE} \
                --learning_rate ${SFT_LR} \
                --max_seq_length ${SFT_MAX_SEQ} \
                --lora_r ${LORA_R} \
                --lora_alpha ${LORA_ALPHA} \
                ${QUANT_FLAG} \
                --logging_steps 50 \
                --save_steps 500 \
                --report_to tensorboard \
                2>&1 | tee "${EXP_LOG}/sft.log"
            echo "[Stage 1/4] SFT complete for ${MODEL_SHORT} x ${DATASET}"
        else
            echo "[Stage 1/4] SFT skipped (already exists or --skip-sft)"
        fi

        if [ -n "${SFT_ONLY}" ]; then
            continue
        fi

        # ------------------------------------------------------------------
        # Stage 2: Reward Model
        # ------------------------------------------------------------------
        if [ -z "${SKIP_REWARD}" ] && [ ! -f "${EXP_DIR}/reward_model/final/config.json" ]; then
            echo "[Stage 2/4] Reward Model Training: ${MODEL_SHORT} on ${DATASET}..."

            # Use SFT model as base if available, otherwise use base model
            RM_BASE="${EXP_DIR}/sft/final"
            if [ ! -d "${RM_BASE}" ]; then
                RM_BASE="${MODEL_ID}"
            fi

            python "${PROJECT_DIR}/scripts/train_reward.py" \
                --base_model "${RM_BASE}" \
                --dataset_path "${DATA_DIR}/${DATASET}/preference_train.json" \
                --eval_dataset_path "${DATA_DIR}/${DATASET}/preference_eval.json" \
                --output_dir "${EXP_DIR}/reward_model" \
                --num_epochs ${RM_EPOCHS} \
                --batch_size ${RM_BATCH_SIZE} \
                --learning_rate ${RM_LR} \
                --max_seq_length 512 \
                --lora_r ${LORA_R} \
                --lora_alpha ${LORA_ALPHA} \
                ${QUANT_FLAG} \
                --logging_steps 50 \
                --save_steps 200 \
                --report_to tensorboard \
                2>&1 | tee "${EXP_LOG}/reward_model.log"
            echo "[Stage 2/4] Reward model complete for ${MODEL_SHORT} x ${DATASET}"
        else
            echo "[Stage 2/4] Reward model skipped"
        fi

        # ------------------------------------------------------------------
        # Stage 3: PPO
        # ------------------------------------------------------------------
        if [ -z "${SKIP_PPO}" ] && [ ! -f "${EXP_DIR}/ppo/final/config.json" ]; then
            echo "[Stage 3/4] PPO Training: ${MODEL_SHORT} on ${DATASET}..."

            python "${PROJECT_DIR}/scripts/train_ppo.py" \
                --policy_model "${EXP_DIR}/sft/final" \
                --reward_model "${EXP_DIR}/reward_model/final" \
                --dataset_path "${DATA_DIR}/${DATASET}/sft_train.json" \
                --output_dir "${EXP_DIR}/ppo" \
                --num_steps ${PPO_STEPS} \
                --batch_size ${PPO_BATCH_SIZE} \
                --mini_batch_size ${PPO_MINI_BATCH} \
                --learning_rate ${PPO_LR} \
                --kl_penalty ${PPO_KL} \
                --lora_r ${LORA_R} \
                --max_prompts 5000 \
                --logging_steps 10 \
                --save_steps 100 \
                2>&1 | tee "${EXP_LOG}/ppo.log"
            echo "[Stage 3/4] PPO complete for ${MODEL_SHORT} x ${DATASET}"
        else
            echo "[Stage 3/4] PPO skipped"
        fi

        # ------------------------------------------------------------------
        # Stage 4: Evaluation
        # ------------------------------------------------------------------
        if [ -z "${SKIP_EVAL}" ]; then
            echo "[Stage 4/4] Evaluating: ${MODEL_SHORT} on ${DATASET}..."

            # Evaluate SFT model
            if [ -d "${EXP_DIR}/sft/final" ]; then
                python "${PROJECT_DIR}/scripts/evaluate.py" \
                    --model_path "${EXP_DIR}/sft/final" \
                    --eval_dataset "${DATA_DIR}/${DATASET}/sft_eval.json" \
                    --output_dir "${EXP_DIR}/eval_sft" \
                    --reward_model_path "${EXP_DIR}/reward_model/final" \
                    --max_samples 200 \
                    --batch_size 4 \
                    2>&1 | tee "${EXP_LOG}/eval_sft.log"
            fi

            # Evaluate PPO model
            if [ -d "${EXP_DIR}/ppo/final" ]; then
                python "${PROJECT_DIR}/scripts/evaluate.py" \
                    --model_path "${EXP_DIR}/ppo/final" \
                    --eval_dataset "${DATA_DIR}/${DATASET}/sft_eval.json" \
                    --output_dir "${EXP_DIR}/eval_ppo" \
                    --reward_model_path "${EXP_DIR}/reward_model/final" \
                    --max_samples 200 \
                    --batch_size 4 \
                    2>&1 | tee "${EXP_LOG}/eval_ppo.log"
            fi

            echo "[Stage 4/4] Evaluation complete for ${MODEL_SHORT} x ${DATASET}"
        fi

        EXP_END=$(date +%s)
        EXP_DURATION=$(( (EXP_END - EXP_START) / 60 ))
        echo "[${EXPERIMENT_NUM}/${TOTAL_EXPERIMENTS}] ${MODEL_SHORT} x ${DATASET} done in ${EXP_DURATION} minutes"
    done
done

# ==============================================================================
# Aggregate results
# ==============================================================================
echo ""
echo "============================================================"
echo "Aggregating results..."
echo "============================================================"

python -c "
import json, os, glob

results_dir = '${RESULTS_DIR}'
output_dir = '${OUTPUT_DIR}'

all_results = []
for eval_dir in sorted(glob.glob(os.path.join(output_dir, '*/*/eval_*/evaluation_results.json'))):
    with open(eval_dir) as f:
        result = json.load(f)
    parts = eval_dir.split('/')
    model = parts[-4]
    dataset = parts[-3]
    stage = 'sft' if 'eval_sft' in eval_dir else 'ppo'
    result['model'] = model
    result['dataset'] = dataset
    result['stage'] = stage
    all_results.append(result)

with open(os.path.join(results_dir, 'all_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2)

# Print summary table
print(f'\n{\"Model\":<20} {\"Dataset\":<15} {\"Stage\":<6} {\"Perplexity\":<12} {\"Reward\":<10}')
print('-' * 65)
for r in all_results:
    ppl = r.get('metrics', {}).get('perplexity', 'N/A')
    rew = r.get('metrics', {}).get('reward_mean', 'N/A')
    ppl_str = f'{ppl:.2f}' if isinstance(ppl, float) else str(ppl)
    rew_str = f'{rew:.4f}' if isinstance(rew, float) else str(rew)
    print(f'{r[\"model\"]:<20} {r[\"dataset\"]:<15} {r[\"stage\"]:<6} {ppl_str:<12} {rew_str:<10}')
"

PIPELINE_END=$(date +%s)
TOTAL_DURATION=$(( (PIPELINE_END - PIPELINE_START) / 60 ))

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "Total time: ${TOTAL_DURATION} minutes"
echo "Results: ${RESULTS_DIR}/all_results.json"
echo "============================================================"
