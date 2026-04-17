#!/bin/bash
# =============================================================================
# Full RLHF Pipeline Script for SLM-RL-Agents
# =============================================================================
# This script runs the complete RLHF training pipeline:
# 1. Data Preparation
# 2. Supervised Fine-Tuning (SFT)
# 3. Reward Model Training
# 4. PPO or DPO Training
# 5. Evaluation
#
# Usage:
#   ./scripts/run_full_pipeline.sh [model_size]
#   
#   model_size options:
#     - pythia-70m    (fastest, for testing)
#     - pythia-160m   (default, good balance)
#     - pythia-410m   (larger, better quality)
#     - smollm2-135m  (modern architecture)
#     - smollm2-360m  (best quality for size)
#
# Example:
#   ./scripts/run_full_pipeline.sh pythia-160m
# =============================================================================

set -e  # Exit on any error

# =============================================================================
# Configuration
# =============================================================================

# Model selection (default: pythia-160m)
MODEL_SIZE=${1:-pythia-160m}

# Map model size to HuggingFace model name
case $MODEL_SIZE in
    pythia-70m)
        MODEL_NAME="EleutherAI/pythia-70m-deduped"
        ;;
    pythia-160m)
        MODEL_NAME="EleutherAI/pythia-160m-deduped"
        ;;
    pythia-410m)
        MODEL_NAME="EleutherAI/pythia-410m-deduped"
        ;;
    smollm2-135m)
        MODEL_NAME="HuggingFaceTB/SmolLM2-135M"
        ;;
    smollm2-360m)
        MODEL_NAME="HuggingFaceTB/SmolLM2-360M"
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE"
        echo "Options: pythia-70m, pythia-160m, pythia-410m, smollm2-135m, smollm2-360m"
        exit 1
        ;;
esac

# Directories
DATA_DIR="./data"
OUTPUT_DIR="./outputs/${MODEL_SIZE}"
SFT_DIR="${OUTPUT_DIR}/sft"
REWARD_DIR="${OUTPUT_DIR}/reward_model"
PPO_DIR="${OUTPUT_DIR}/ppo"
DPO_DIR="${OUTPUT_DIR}/dpo"
EVAL_DIR="${OUTPUT_DIR}/evaluation"

# Training parameters (adjust based on GPU memory)
BATCH_SIZE=8
GRADIENT_ACCUM=4
MAX_SAMPLES=50000  # Set to null for full dataset
USE_DPO=false  # Set to true to use DPO instead of PPO

# =============================================================================
# Print Configuration
# =============================================================================

echo "=============================================="
echo "SLM-RL-Agents Full Pipeline"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "Output: $OUTPUT_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "Gradient Accumulation: $GRADIENT_ACCUM"
echo "Max Samples: $MAX_SAMPLES"
echo "Training Method: $([ "$USE_DPO" = true ] && echo "DPO" || echo "PPO")"
echo "=============================================="
echo ""

# Create directories
mkdir -p "$DATA_DIR" "$OUTPUT_DIR"

# =============================================================================
# Stage 0: Data Preparation
# =============================================================================

echo "=============================================="
echo "Stage 0: Preparing Data"
echo "=============================================="

if [ ! -f "${DATA_DIR}/sft_train.json" ]; then
    python scripts/prepare_data.py \
        --output_dir "$DATA_DIR" \
        --sft_dataset "HuggingFaceH4/ultrachat_200k" \
        --preference_dataset "HuggingFaceH4/ultrafeedback_binarized" \
        --max_samples "$MAX_SAMPLES" \
        --eval_ratio 0.05
    echo "Data preparation complete!"
else
    echo "Data already exists, skipping preparation."
fi

# =============================================================================
# Stage 1: Supervised Fine-Tuning (SFT)
# =============================================================================

echo ""
echo "=============================================="
echo "Stage 1: Supervised Fine-Tuning"
echo "=============================================="

if [ ! -d "${SFT_DIR}/final" ]; then
    python scripts/train_sft.py \
        --model_name "$MODEL_NAME" \
        --dataset_path "${DATA_DIR}/sft_train.json" \
        --eval_dataset_path "${DATA_DIR}/sft_eval.json" \
        --output_dir "$SFT_DIR" \
        --num_epochs 3 \
        --batch_size "$BATCH_SIZE" \
        --gradient_accumulation_steps "$GRADIENT_ACCUM" \
        --learning_rate 2e-5 \
        --max_seq_length 1024 \
        --use_lora \
        --lora_r 16 \
        --use_4bit \
        --packing \
        --logging_steps 10 \
        --save_steps 500 \
        --eval_steps 100 \
        --report_to tensorboard
    echo "SFT training complete!"
else
    echo "SFT model already exists, skipping training."
fi

# =============================================================================
# Stage 2: Reward Model Training
# =============================================================================

echo ""
echo "=============================================="
echo "Stage 2: Reward Model Training"
echo "=============================================="

if [ ! -d "${REWARD_DIR}/final" ]; then
    python scripts/train_reward.py \
        --base_model "${SFT_DIR}/final" \
        --dataset_path "${DATA_DIR}/preference_train.json" \
        --eval_dataset_path "${DATA_DIR}/preference_eval.json" \
        --output_dir "$REWARD_DIR" \
        --num_epochs 1 \
        --batch_size 4 \
        --gradient_accumulation_steps "$GRADIENT_ACCUM" \
        --learning_rate 1e-5 \
        --max_seq_length 512 \
        --use_lora \
        --lora_r 16 \
        --logging_steps 10 \
        --save_steps 200 \
        --report_to tensorboard
    echo "Reward model training complete!"
else
    echo "Reward model already exists, skipping training."
fi

# =============================================================================
# Stage 3: RL Training (PPO or DPO)
# =============================================================================

echo ""
echo "=============================================="
echo "Stage 3: RL Training"
echo "=============================================="

if [ "$USE_DPO" = true ]; then
    # DPO Training
    if [ ! -d "${DPO_DIR}/final" ]; then
        echo "Running DPO training..."
        python scripts/train_dpo.py \
            --model_path "${SFT_DIR}/final" \
            --dataset_path "${DATA_DIR}/preference_train.json" \
            --eval_dataset_path "${DATA_DIR}/preference_eval.json" \
            --output_dir "$DPO_DIR" \
            --num_epochs 1 \
            --batch_size 4 \
            --gradient_accumulation_steps "$GRADIENT_ACCUM" \
            --learning_rate 5e-7 \
            --beta 0.1 \
            --max_seq_length 1024 \
            --use_lora \
            --lora_r 16 \
            --logging_steps 10 \
            --save_steps 200 \
            --report_to tensorboard
        echo "DPO training complete!"
        FINAL_MODEL="${DPO_DIR}/final"
    else
        echo "DPO model already exists, skipping training."
        FINAL_MODEL="${DPO_DIR}/final"
    fi
else
    # PPO Training
    if [ ! -d "${PPO_DIR}/final" ]; then
        echo "Running PPO training..."
        python scripts/train_ppo.py \
            --policy_model "${SFT_DIR}/final" \
            --reward_model "${REWARD_DIR}/final" \
            --output_dir "$PPO_DIR" \
            --num_steps 1000 \
            --batch_size 64 \
            --mini_batch_size 8 \
            --learning_rate 1e-5 \
            --kl_penalty 0.1 \
            --clip_range 0.2 \
            --max_new_tokens 256 \
            --logging_steps 10 \
            --save_steps 200 \
            --report_to tensorboard
        echo "PPO training complete!"
        FINAL_MODEL="${PPO_DIR}/final"
    else
        echo "PPO model already exists, skipping training."
        FINAL_MODEL="${PPO_DIR}/final"
    fi
fi

# =============================================================================
# Stage 4: Evaluation
# =============================================================================

echo ""
echo "=============================================="
echo "Stage 4: Evaluation"
echo "=============================================="

python scripts/evaluate.py \
    --model_path "$FINAL_MODEL" \
    --eval_dataset "${DATA_DIR}/sft_eval.json" \
    --output_dir "$EVAL_DIR" \
    --num_samples 500 \
    --compute_perplexity \
    --compute_diversity \
    --run_benchmarks

echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo ""
echo "Results:"
echo "  - SFT Model: ${SFT_DIR}/final"
echo "  - Reward Model: ${REWARD_DIR}/final"
echo "  - Final Model: $FINAL_MODEL"
echo "  - Evaluation: $EVAL_DIR"
echo ""
echo "To test the model interactively:"
echo "  python scripts/run_agent.py --model_path $FINAL_MODEL --mode interactive"
echo ""
echo "To start the inference server:"
echo "  python scripts/run_agent.py --model_path $FINAL_MODEL --mode server --port 8000"
echo ""
