#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque
"""

"""
DPO Training Script for SLM-RL-Agent

This script implements Direct Preference Optimization (DPO), an alternative to PPO
that directly optimizes the policy on preference data without needing a separate
reward model. DPO is simpler to implement and often achieves comparable results.

Usage:
    python scripts/train_dpo.py \
        --model_path "./outputs/sft/final" \
        --dataset_path "./data/preference_train.json" \
        --output_dir "./outputs/dpo" \
        --beta 0.1

DPO derives from the observation that the optimal policy can be expressed in
closed form given the reward function. This allows us to directly optimize:
    L_DPO = -E[log σ(β log(π_θ(y_w|x)/π_ref(y_w|x)) - β log(π_θ(y_l|x)/π_ref(y_l|x)))]

Where y_w is the preferred (winning) response and y_l is the rejected (losing) response.

References:
    Rafailov et al. (2023): Direct Preference Optimization: Your Language Model
    is Secretly a Reward Model
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_preference_data(dataset_path: str) -> Dataset:
    """Load and prepare preference dataset for DPO."""
    logger.info(f"Loading preference data from {dataset_path}")
    
    if dataset_path.endswith(".json"):
        with open(dataset_path) as f:
            data = json.load(f)
        dataset = Dataset.from_list(data)
    else:
        dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    # Ensure required columns exist
    required = ["prompt", "chosen", "rejected"]
    for col in required:
        if col not in dataset.column_names:
            raise ValueError(f"Dataset missing required column: {col}")
    
    logger.info(f"Loaded {len(dataset)} preference pairs")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="DPO Training for SLM-RL-Agent")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to SFT model")
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", action="store_false", dest="use_lora")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--use_quantization", action="store_true", default=True)
    parser.add_argument("--no_quantization", action="store_false", dest="use_quantization")
    
    # Data arguments
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--eval_dataset_path", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    
    # DPO arguments
    parser.add_argument("--output_dir", type=str, default="./outputs/dpo")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta parameter (KL penalty strength)")
    parser.add_argument("--loss_type", type=str, default="sigmoid",
                        choices=["sigmoid", "hinge", "ipo", "kto_pair"],
                        help="DPO loss variant")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Model loading configuration
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    
    if args.use_quantization:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    
    # Prepare for training
    if args.use_quantization:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
    
    # Apply LoRA
    if args.use_lora:
        logger.info(f"Applying LoRA with r={args.lora_r}")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None
    
    # Load reference model (for DPO, we need the original model for KL computation)
    # If using LoRA, the reference model is implicitly the frozen base weights
    ref_model = None
    if not args.use_lora:
        logger.info("Loading reference model...")
        ref_model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    
    # Load datasets
    train_dataset = load_preference_data(args.dataset_path)
    eval_dataset = None
    if args.eval_dataset_path:
        eval_dataset = load_preference_data(args.eval_dataset_path)
    
    # DPO configuration
    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.save_steps if eval_dataset else None,
        load_best_model_at_end=eval_dataset is not None,
        report_to=args.report_to if args.report_to != "none" else None,
        seed=args.seed,
        # DPO-specific
        beta=args.beta,
        loss_type=args.loss_type,
        label_smoothing=args.label_smoothing,
        max_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
    )
    
    # Create DPO trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )
    
    # Train
    logger.info("Starting DPO training...")
    logger.info(f"Beta (KL penalty): {args.beta}")
    logger.info(f"Loss type: {args.loss_type}")
    
    train_result = trainer.train()
    
    # Log metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Save final model
    final_dir = os.path.join(args.output_dir, "final")
    logger.info(f"Saving model to {final_dir}")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    logger.info("DPO training complete!")
    logger.info(f"Model saved to: {final_dir}")


if __name__ == "__main__":
    main()
