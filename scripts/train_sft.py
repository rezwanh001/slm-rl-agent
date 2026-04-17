#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque
"""

"""
SFT Training Script for SLM-RL-Agents

This script runs Stage 1 of the RLHF pipeline: Supervised Fine-Tuning.
It takes a base language model and fine-tunes it on instruction-following data.

Usage:
    python scripts/train_sft.py \
        --model_name "EleutherAI/pythia-160m-deduped" \
        --dataset_path "./data/sft_train.json" \
        --output_dir "./outputs/sft" \
        --num_epochs 3

The trained model from this script will be used as:
1. The starting point for reward model initialization
2. The reference model for PPO/DPO KL constraints
3. The policy model to be further aligned
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_training_data(dataset_path: str) -> Dataset:
    """Load and prepare training data."""
    logger.info(f"Loading dataset from {dataset_path}")
    
    if dataset_path.endswith(".json"):
        with open(dataset_path) as f:
            data = json.load(f)
        dataset = Dataset.from_list(data)
    elif dataset_path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=dataset_path, split="train")
    else:
        # Assume it's a HuggingFace dataset
        dataset = load_dataset(dataset_path, split="train")
    
    logger.info(f"Loaded {len(dataset)} training examples")
    return dataset


def setup_model_and_tokenizer(
    model_name: str,
    use_lora: bool = True,
    use_quantization: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    """Load and configure model with optional LoRA and quantization."""
    logger.info(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure quantization
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    
    if use_quantization:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # Prepare for training
    if use_quantization:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
    
    # Apply LoRA
    if use_lora:
        logger.info(f"Applying LoRA with r={lora_r}, alpha={lora_alpha}")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="SFT Training for SLM-RL-Agents")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-160m-deduped",
                        help="Base model name or path")
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Use LoRA for efficient training")
    parser.add_argument("--no_lora", action="store_false", dest="use_lora")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--use_quantization", action="store_true", default=True,
                        help="Use 4-bit quantization")
    parser.add_argument("--no_quantization", action="store_false", dest="use_quantization")
    
    # Data arguments
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to training data")
    parser.add_argument("--eval_dataset_path", type=str, default=None,
                        help="Path to evaluation data")
    parser.add_argument("--max_seq_length", type=int, default=1024,
                        help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./outputs/sft",
                        help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Checkpoint save frequency")
    parser.add_argument("--packing", action="store_true", default=True,
                        help="Use packing for efficiency")
    parser.add_argument("--neftune_noise_alpha", type=float, default=5.0,
                        help="NEFTune noise alpha (0 to disable)")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--report_to", type=str, default="tensorboard",
                        help="Logging backend (tensorboard, wandb, none)")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Load data
    train_dataset = load_training_data(args.dataset_path)
    eval_dataset = None
    if args.eval_dataset_path:
        eval_dataset = load_training_data(args.eval_dataset_path)
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer(
        args.model_name,
        use_lora=args.use_lora,
        use_quantization=args.use_quantization,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    
    # Configure training
    sft_config = SFTConfig(
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
        # SFT-specific
        max_seq_length=args.max_seq_length,
        packing=args.packing,
        dataset_text_field="text",
        neftune_noise_alpha=args.neftune_noise_alpha if args.neftune_noise_alpha > 0 else None,
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # Train
    logger.info("Starting SFT training...")
    train_result = trainer.train()
    
    # Log final metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Save final model
    final_dir = os.path.join(args.output_dir, "final")
    logger.info(f"Saving final model to {final_dir}")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    logger.info("SFT training complete!")
    logger.info(f"Model saved to: {final_dir}")


if __name__ == "__main__":
    main()
