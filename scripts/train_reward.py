#!/usr/bin/env python3
"""
Reward Model Training Script for SLM-RL-Agent

Uses TRL's RewardTrainer for robust preference-based training.

Usage:
    python scripts/train_reward.py \
        --base_model "./outputs/sft/final" \
        --dataset_path "./data/preference_train.json" \
        --output_dir "./outputs/reward_model"
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
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import RewardTrainer, RewardConfig
from peft import LoraConfig, prepare_model_for_kbit_training, TaskType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_preference_data(dataset_path: str, tokenizer, max_length: int = 512) -> Dataset:
    """Load and format preference dataset for TRL RewardTrainer."""
    logger.info(f"Loading preference data from {dataset_path}")

    if dataset_path.endswith(".json"):
        with open(dataset_path) as f:
            data = json.load(f)
        dataset = Dataset.from_list(data)
    else:
        dataset = load_dataset("json", data_files=dataset_path, split="train")

    # RewardTrainer expects columns: input_ids_chosen, attention_mask_chosen,
    # input_ids_rejected, attention_mask_rejected
    def tokenize_pair(example):
        chosen_text = f"{example['prompt']}\n\n{example['chosen']}"
        rejected_text = f"{example['prompt']}\n\n{example['rejected']}"

        chosen_enc = tokenizer(
            chosen_text, truncation=True, max_length=max_length, padding="max_length"
        )
        rejected_enc = tokenizer(
            rejected_text, truncation=True, max_length=max_length, padding="max_length"
        )

        return {
            "input_ids_chosen": chosen_enc["input_ids"],
            "attention_mask_chosen": chosen_enc["attention_mask"],
            "input_ids_rejected": rejected_enc["input_ids"],
            "attention_mask_rejected": rejected_enc["attention_mask"],
        }

    dataset = dataset.map(tokenize_pair, remove_columns=dataset.column_names)
    logger.info(f"Prepared {len(dataset)} preference pairs")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Reward Model Training")

    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", action="store_false", dest="use_lora")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--use_quantization", action="store_true", default=True)
    parser.add_argument("--no_quantization", action="store_false", dest="use_quantization")

    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--eval_dataset_path", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=512)

    parser.add_argument("--output_dir", type=str, default="./outputs/reward_model")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", type=str, default="tensorboard")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "num_labels": 1,
    }

    if args.use_quantization:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    logger.info(f"Loading model from {args.base_model}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, **model_kwargs
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.use_quantization:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # LoRA config
    peft_config = None
    if args.use_lora:
        logger.info(f"Will apply LoRA with r={args.lora_r}")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type=TaskType.SEQ_CLS,
        )

    # Load data
    train_dataset = load_preference_data(args.dataset_path, tokenizer, args.max_seq_length)
    eval_dataset = None
    if args.eval_dataset_path:
        eval_dataset = load_preference_data(args.eval_dataset_path, tokenizer, args.max_seq_length)

    # Training config
    reward_config = RewardConfig(
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
        remove_unused_columns=False,
        max_length=args.max_seq_length,
    )

    # Create trainer
    trainer = RewardTrainer(
        model=model,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    # Train
    logger.info("Starting reward model training...")
    train_result = trainer.train()

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Save final model
    final_dir = os.path.join(args.output_dir, "final")
    logger.info(f"Saving reward model to {final_dir}")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    logger.info("Reward model training complete!")
    logger.info(f"Model saved to: {final_dir}")


if __name__ == "__main__":
    main()
