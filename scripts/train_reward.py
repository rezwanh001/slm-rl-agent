#!/usr/bin/env python3
"""
Reward Model Training Script for SLM-RL-Agent

This script runs Stage 2 of the RLHF pipeline: Reward Model Training.
The reward model learns to predict human preferences from comparison data,
providing the reward signal for policy optimization.

Usage:
    python scripts/train_reward.py \
        --base_model "./outputs/sft/final" \
        --dataset_path "./data/preference_train.json" \
        --output_dir "./outputs/reward_model"

The reward model is trained using the Bradley-Terry preference model:
    P(chosen > rejected | prompt) = sigmoid(r_chosen - r_rejected)

This converts preference comparisons into a scalar reward function.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModel,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


class RewardModelTrainer(Trainer):
    """Custom trainer for reward model with preference loss."""
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute Bradley-Terry preference loss."""
        # Forward pass for chosen
        chosen_outputs = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"],
        )
        chosen_rewards = chosen_outputs.logits[:, 0]  # Get scalar reward
        
        # Forward pass for rejected
        rejected_outputs = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"],
        )
        rejected_rewards = rejected_outputs.logits[:, 0]
        
        # Bradley-Terry loss: -log(sigmoid(r_chosen - r_rejected))
        loss = -torch.nn.functional.logsigmoid(chosen_rewards - rejected_rewards).mean()
        
        if return_outputs:
            return loss, {"chosen_rewards": chosen_rewards, "rejected_rewards": rejected_rewards}
        return loss


class PreferenceDataCollator:
    """Collator for preference pairs."""
    
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, features):
        batch = {
            "chosen_input_ids": [],
            "chosen_attention_mask": [],
            "rejected_input_ids": [],
            "rejected_attention_mask": [],
        }
        
        for f in features:
            # Tokenize chosen
            chosen_text = f"{f['prompt']}\n\n{f['chosen']}"
            chosen_enc = self.tokenizer(
                chosen_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            batch["chosen_input_ids"].append(chosen_enc["input_ids"].squeeze(0))
            batch["chosen_attention_mask"].append(chosen_enc["attention_mask"].squeeze(0))
            
            # Tokenize rejected
            rejected_text = f"{f['prompt']}\n\n{f['rejected']}"
            rejected_enc = self.tokenizer(
                rejected_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            batch["rejected_input_ids"].append(rejected_enc["input_ids"].squeeze(0))
            batch["rejected_attention_mask"].append(rejected_enc["attention_mask"].squeeze(0))
        
        # Stack tensors
        return {
            "chosen_input_ids": torch.stack(batch["chosen_input_ids"]),
            "chosen_attention_mask": torch.stack(batch["chosen_attention_mask"]),
            "rejected_input_ids": torch.stack(batch["rejected_input_ids"]),
            "rejected_attention_mask": torch.stack(batch["rejected_attention_mask"]),
        }


def load_preference_data(dataset_path: str) -> Dataset:
    """Load preference dataset."""
    logger.info(f"Loading preference data from {dataset_path}")
    
    if dataset_path.endswith(".json"):
        with open(dataset_path) as f:
            data = json.load(f)
        dataset = Dataset.from_list(data)
    else:
        dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    # Validate format
    required_cols = ["prompt", "chosen", "rejected"]
    for col in required_cols:
        if col not in dataset.column_names:
            raise ValueError(f"Dataset missing required column: {col}")
    
    logger.info(f"Loaded {len(dataset)} preference pairs")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Reward Model Training")
    
    # Model arguments
    parser.add_argument("--base_model", type=str, required=True,
                        help="Path to SFT model or base model")
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", action="store_false", dest="use_lora")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--use_quantization", action="store_true", default=True)
    parser.add_argument("--no_quantization", action="store_false", dest="use_quantization")
    
    # Data arguments
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--eval_dataset_path", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=512)
    
    # Training arguments
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
    
    # Save config
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model for sequence classification (outputs scalar reward)
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "num_labels": 1,  # Scalar reward output
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
        args.base_model,
        **model_kwargs,
    )
    
    # Set pad token id for the model
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Prepare for training
    if args.use_quantization:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    # Apply LoRA
    if args.use_lora:
        logger.info(f"Applying LoRA with r={args.lora_r}")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Load data
    train_dataset = load_preference_data(args.dataset_path)
    eval_dataset = None
    if args.eval_dataset_path:
        eval_dataset = load_preference_data(args.eval_dataset_path)
    
    # Create data collator
    data_collator = PreferenceDataCollator(tokenizer, args.max_seq_length)
    
    # Training arguments
    training_args = TrainingArguments(
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
        remove_unused_columns=False,  # Keep all columns for custom collator
    )
    
    # Create trainer
    trainer = RewardModelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    logger.info("Starting reward model training...")
    train_result = trainer.train()
    
    # Log metrics
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
