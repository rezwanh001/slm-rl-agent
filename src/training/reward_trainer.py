#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque
"""

"""
Reward Model Trainer for SLM-RL-Agent

This module implements Stage 2 of the RLHF pipeline: Reward Model Training.
The reward model learns to predict human preferences from comparison data,
providing a learned reward signal for subsequent policy optimization.

The reward model is typically initialized from the SFT checkpoint and trained
using the Bradley-Terry preference model:

    P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))

where y_w is the preferred response and y_l is the less preferred response.

Key Insights from InstructGPT:
    - Smaller reward models work well (6B RM for 175B policy)
    - Training on all K comparisons per prompt as one batch item prevents overfitting
    - Reward model accuracy correlates with downstream policy quality

References:
    - Christiano et al. (2017): Deep reinforcement learning from human preferences
    - Ouyang et al. (2022): Training language models to follow instructions
    - Stiennon et al. (2020): Learning to summarize from human feedback
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    PreTrainedModel,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import RewardTrainer, RewardConfig

logger = logging.getLogger(__name__)


@dataclass
class RewardTrainingConfig:
    """
    Configuration for Reward Model Training.
    
    Attributes:
        model_name_or_path: Base model (typically SFT checkpoint)
        output_dir: Directory for saving the reward model
        
        # Training hyperparameters
        learning_rate: Lower than SFT (1e-5 typical)
        num_train_epochs: Often 1 epoch is sufficient
        per_device_train_batch_size: Smaller batches due to paired data
        max_seq_length: Maximum length for prompt + response
        
        # Model architecture
        num_head_layers: Layers in reward head (1 = linear)
        
        # Efficiency
        use_lora: Use LoRA for efficient training
        use_quantization: Use 4-bit quantization
    """
    # Model settings
    model_name_or_path: str = "./outputs/sft/final"
    output_dir: str = "./outputs/reward_model"
    
    # Training hyperparameters  
    learning_rate: float = 1e-5
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 512
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Learning rate scheduler
    lr_scheduler_type: str = "cosine"
    
    # Reward model specific
    num_head_layers: int = 1
    center_rewards_coefficient: float = 0.01
    
    # Efficiency settings
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    use_quantization: bool = True
    gradient_checkpointing: bool = True
    
    # Logging and saving
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 200
    save_total_limit: int = 2
    
    # Mixed precision
    bf16: bool = True
    
    # Misc
    seed: int = 42
    report_to: str = "tensorboard"


class RewardTrainerWrapper:
    """
    High-level wrapper for Reward Model Training.
    
    This class handles the complete workflow of training a reward model:
    1. Load base model (from SFT checkpoint)
    2. Add reward head
    3. Configure LoRA if enabled
    4. Train on preference data
    5. Save the trained model
    
    Example:
        >>> config = RewardTrainingConfig(
        ...     model_name_or_path="./outputs/sft/final",
        ...     output_dir="./outputs/reward_model"
        ... )
        >>> trainer = RewardTrainerWrapper(config)
        >>> trainer.train(train_dataset, eval_dataset)
        >>> trainer.save_model()
    """
    
    def __init__(self, config: RewardTrainingConfig):
        """Initialize the reward model trainer."""
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        self._load_model_and_tokenizer()
    
    def _load_model_and_tokenizer(self):
        """Load the base model and add reward head."""
        logger.info(f"Loading base model from: {self.config.model_name_or_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Build model kwargs
        model_kwargs = {"device_map": "auto", "trust_remote_code": True}
        
        if self.config.bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
        
        if self.config.use_quantization:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        
        # Load model with sequence classification head (outputs single score)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name_or_path,
            num_labels=1,  # Single scalar reward
            **model_kwargs,
        )
        
        # Handle pad token embedding
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Prepare for quantized training
        if self.config.use_quantization:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.config.gradient_checkpointing,
            )
        elif self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Apply LoRA
        if self.config.use_lora:
            logger.info(f"Applying LoRA with r={self.config.lora_r}")
            
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules="all-linear",
                bias="none",
                task_type=TaskType.SEQ_CLS,
            )
            self.model = get_peft_model(self.model, lora_config)
        
        self._log_model_info()
    
    def _log_model_info(self):
        """Log model parameter information."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Reward model: {total:,} total, {trainable:,} trainable ({100*trainable/total:.2f}%)")
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
    ):
        """
        Train the reward model on preference data.
        
        Args:
            train_dataset: Dataset with 'prompt', 'chosen', 'rejected' columns
            eval_dataset: Optional evaluation dataset
        """
        logger.info("Starting reward model training...")
        logger.info(f"Training samples: {len(train_dataset)}")
        
        # Build training args
        training_args = RewardConfig(
            output_dir=self.config.output_dir,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            weight_decay=self.config.weight_decay,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=self.config.eval_steps if eval_dataset else None,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=True if eval_dataset else False,
            seed=self.config.seed,
            report_to=self.config.report_to,
            max_length=self.config.max_seq_length,
            center_rewards_coefficient=self.config.center_rewards_coefficient,
            remove_unused_columns=False,
        )
        
        # Create trainer
        self.trainer = RewardTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        # Train
        self.trainer.train()
        
        logger.info("Reward model training completed!")
    
    def save_model(self, output_dir: Optional[str] = None):
        """Save the trained reward model."""
        if output_dir is None:
            output_dir = os.path.join(self.config.output_dir, "final")
        
        logger.info(f"Saving reward model to {output_dir}")
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
    
    def evaluate(self, eval_dataset: Dataset) -> Dict[str, float]:
        """Evaluate the reward model."""
        return self.trainer.evaluate(eval_dataset)


def train_reward_model(
    model_name_or_path: str,
    train_dataset: Dataset,
    output_dir: str,
    eval_dataset: Optional[Dataset] = None,
    num_epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 1e-5,
    use_lora: bool = True,
    **kwargs,
) -> str:
    """
    Convenience function to train a reward model.
    
    Args:
        model_name_or_path: Path to SFT model or base model
        train_dataset: Preference dataset with 'prompt', 'chosen', 'rejected'
        output_dir: Where to save the reward model
        eval_dataset: Optional evaluation dataset
        num_epochs: Training epochs (1 is often enough)
        batch_size: Batch size per device
        learning_rate: Learning rate
        use_lora: Whether to use LoRA
        **kwargs: Additional config parameters
    
    Returns:
        Path to saved reward model
    """
    config = RewardTrainingConfig(
        model_name_or_path=model_name_or_path,
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        use_lora=use_lora,
        **kwargs,
    )
    
    trainer = RewardTrainerWrapper(config)
    trainer.train(train_dataset, eval_dataset)
    
    final_path = os.path.join(output_dir, "final")
    trainer.save_model(final_path)
    
    return final_path
