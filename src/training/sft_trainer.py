#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque
"""

"""
Supervised Fine-Tuning (SFT) Trainer for SLM-RL-Agent

This module implements Stage 1 of the RLHF pipeline: Supervised Fine-Tuning.
SFT trains the base language model to follow instructions using demonstration data,
creating the foundation for subsequent alignment stages.

The SFT stage is critical because:
1. It teaches the model the desired response format and style
2. It provides a stable starting point for reward model training
3. It serves as the reference model for KL divergence constraints in PPO/DPO

Key Features:
    - Integration with HuggingFace TRL's SFTTrainer
    - Support for LoRA/QLoRA for efficient training
    - Packing for improved training efficiency
    - NEFTune noise for regularization
    - Completion-only loss masking

References:
    - Ouyang et al. (2022): Training language models to follow instructions
    - LIMA (Zhou et al., 2023): Less Is More for Alignment
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

logger = logging.getLogger(__name__)


@dataclass
class SFTTrainingConfig:
    """
    Configuration for Supervised Fine-Tuning.
    
    This dataclass encapsulates all hyperparameters and settings needed for SFT.
    Default values are optimized for small language models (30M-500M parameters).
    
    Attributes:
        model_name_or_path: Base model to fine-tune
        output_dir: Directory to save checkpoints and final model
        
        # Training hyperparameters
        learning_rate: Peak learning rate (2e-5 is standard for SFT)
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per GPU
        gradient_accumulation_steps: Steps before optimizer update
        max_seq_length: Maximum sequence length for training
        
        # Efficiency settings
        use_lora: Enable LoRA for parameter-efficient training
        lora_r: LoRA rank (8-32 typical for SLMs)
        lora_alpha: LoRA scaling factor
        use_quantization: Enable 4-bit quantization (QLoRA)
        gradient_checkpointing: Trade compute for memory savings
        
        # Advanced features
        packing: Pack multiple sequences per training example
        neftune_noise_alpha: NEFTune regularization strength (0 to disable)
        completion_only: Only compute loss on response tokens
    """
    # Model settings
    model_name_or_path: str = "EleutherAI/pythia-160m-deduped"
    output_dir: str = "./outputs/sft"
    
    # Training hyperparameters
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 1024
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Learning rate scheduler
    lr_scheduler_type: str = "cosine"
    
    # Efficiency settings
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    
    use_quantization: bool = True
    quantization_bits: int = 4
    
    gradient_checkpointing: bool = True
    
    # SFT-specific features
    packing: bool = True
    neftune_noise_alpha: float = 5.0
    completion_only: bool = False
    response_template: str = "### Response:"
    
    # Logging and saving
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 3
    
    # Mixed precision
    bf16: bool = True
    fp16: bool = False
    
    # Misc
    seed: int = 42
    dataloader_num_workers: int = 4
    
    # W&B logging
    report_to: str = "tensorboard"
    run_name: Optional[str] = None


class SFTTrainerWrapper:
    """
    High-level wrapper for Supervised Fine-Tuning.
    
    This class provides a simplified interface for SFT training,
    handling model loading, dataset preparation, and training orchestration.
    
    Example:
        >>> config = SFTTrainingConfig(
        ...     model_name_or_path="EleutherAI/pythia-160m-deduped",
        ...     output_dir="./outputs/sft",
        ...     num_train_epochs=3
        ... )
        >>> trainer = SFTTrainerWrapper(config)
        >>> trainer.train(train_dataset, eval_dataset)
        >>> trainer.save_model()
    """
    
    def __init__(self, config: SFTTrainingConfig):
        """
        Initialize the SFT trainer with the given configuration.
        
        Args:
            config: SFTTrainingConfig with all training settings
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Load model and tokenizer
        self._load_model_and_tokenizer()
    
    def _load_model_and_tokenizer(self):
        """Load and configure the model and tokenizer."""
        logger.info(f"Loading model: {self.config.model_name_or_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True,
        )
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {self.tokenizer.pad_token}")
        
        # Build model loading kwargs
        model_kwargs = self._build_model_kwargs()
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            **model_kwargs,
        )
        
        # Prepare for quantized training if using QLoRA
        if self.config.use_quantization:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.config.gradient_checkpointing,
            )
        elif self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Apply LoRA if enabled
        if self.config.use_lora:
            self.model = self._apply_lora()
        
        # Log parameter counts
        self._log_model_info()
    
    def _build_model_kwargs(self) -> Dict[str, Any]:
        """Build kwargs for model loading."""
        kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
        }
        
        # Set dtype
        if self.config.bf16:
            kwargs["torch_dtype"] = torch.bfloat16
        elif self.config.fp16:
            kwargs["torch_dtype"] = torch.float16
        
        # Configure quantization
        if self.config.use_quantization:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=(self.config.quantization_bits == 4),
                load_in_8bit=(self.config.quantization_bits == 8),
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        
        # Try to use Flash Attention
        try:
            kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:
            kwargs["attn_implementation"] = "sdpa"
        
        return kwargs
    
    def _apply_lora(self) -> PreTrainedModel:
        """Apply LoRA adapters to the model."""
        logger.info(f"Applying LoRA with r={self.config.lora_r}, alpha={self.config.lora_alpha}")
        
        target_modules = self.config.lora_target_modules
        if target_modules is None:
            target_modules = "all-linear"
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        return get_peft_model(self.model, lora_config)
    
    def _log_model_info(self):
        """Log information about the model."""
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"Model loaded successfully:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        resume_from_checkpoint: Optional[str] = None,
    ):
        """
        Run the SFT training loop.
        
        Args:
            train_dataset: Dataset with 'text' column for training
            eval_dataset: Optional evaluation dataset
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        logger.info("Starting SFT training...")
        logger.info(f"Train dataset size: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Eval dataset size: {len(eval_dataset)}")
        
        # Build training arguments
        training_args = self._build_training_args()
        
        # Build data collator for completion-only loss if enabled
        data_collator = None
        if self.config.completion_only:
            data_collator = DataCollatorForCompletionOnlyLM(
                response_template=self.config.response_template,
                tokenizer=self.tokenizer,
            )
        
        # Create SFT trainer
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            max_seq_length=self.config.max_seq_length,
            packing=self.config.packing,
            neftune_noise_alpha=self.config.neftune_noise_alpha if self.config.neftune_noise_alpha > 0 else None,
            dataset_text_field="text",
        )
        
        # Train
        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Log final metrics
        logger.info("Training completed!")
        if self.trainer.state.log_history:
            final_loss = self.trainer.state.log_history[-1].get("train_loss", "N/A")
            logger.info(f"Final training loss: {final_loss}")
    
    def _build_training_args(self) -> SFTConfig:
        """Build SFTConfig from our configuration."""
        return SFTConfig(
            output_dir=self.config.output_dir,
            
            # Training hyperparameters
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            
            # Learning rate schedule
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            weight_decay=self.config.weight_decay,
            
            # Mixed precision
            bf16=self.config.bf16,
            fp16=self.config.fp16,
            
            # Gradient settings
            max_grad_norm=1.0,
            
            # Logging and saving
            logging_steps=self.config.logging_steps,
            eval_strategy="steps" if self.config.eval_steps > 0 else "no",
            eval_steps=self.config.eval_steps if self.config.eval_steps > 0 else None,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=True if self.config.eval_steps > 0 else False,
            
            # Other settings
            seed=self.config.seed,
            dataloader_num_workers=self.config.dataloader_num_workers,
            report_to=self.config.report_to,
            run_name=self.config.run_name,
            
            # Disable default behaviors we handle ourselves
            remove_unused_columns=False,
        )
    
    def save_model(self, output_dir: Optional[str] = None):
        """
        Save the trained model.
        
        Args:
            output_dir: Directory to save to (uses config.output_dir/final if not specified)
        """
        if output_dir is None:
            output_dir = os.path.join(self.config.output_dir, "final")
        
        logger.info(f"Saving model to {output_dir}")
        
        # Save model
        self.trainer.save_model(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info("Model saved successfully!")
    
    def evaluate(self, eval_dataset: Dataset) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            eval_dataset: Dataset to evaluate on
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.trainer is None:
            raise ValueError("Must call train() before evaluate()")
        
        return self.trainer.evaluate(eval_dataset)


def train_sft(
    model_name_or_path: str,
    train_dataset: Dataset,
    output_dir: str,
    eval_dataset: Optional[Dataset] = None,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_seq_length: int = 1024,
    use_lora: bool = True,
    lora_r: int = 16,
    use_quantization: bool = True,
    **kwargs,
) -> str:
    """
    Convenience function to run SFT training.
    
    This function provides a simple interface for common SFT training scenarios.
    For more control, use SFTTrainerWrapper directly.
    
    Args:
        model_name_or_path: Base model to fine-tune
        train_dataset: Training dataset with 'text' column
        output_dir: Directory to save the trained model
        eval_dataset: Optional evaluation dataset
        num_epochs: Number of training epochs
        batch_size: Batch size per device
        learning_rate: Learning rate
        max_seq_length: Maximum sequence length
        use_lora: Whether to use LoRA
        lora_r: LoRA rank
        use_quantization: Whether to use 4-bit quantization
        **kwargs: Additional arguments passed to SFTTrainingConfig
    
    Returns:
        Path to the saved model directory
    
    Example:
        >>> from src.data import load_sft_dataset
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
        >>> dataset = load_sft_dataset("HuggingFaceH4/ultrachat_200k", tokenizer)
        >>> model_path = train_sft(
        ...     model_name_or_path="EleutherAI/pythia-160m-deduped",
        ...     train_dataset=dataset,
        ...     output_dir="./outputs/sft",
        ...     num_epochs=3
        ... )
    """
    config = SFTTrainingConfig(
        model_name_or_path=model_name_or_path,
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        use_lora=use_lora,
        lora_r=lora_r,
        use_quantization=use_quantization,
        **kwargs,
    )
    
    trainer = SFTTrainerWrapper(config)
    trainer.train(train_dataset, eval_dataset)
    
    final_model_path = os.path.join(output_dir, "final")
    trainer.save_model(final_model_path)
    
    return final_model_path
