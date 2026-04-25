#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque
"""

"""
DPO Trainer for SLM-RL-Agents

This module implements Direct Preference Optimization (DPO), a simpler and more stable
alternative to PPO for aligning language models with human preferences.

DPO eliminates the need for a separate reward model by reparameterizing the RLHF objective.
Instead of training a reward model and then using RL, DPO directly optimizes the policy
to satisfy preferences using a classification-style loss.

The DPO loss is:
    L_DPO = -E[log σ(β(log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]

where y_w is the preferred response, y_l is the less preferred response, π is the
policy being trained, π_ref is the reference policy (SFT model), and β controls
the deviation from the reference.

Advantages over PPO:
    - No separate reward model needed
    - No value function needed
    - More stable training (no RL instabilities)
    - 2-3x faster training
    - Simpler hyperparameter tuning

References:
    - Rafailov et al. (2023): Direct Preference Optimization: Your Language Model 
      is Secretly a Reward Model
    - Tunstall et al. (2023): Zephyr: Direct Distillation of LM Alignment
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

logger = logging.getLogger(__name__)


@dataclass
class DPOTrainingConfig:
    """
    Configuration for Direct Preference Optimization training.
    
    DPO is simpler than PPO but requires careful tuning of the beta parameter
    which controls how much the model can deviate from the reference policy.
    
    Attributes:
        model_name_or_path: Path to SFT model to align
        output_dir: Directory for saving the aligned model
        
        # DPO hyperparameters
        beta: KL penalty strength (0.1-0.5 typical range)
            - Higher = more conservative, stays closer to reference
            - Lower = allows more deviation, can lead to reward hacking
        learning_rate: Very low learning rates work best (1e-7 to 5e-6)
        
        # Loss variants
        loss_type: "sigmoid" (standard), "hinge", "ipo", "kto_pair"
    """
    # Model settings
    model_name_or_path: str = "./outputs/sft/final"
    output_dir: str = "./outputs/dpo"
    
    # DPO hyperparameters
    beta: float = 0.1
    learning_rate: float = 5e-7
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 1024
    max_prompt_length: int = 512
    
    # Learning rate schedule
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # DPO-specific settings
    loss_type: str = "sigmoid"  # sigmoid, hinge, ipo, kto_pair
    label_smoothing: float = 0.0
    
    # Reference model
    use_ref_model: bool = True
    precompute_ref_log_probs: bool = False
    
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
    save_steps: int = 100
    save_total_limit: int = 3
    
    # Mixed precision
    bf16: bool = True
    
    # Misc
    seed: int = 42
    report_to: str = "tensorboard"
    max_steps: int = -1  # -1 means use num_epochs


class DPOTrainerWrapper:
    """
    High-level wrapper for DPO training.
    
    DPO is the recommended approach for SLM alignment due to its simplicity
    and stability. It directly learns from preference pairs without needing
    a separate reward model or value function.
    
    Training workflow:
        1. Load SFT model (policy) and create frozen reference copy
        2. Configure LoRA for efficient training
        3. Train on preference data using DPO loss
        4. Save aligned model
    
    Example:
        >>> config = DPOTrainingConfig(
        ...     model_name_or_path="./outputs/sft/final",
        ...     output_dir="./outputs/dpo",
        ...     beta=0.1
        ... )
        >>> trainer = DPOTrainerWrapper(config)
        >>> trainer.train(train_dataset, eval_dataset)
        >>> trainer.save_model()
    
    Tips for good results:
        - Start with beta=0.1, increase if model drifts too much
        - Use very low learning rates (1e-7 to 5e-6)
        - 1 epoch is usually sufficient
        - Ensure preference data quality is high
    """
    
    def __init__(self, config: DPOTrainingConfig):
        """Initialize DPO trainer."""
        self.config = config
        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.trainer = None
        
        self._setup_models()
    
    def _setup_models(self):
        """Load model, reference model, and tokenizer."""
        logger.info(f"Loading model from: {self.config.model_name_or_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Model loading kwargs
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }
        
        if self.config.bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
        
        if self.config.use_quantization:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        
        # Try to use Flash Attention
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:
            model_kwargs["attn_implementation"] = "sdpa"
        
        # Load policy model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            **model_kwargs,
        )
        
        # Prepare for training
        if self.config.use_quantization:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.config.gradient_checkpointing,
            )
        elif self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Apply LoRA
        if self.config.use_lora:
            logger.info(f"Applying LoRA with r={self.config.lora_r}, alpha={self.config.lora_alpha}")
            
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules="all-linear",
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)
        
        # Load reference model if using explicit reference
        if self.config.use_ref_model:
            logger.info("Loading reference model (frozen)...")
            
            # Reference model doesn't need LoRA or quantization overhead
            ref_kwargs = {"trust_remote_code": True, "device_map": "auto"}
            if self.config.bf16:
                ref_kwargs["torch_dtype"] = torch.bfloat16
            
            # For memory efficiency with QLoRA, we can keep ref model in 4-bit too
            if self.config.use_quantization:
                ref_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name_or_path,
                **ref_kwargs,
            )
            
            # Freeze reference model
            for param in self.ref_model.parameters():
                param.requires_grad = False
        else:
            self.ref_model = None
            logger.info("Using implicit reference (initial model weights)")
        
        self._log_model_info()
    
    def _log_model_info(self):
        """Log model information."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model: {total:,} total params, {trainable:,} trainable ({100*trainable/total:.2f}%)")
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
    ):
        """
        Run DPO training.
        
        Args:
            train_dataset: Dataset with 'prompt', 'chosen', 'rejected' columns
            eval_dataset: Optional evaluation dataset
            
        The dataset should have the following format:
            - prompt: The input prompt/question
            - chosen: The preferred response
            - rejected: The less preferred response
        """
        logger.info("Starting DPO training...")
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Beta (KL penalty): {self.config.beta}")
        
        # Build DPO config
        training_args = DPOConfig(
            output_dir=self.config.output_dir,
            
            # Core hyperparameters
            beta=self.config.beta,
            loss_type=self.config.loss_type,
            label_smoothing=self.config.label_smoothing,
            
            # Training settings
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            
            # Sequence lengths
            max_length=self.config.max_seq_length,
            max_prompt_length=self.config.max_prompt_length,
            
            # Learning rate schedule
            lr_scheduler_type=self.config.lr_scheduler_type,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            
            # Reference model settings
            precompute_ref_log_probs=self.config.precompute_ref_log_probs,
            
            # Mixed precision
            bf16=self.config.bf16,
            
            # Gradient settings
            max_grad_norm=1.0,
            
            # Logging and saving
            logging_steps=self.config.logging_steps,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=self.config.eval_steps if eval_dataset else None,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=True if eval_dataset else False,
            
            # Other settings
            seed=self.config.seed,
            report_to=self.config.report_to,
            max_steps=self.config.max_steps,
            
            remove_unused_columns=False,
        )
        
        # Create DPO trainer
        self.trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train
        self.trainer.train()
        
        # Log final metrics
        logger.info("DPO training completed!")
        if self.trainer.state.log_history:
            for key in ["train_loss", "eval_loss", "rewards/chosen", "rewards/rejected"]:
                for entry in reversed(self.trainer.state.log_history):
                    if key in entry:
                        logger.info(f"Final {key}: {entry[key]:.4f}")
                        break
    
    def save_model(self, output_dir: Optional[str] = None, merge_lora: bool = False):
        """
        Save the trained model.
        
        Args:
            output_dir: Directory to save to (default: config.output_dir/final)
            merge_lora: Whether to merge LoRA weights into base model
        """
        if output_dir is None:
            output_dir = os.path.join(self.config.output_dir, "final")
        
        logger.info(f"Saving model to {output_dir}")
        
        if merge_lora and self.config.use_lora:
            # Merge LoRA weights and save full model
            logger.info("Merging LoRA weights...")
            merged_model = self.model.merge_and_unload()
            merged_model.save_pretrained(output_dir)
        else:
            # Save with LoRA adapter
            self.trainer.save_model(output_dir)
        
        self.tokenizer.save_pretrained(output_dir)
        logger.info("Model saved successfully!")
    
    def evaluate(self, eval_dataset: Dataset) -> Dict[str, float]:
        """Evaluate the model on a dataset."""
        return self.trainer.evaluate(eval_dataset)


def train_dpo(
    model_name_or_path: str,
    train_dataset: Dataset,
    output_dir: str,
    eval_dataset: Optional[Dataset] = None,
    beta: float = 0.1,
    num_epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 5e-7,
    use_lora: bool = True,
    **kwargs,
) -> str:
    """
    Convenience function to run DPO training.
    
    This is the simplest way to align a model using DPO. For more control,
    use DPOTrainerWrapper directly.
    
    Args:
        model_name_or_path: Path to SFT model
        train_dataset: Preference dataset with 'prompt', 'chosen', 'rejected'
        output_dir: Output directory
        eval_dataset: Optional evaluation dataset
        beta: KL penalty strength (higher = more conservative)
        num_epochs: Number of training epochs
        batch_size: Batch size per device
        learning_rate: Learning rate (keep very low, 1e-7 to 5e-6)
        use_lora: Whether to use LoRA
        **kwargs: Additional config parameters
        
    Returns:
        Path to saved model
        
    Example:
        >>> from src.slm_rl_agent.data import load_preference_dataset  # পূর্বে: from src.data import ...
        >>> dataset = load_preference_dataset("HuggingFaceH4/ultrafeedback_binarized")
        >>> model_path = train_dpo(
        ...     model_name_or_path="./outputs/sft/final",
        ...     train_dataset=dataset,
        ...     output_dir="./outputs/dpo",
        ...     beta=0.1
        ... )
    """
    config = DPOTrainingConfig(
        model_name_or_path=model_name_or_path,
        output_dir=output_dir,
        beta=beta,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        use_lora=use_lora,
        **kwargs,
    )
    
    trainer = DPOTrainerWrapper(config)
    trainer.train(train_dataset, eval_dataset)
    
    final_path = os.path.join(output_dir, "final")
    trainer.save_model(final_path)
    
    return final_path
