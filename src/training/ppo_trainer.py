"""
PPO Trainer for SLM-RL-Agent

This module implements Stage 3 of the RLHF pipeline: Policy Optimization with PPO.
Proximal Policy Optimization fine-tunes the SFT model using rewards from the learned
reward model while maintaining proximity to the original SFT policy via KL divergence.

The PPO objective maximizes:
    E[r_θ(x,y) - β * KL(π_θ || π_SFT)]

where r_θ is the reward, π_θ is the current policy, π_SFT is the reference policy,
and β controls the KL penalty strength.

Key Components:
    1. Actor (Policy): Generates responses given prompts
    2. Critic (Value): Estimates expected cumulative reward
    3. Reference Model: Frozen SFT model for KL computation
    4. Reward Model: Provides scalar rewards for responses

Training Loop:
    1. Sample prompts from dataset
    2. Generate responses using current policy
    3. Compute rewards using reward model
    4. Compute advantages using GAE
    5. Update policy with clipped PPO objective
    6. Update value function

References:
    - Schulman et al. (2017): Proximal Policy Optimization Algorithms
    - Ziegler et al. (2019): Fine-Tuning Language Models from Human Preferences
    - Zheng et al. (2023): Secrets of RLHF in Large Language Models Part I: PPO
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

logger = logging.getLogger(__name__)


@dataclass
class PPOTrainingConfig:
    """
    Configuration for PPO Training.
    
    This dataclass contains all hyperparameters for PPO training, with defaults
    optimized for small language models based on the "Secrets of RLHF" paper.
    
    Attributes:
        policy_model_path: Path to the SFT model (policy)
        reward_model_path: Path to the trained reward model
        output_dir: Directory for saving checkpoints
        
        # PPO hyperparameters
        learning_rate: Policy learning rate
        batch_size: Number of experiences per batch
        mini_batch_size: Mini-batch size for PPO updates
        ppo_epochs: Number of PPO epochs per batch
        
        # KL penalty
        init_kl_coef: Initial KL penalty coefficient (β)
        target_kl: Target KL divergence
        adap_kl_ctrl: Whether to adaptively adjust KL coefficient
        
        # GAE parameters
        gamma: Discount factor
        gae_lambda: GAE lambda for advantage estimation
        
        # Clipping
        cliprange: PPO clipping range for policy
        cliprange_value: Clipping range for value function
    """
    # Model paths
    policy_model_path: str = "./outputs/sft/final"
    reward_model_path: str = "./outputs/reward_model/final"
    output_dir: str = "./outputs/ppo"
    
    # PPO hyperparameters
    learning_rate: float = 1e-5
    batch_size: int = 64
    mini_batch_size: int = 8
    ppo_epochs: int = 4
    num_ppo_steps: int = 1000
    
    # KL penalty
    init_kl_coef: float = 0.1
    target_kl: float = 0.1
    adap_kl_ctrl: bool = True
    kl_penalty: str = "kl"
    
    # GAE
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # Clipping
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    
    # Value function
    vf_coef: float = 0.5
    
    # Generation settings
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    
    # Reward processing
    clip_reward: float = 10.0
    whiten_rewards: bool = False
    
    # Efficiency
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    use_quantization: bool = False  # PPO is memory-intensive, quantization can help
    gradient_checkpointing: bool = True
    gradient_accumulation_steps: int = 1
    
    # Logging
    logging_steps: int = 1
    save_steps: int = 100
    
    # Mixed precision
    bf16: bool = True
    
    # Misc
    seed: int = 42


class PPOTrainerWrapper:
    """
    High-level wrapper for PPO training in RLHF.
    
    This class orchestrates the complete PPO training loop:
    1. Load policy model (with value head) and reference model
    2. Load reward model
    3. Run PPO optimization loop
    4. Save the aligned model
    
    The implementation follows best practices from:
    - "Secrets of RLHF in Large Language Models Part I: PPO"
    - TRL library documentation
    
    Example:
        >>> config = PPOTrainingConfig(
        ...     policy_model_path="./outputs/sft/final",
        ...     reward_model_path="./outputs/reward_model/final",
        ...     output_dir="./outputs/ppo"
        ... )
        >>> trainer = PPOTrainerWrapper(config)
        >>> trainer.train(prompt_dataset)
        >>> trainer.save_model()
    """
    
    def __init__(self, config: PPOTrainingConfig):
        """Initialize PPO trainer with models and configuration."""
        self.config = config
        
        self.policy_model = None
        self.ref_model = None
        self.reward_model = None
        self.tokenizer = None
        self.ppo_trainer = None
        
        self._setup_models()
    
    def _setup_models(self):
        """Load and configure all models for PPO training."""
        logger.info("Setting up models for PPO training...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.policy_model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # Important for generation
        
        # Model loading kwargs
        model_kwargs = {"trust_remote_code": True}
        if self.config.bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
        
        if self.config.use_quantization:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = "auto"
        
        # Load policy model with value head
        logger.info(f"Loading policy model from: {self.config.policy_model_path}")
        self.policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.config.policy_model_path,
            **model_kwargs,
        )
        
        # Apply LoRA if enabled
        if self.config.use_lora:
            if self.config.use_quantization:
                self.policy_model = prepare_model_for_kbit_training(
                    self.policy_model,
                    use_gradient_checkpointing=self.config.gradient_checkpointing,
                )
            
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=0.05,
                target_modules="all-linear",
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.policy_model = get_peft_model(self.policy_model, lora_config)
        
        if self.config.gradient_checkpointing and not self.config.use_quantization:
            self.policy_model.gradient_checkpointing_enable()
        
        # Load reference model (frozen copy of SFT model)
        logger.info("Loading reference model...")
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            self.config.policy_model_path,
            **model_kwargs,
        )
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Load reward model
        logger.info(f"Loading reward model from: {self.config.reward_model_path}")
        self._load_reward_model()
        
        self._log_model_info()
    
    def _load_reward_model(self):
        """Load the reward model for scoring responses."""
        from transformers import AutoModelForSequenceClassification
        
        model_kwargs = {"trust_remote_code": True}
        if self.config.bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["device_map"] = "auto"
        
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.reward_model_path,
            num_labels=1,
            **model_kwargs,
        )
        self.reward_model.eval()
        
        # Freeze reward model
        for param in self.reward_model.parameters():
            param.requires_grad = False
    
    def _log_model_info(self):
        """Log information about loaded models."""
        trainable = sum(p.numel() for p in self.policy_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.policy_model.parameters())
        logger.info(f"Policy model: {total:,} total, {trainable:,} trainable")
    
    def _compute_rewards(
        self,
        queries: List[torch.Tensor],
        responses: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
        Compute rewards for query-response pairs using the reward model.
        
        Args:
            queries: List of tokenized queries
            responses: List of tokenized responses
            
        Returns:
            List of scalar rewards
        """
        rewards = []
        
        for query, response in zip(queries, responses):
            # Combine query and response
            full_input = torch.cat([query, response])
            
            # Prepare input for reward model
            input_ids = full_input.unsqueeze(0).to(self.reward_model.device)
            attention_mask = torch.ones_like(input_ids)
            
            # Get reward
            with torch.no_grad():
                outputs = self.reward_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                reward = outputs.logits.squeeze()
            
            # Clip reward
            reward = torch.clamp(reward, -self.config.clip_reward, self.config.clip_reward)
            rewards.append(reward)
        
        return rewards
    
    def train(
        self,
        prompt_dataset: Dataset,
        num_steps: Optional[int] = None,
    ):
        """
        Run PPO training loop.
        
        Args:
            prompt_dataset: Dataset with 'query' column containing prompts
            num_steps: Number of PPO steps (overrides config if provided)
        """
        logger.info("Starting PPO training...")
        
        num_steps = num_steps or self.config.num_ppo_steps
        
        # Build PPO config
        ppo_config = PPOConfig(
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            mini_batch_size=self.config.mini_batch_size,
            ppo_epochs=self.config.ppo_epochs,
            init_kl_coef=self.config.init_kl_coef,
            target_kl=self.config.target_kl,
            adap_kl_ctrl=self.config.adap_kl_ctrl,
            kl_penalty=self.config.kl_penalty,
            gamma=self.config.gamma,
            lam=self.config.gae_lambda,
            cliprange=self.config.cliprange,
            cliprange_value=self.config.cliprange_value,
            vf_coef=self.config.vf_coef,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_kwargs={"logging_dir": os.path.join(self.config.output_dir, "logs")},
            seed=self.config.seed,
        )
        
        # Create PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.policy_model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
            dataset=prompt_dataset,
        )
        
        # Generation kwargs
        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_k": self.config.top_k,
            "top_p": self.config.top_p,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        # Training loop
        logger.info(f"Running {num_steps} PPO steps...")
        
        for step, batch in enumerate(self.ppo_trainer.dataloader):
            if step >= num_steps:
                break
            
            # Get queries from batch
            query_tensors = batch["input_ids"]
            
            # Generate responses
            response_tensors = self.ppo_trainer.generate(
                query_tensors,
                **generation_kwargs,
            )
            
            # Decode for reward computation
            batch["response"] = self.tokenizer.batch_decode(
                response_tensors, skip_special_tokens=True
            )
            
            # Compute rewards
            rewards = self._compute_rewards(query_tensors, response_tensors)
            
            # Run PPO step
            stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)
            
            # Log
            if step % self.config.logging_steps == 0:
                logger.info(
                    f"Step {step}: "
                    f"reward={stats['ppo/mean_scores']:.4f}, "
                    f"kl={stats['objective/kl']:.4f}, "
                    f"entropy={stats['objective/entropy']:.4f}"
                )
            
            # Save checkpoint
            if step > 0 and step % self.config.save_steps == 0:
                self._save_checkpoint(step)
        
        logger.info("PPO training completed!")
    
    def _save_checkpoint(self, step: int):
        """Save a training checkpoint."""
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.ppo_trainer.save_pretrained(checkpoint_dir)
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    def save_model(self, output_dir: Optional[str] = None):
        """Save the final trained model."""
        if output_dir is None:
            output_dir = os.path.join(self.config.output_dir, "final")
        
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving final model to {output_dir}")
        self.ppo_trainer.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)


def train_ppo(
    policy_model_path: str,
    reward_model_path: str,
    prompt_dataset: Dataset,
    output_dir: str,
    num_steps: int = 1000,
    batch_size: int = 64,
    learning_rate: float = 1e-5,
    kl_penalty: float = 0.1,
    **kwargs,
) -> str:
    """
    Convenience function to run PPO training.
    
    Args:
        policy_model_path: Path to SFT model
        reward_model_path: Path to reward model
        prompt_dataset: Dataset with prompts for training
        output_dir: Output directory
        num_steps: Number of PPO steps
        batch_size: Batch size
        learning_rate: Learning rate
        kl_penalty: KL penalty coefficient
        **kwargs: Additional config parameters
        
    Returns:
        Path to saved model
    """
    config = PPOTrainingConfig(
        policy_model_path=policy_model_path,
        reward_model_path=reward_model_path,
        output_dir=output_dir,
        num_ppo_steps=num_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        init_kl_coef=kl_penalty,
        **kwargs,
    )
    
    trainer = PPOTrainerWrapper(config)
    trainer.train(prompt_dataset)
    
    final_path = os.path.join(output_dir, "final")
    trainer.save_model(final_path)
    
    return final_path
