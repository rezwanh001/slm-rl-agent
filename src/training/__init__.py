"""
Training Pipelines for SLM-RL-Agent

This module provides trainers for all stages of the RLHF pipeline:
    - SFTTrainer: Supervised Fine-Tuning on instruction data
    - RewardTrainer: Training reward models on preference data
    - PPOTrainer: Proximal Policy Optimization with learned rewards
    - DPOTrainer: Direct Preference Optimization (simpler alternative to PPO)
    - GRPOTrainer: Group Relative Policy Optimization (advanced)

Each trainer is designed to work efficiently with small language models,
supporting LoRA, quantization, and gradient checkpointing.
"""

from src.training.sft_trainer import SFTTrainerWrapper, train_sft
from src.training.reward_trainer import RewardTrainerWrapper, train_reward_model
from src.training.ppo_trainer import PPOTrainerWrapper, train_ppo
from src.training.dpo_trainer import DPOTrainerWrapper, train_dpo
from src.training.grpo_trainer import GRPOTrainer, GRPOConfig, create_verifiable_reward_function

__all__ = [
    "SFTTrainerWrapper",
    "train_sft",
    "RewardTrainerWrapper", 
    "train_reward_model",
    "PPOTrainerWrapper",
    "train_ppo",
    "DPOTrainerWrapper",
    "train_dpo",
    "GRPOTrainer",
    "GRPOConfig",
    "create_verifiable_reward_function",
]
