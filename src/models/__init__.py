"""
Model architectures for SLM-RL-Agent.

This module provides model wrappers and custom architectures for:
    - Base SLM model loading with efficient training support
    - Reward models for RLHF
    - Value heads for PPO training
"""

from src.models.slm_model import SLMModel, load_model_for_training
from src.models.reward_model import RewardModel, RewardModelOutput
from src.models.value_head import ValueHead, AutoModelForCausalLMWithValueHead

__all__ = [
    "SLMModel",
    "load_model_for_training",
    "RewardModel",
    "RewardModelOutput",
    "ValueHead",
    "AutoModelForCausalLMWithValueHead",
]
