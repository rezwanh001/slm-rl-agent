"""
Model architectures for SLM-RL-Agents.

This module provides model wrappers and custom architectures for:
    - Base SLM model loading with efficient training support
    - Reward models for RLHF
    - Value heads for PPO training
"""

# পূর্বের পথ:
# from src.models.slm_model    import SLMModel, load_model_for_training
# from src.models.reward_model import RewardModel, RewardModelOutput
# from src.models.value_head   import ValueHead, AutoModelForCausalLMWithValueHead
#
# RewardModel এখন `src/slm_rl_agent/rewards/` সাব-প্যাকেজে আলাদা করা হয়েছে
# (README-এর কাঠামোর সাথে মিল রাখতে); এখান থেকেও re-export রাখা হলো যাতে
# পুরাতন `from src.slm_rl_agent.models import RewardModel` কোডও কাজ করে।
from src.slm_rl_agent.models.slm_model import SLMModel, load_model_for_training
from src.slm_rl_agent.models.value_head import ValueHead, AutoModelForCausalLMWithValueHead
from src.slm_rl_agent.rewards.reward_model import RewardModel, RewardModelOutput

__all__ = [
    "SLMModel",
    "load_model_for_training",
    "RewardModel",
    "RewardModelOutput",
    "ValueHead",
    "AutoModelForCausalLMWithValueHead",
]
