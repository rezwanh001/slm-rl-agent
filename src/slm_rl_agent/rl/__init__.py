"""
RL Training Pipelines for SLM-RL-Agents (formerly src/training).

# README-এর Project Structure ব্লক অনুযায়ী এই প্যাকেজের নাম `rl`। আগের
# `src/training/` পথটি এখন `src/slm_rl_agent/rl/` — শুধু path বদলেছে,
# trainer-এর logic হুবহু একই।

This module provides trainers for all stages of the RLHF pipeline:
    - SFTTrainer: Supervised Fine-Tuning on instruction data
    - RewardTrainer: Training reward models on preference data
    - PPOTrainer: Proximal Policy Optimization with learned rewards
    - DPOTrainer: Direct Preference Optimization (simpler alternative to PPO)
    - GRPOTrainer: Group Relative Policy Optimization (advanced)

Each trainer is designed to work efficiently with small language models,
supporting LoRA, quantization, and gradient checkpointing.
"""

# পূর্বের পথ:
# from src.training.sft_trainer    import SFTTrainerWrapper, train_sft
# from src.training.reward_trainer import RewardTrainerWrapper, train_reward_model
# from src.training.ppo_trainer    import PPOTrainerWrapper, train_ppo
# from src.training.dpo_trainer    import DPOTrainerWrapper, train_dpo
# from src.training.grpo_trainer   import GRPOTrainer, GRPOConfig, create_verifiable_reward_function
from src.slm_rl_agent.rl.sft_trainer import SFTTrainerWrapper, train_sft
from src.slm_rl_agent.rl.reward_trainer import RewardTrainerWrapper, train_reward_model
from src.slm_rl_agent.rl.ppo_trainer import PPOTrainerWrapper, train_ppo
from src.slm_rl_agent.rl.dpo_trainer import DPOTrainerWrapper, train_dpo
from src.slm_rl_agent.rl.grpo_trainer import GRPOTrainer, GRPOConfig, create_verifiable_reward_function

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
