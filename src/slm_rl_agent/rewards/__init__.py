#-*- coding: utf-8 -*-
"""
Reward modelling package for SLM-RL-Agents.

# Bradley–Terry reward মডেলটি README-এর slm_rl_agent/rewards/ পথে রাখা হয়েছে
# (পূর্বে src/models/reward_model.py-তে ছিল)। পুরাতন import path-ও কমেন্ট আকারে
# সংরক্ষিত আছে — কোনো তথ্য হারায়নি।
"""

from src.slm_rl_agent.rewards.reward_model import RewardModel, RewardModelOutput

__all__ = [
    "RewardModel",
    "RewardModelOutput",
]
