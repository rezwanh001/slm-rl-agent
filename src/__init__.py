"""
SLM-RL-Agents: Efficient Small Language Model Agents with Reinforcement Learning

This package provides a complete framework for training Small Language Model (SLM) 
AI Agents using Reinforcement Learning from Human Feedback (RLHF).

Main Components:
    - data: Dataset loading and preprocessing utilities
    - models: Model architectures including reward models and value heads
    - training: Training pipelines for SFT, reward model, PPO, DPO
    - evaluation: Comprehensive evaluation metrics
    - agent: Production-ready agent implementation
    - utils: Utility functions for logging, checkpointing, etc.

Quick Start:
    >>> from src.agent import SLMAgent
    >>> agent = SLMAgent.from_pretrained("path/to/model")
    >>> response = agent.generate("Hello, how are you?")
    
References:
    - Ouyang et al. (2022): Training language models to follow instructions
    - Christiano et al. (2017): Deep reinforcement learning from human preferences
    - Rafailov et al. (2023): Direct Preference Optimization
"""

__version__ = "1.0.0"
__author__ = "SLM-RL-Agents Team"

from src.agent import SLMAgent
from src.models import SLMModel, RewardModel

__all__ = [
    "SLMAgent",
    "SLMModel", 
    "RewardModel",
    "__version__",
]
