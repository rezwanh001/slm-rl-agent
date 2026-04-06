"""
Configuration Utilities for SLM-RL-Agent

This module provides utilities for loading, saving, and managing configuration files.
Configuration management is essential for reproducible experiments and easy hyperparameter tuning.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a configuration file (YAML or JSON).
    
    Args:
        config_path: Path to the configuration file
    
    Returns:
        Dictionary with configuration values
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        if config_path.suffix in [".yaml", ".yml"]:
            config = yaml.safe_load(f)
        elif config_path.suffix == ".json":
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save a configuration to a file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save the configuration
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        if output_path.suffix in [".yaml", ".yml"]:
            yaml.dump(config, f, default_flow_style=False)
        elif output_path.suffix == ".json":
            json.dump(config, f, indent=2)
        else:
            # Default to YAML
            yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Saved configuration to {output_path}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.
    
    Values in override_config take precedence over base_config.
    
    Args:
        base_config: Base configuration
        override_config: Configuration with values to override
    
    Returns:
        Merged configuration dictionary
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def get_default_config(config_type: str) -> Dict[str, Any]:
    """
    Get default configuration for a specific training type.
    
    Args:
        config_type: Type of configuration ("sft", "reward", "ppo", "dpo")
    
    Returns:
        Default configuration dictionary
    """
    defaults = {
        "sft": {
            "learning_rate": 2e-5,
            "num_epochs": 3,
            "batch_size": 8,
            "max_seq_length": 1024,
            "warmup_ratio": 0.1,
            "use_lora": True,
            "lora_r": 16,
            "lora_alpha": 32,
        },
        "reward": {
            "learning_rate": 1e-5,
            "num_epochs": 1,
            "batch_size": 4,
            "max_seq_length": 512,
        },
        "ppo": {
            "learning_rate": 1e-5,
            "num_steps": 1000,
            "batch_size": 64,
            "mini_batch_size": 8,
            "kl_penalty": 0.1,
            "clip_range": 0.2,
        },
        "dpo": {
            "learning_rate": 5e-7,
            "num_epochs": 1,
            "batch_size": 4,
            "beta": 0.1,
        },
    }
    
    if config_type not in defaults:
        raise ValueError(f"Unknown config type: {config_type}. Available: {list(defaults.keys())}")
    
    return defaults[config_type].copy()
