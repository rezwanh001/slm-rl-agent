"""
Checkpoint Utilities for SLM-RL-Agent

This module provides utilities for saving and loading model checkpoints during training.
Proper checkpoint management is crucial for long training runs, enabling recovery from
failures and keeping track of the best performing models.
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: Any,
    tokenizer: Any,
    output_dir: str,
    step: int,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
    is_best: bool = False,
) -> str:
    """
    Save a training checkpoint.
    
    Args:
        model: The model to save
        tokenizer: The tokenizer
        output_dir: Directory to save the checkpoint
        step: Current training step
        metrics: Optional metrics at this checkpoint
        config: Optional configuration to save
        is_best: Whether this is the best checkpoint so far
    
    Returns:
        Path to the saved checkpoint
    """
    checkpoint_dir = Path(output_dir) / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(checkpoint_dir))
    else:
        torch.save(model.state_dict(), checkpoint_dir / "pytorch_model.bin")
    
    # Save tokenizer
    tokenizer.save_pretrained(str(checkpoint_dir))
    
    # Save metadata
    metadata = {
        "step": step,
        "metrics": metrics or {},
        "config": config or {},
    }
    with open(checkpoint_dir / "checkpoint_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # If best, copy to best_checkpoint directory
    if is_best:
        best_dir = Path(output_dir) / "best_checkpoint"
        if best_dir.exists():
            shutil.rmtree(best_dir)
        shutil.copytree(checkpoint_dir, best_dir)
        logger.info(f"Saved best checkpoint at step {step}")
    
    logger.info(f"Saved checkpoint to {checkpoint_dir}")
    return str(checkpoint_dir)


def load_checkpoint(
    checkpoint_path: str,
    model: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Load a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        model: Optional model to load weights into
    
    Returns:
        Dictionary with checkpoint data including metadata
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Load metadata
    metadata_path = checkpoint_path / "checkpoint_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Load model if provided
    if model is not None:
        model_path = checkpoint_path / "pytorch_model.bin"
        if model_path.exists():
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)
        elif hasattr(model, "from_pretrained"):
            # Try to load using from_pretrained
            pass  # Model should be loaded separately
    
    return {
        "path": str(checkpoint_path),
        "metadata": metadata,
    }


def get_best_checkpoint(output_dir: str) -> Optional[str]:
    """
    Get the path to the best checkpoint in the output directory.
    
    Args:
        output_dir: Directory containing checkpoints
    
    Returns:
        Path to the best checkpoint, or None if not found
    """
    best_dir = Path(output_dir) / "best_checkpoint"
    if best_dir.exists():
        return str(best_dir)
    
    # Fallback: find the latest checkpoint
    checkpoints = list(Path(output_dir).glob("checkpoint-*"))
    if checkpoints:
        # Sort by step number
        checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
        return str(checkpoints[-1])
    
    return None


def cleanup_checkpoints(
    output_dir: str,
    keep_last_n: int = 3,
    keep_best: bool = True,
) -> List[str]:
    """
    Clean up old checkpoints to save disk space.
    
    Args:
        output_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
        keep_best: Whether to keep the best checkpoint
    
    Returns:
        List of removed checkpoint paths
    """
    checkpoints = list(Path(output_dir).glob("checkpoint-*"))
    checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
    
    # Determine which to remove
    if len(checkpoints) <= keep_last_n:
        return []
    
    to_remove = checkpoints[:-keep_last_n]
    removed = []
    
    for checkpoint in to_remove:
        shutil.rmtree(checkpoint)
        removed.append(str(checkpoint))
        logger.info(f"Removed old checkpoint: {checkpoint}")
    
    return removed
