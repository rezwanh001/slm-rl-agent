#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque
"""

"""
Data Preparation Script for SLM-RL-Agent

This script downloads and prepares datasets for all stages of RLHF training:
1. SFT datasets (instruction-following data)
2. Preference datasets (for reward model and DPO)
3. Evaluation datasets

Usage:
    python scripts/prepare_data.py --output_dir ./data

This script is the first step in the training pipeline. It handles the complexity
of downloading from HuggingFace, processing into standard formats, and creating
train/eval splits for consistent experimentation.
"""

import argparse
import json
import logging
import os
from pathlib import Path

from datasets import load_dataset, Dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def prepare_sft_dataset(
    dataset_name: str,
    output_dir: Path,
    max_samples: int = None,
) -> Path:
    """
    Download and prepare an SFT dataset.
    
    Args:
        dataset_name: HuggingFace dataset name
        output_dir: Directory to save processed data
        max_samples: Maximum number of samples to use
    
    Returns:
        Path to the saved dataset
    """
    logger.info(f"Preparing SFT dataset: {dataset_name}")
    
    # Load dataset based on name
    if "ultrachat" in dataset_name.lower():
        dataset = load_dataset(dataset_name, split="train_sft")
    else:
        dataset = load_dataset(dataset_name, split="train")
    
    # Limit samples if specified
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # Process into standard format
    def process_ultrachat(example):
        """Process UltraChat format to text."""
        messages = example.get("messages", [])
        text_parts = []
        for msg in messages:
            role = msg.get("role", "").capitalize()
            content = msg.get("content", "")
            text_parts.append(f"{role}: {content}")
        return {"text": "\n\n".join(text_parts)}
    
    if "messages" in dataset.column_names:
        dataset = dataset.map(process_ultrachat, remove_columns=dataset.column_names)
    elif "text" not in dataset.column_names:
        # Try to construct text from other columns
        if "prompt" in dataset.column_names and "response" in dataset.column_names:
            dataset = dataset.map(
                lambda x: {"text": f"User: {x['prompt']}\n\nAssistant: {x['response']}"},
                remove_columns=dataset.column_names
            )
    
    # Save
    output_path = output_dir / "sft_train.json"
    dataset.to_json(str(output_path))
    logger.info(f"Saved {len(dataset)} SFT examples to {output_path}")
    
    return output_path


def prepare_preference_dataset(
    dataset_name: str,
    output_dir: Path,
    max_samples: int = None,
) -> Path:
    """
    Download and prepare a preference dataset.
    
    Args:
        dataset_name: HuggingFace dataset name
        output_dir: Directory to save processed data
        max_samples: Maximum number of samples
    
    Returns:
        Path to the saved dataset
    """
    logger.info(f"Preparing preference dataset: {dataset_name}")
    
    # Load dataset
    if "ultrafeedback" in dataset_name.lower():
        dataset = load_dataset(dataset_name, split="train_prefs")
    elif "hh-rlhf" in dataset_name.lower():
        dataset = load_dataset(dataset_name, split="train")
    else:
        dataset = load_dataset(dataset_name, split="train")
    
    # Limit samples
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # Process into standard format (prompt, chosen, rejected)
    def process_preference(example):
        """Convert to standard preference format."""
        result = {}
        
        # Handle different dataset formats
        if "prompt" in example and "chosen" in example and "rejected" in example:
            # Already in correct format
            result["prompt"] = example["prompt"]
            
            # Handle cases where chosen/rejected are message lists
            if isinstance(example["chosen"], list):
                # Extract the assistant response
                for msg in example["chosen"]:
                    if msg.get("role") == "assistant":
                        result["chosen"] = msg.get("content", "")
                        break
                else:
                    result["chosen"] = str(example["chosen"])
            else:
                result["chosen"] = example["chosen"]
            
            if isinstance(example["rejected"], list):
                for msg in example["rejected"]:
                    if msg.get("role") == "assistant":
                        result["rejected"] = msg.get("content", "")
                        break
                else:
                    result["rejected"] = str(example["rejected"])
            else:
                result["rejected"] = example["rejected"]
        
        elif "chosen" in example and "rejected" in example:
            # HH-RLHF format: full conversations
            result["prompt"] = ""  # Full context is in chosen/rejected
            result["chosen"] = example["chosen"]
            result["rejected"] = example["rejected"]
        
        else:
            # Unknown format, try to extract what we can
            result["prompt"] = example.get("prompt", example.get("question", ""))
            result["chosen"] = example.get("chosen", example.get("best", ""))
            result["rejected"] = example.get("rejected", example.get("worst", ""))
        
        return result
    
    # Get columns to remove
    cols_to_remove = [c for c in dataset.column_names if c not in ["prompt", "chosen", "rejected"]]
    dataset = dataset.map(process_preference, remove_columns=cols_to_remove)
    
    # Filter out empty examples
    dataset = dataset.filter(
        lambda x: len(x["chosen"]) > 0 and len(x["rejected"]) > 0
    )
    
    # Save
    output_path = output_dir / "preference_train.json"
    dataset.to_json(str(output_path))
    logger.info(f"Saved {len(dataset)} preference examples to {output_path}")
    
    return output_path


def create_eval_split(
    train_path: Path,
    eval_ratio: float = 0.1,
    max_eval_samples: int = 1000,
) -> Path:
    """
    Create an evaluation split from a training dataset.
    
    Args:
        train_path: Path to training data
        eval_ratio: Fraction of data to use for evaluation
        max_eval_samples: Maximum evaluation samples
    
    Returns:
        Path to evaluation dataset
    """
    # Load train data
    with open(train_path) as f:
        if train_path.suffix == ".jsonl":
            data = [json.loads(line) for line in f]
        else:
            data = json.load(f)
    
    # Calculate split
    num_eval = min(int(len(data) * eval_ratio), max_eval_samples)
    
    # Shuffle and split
    import random
    random.seed(42)
    random.shuffle(data)
    
    eval_data = data[:num_eval]
    train_data = data[num_eval:]
    
    # Save updated train and eval
    train_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(train_path, "w") as f:
        json.dump(train_data, f)
    
    eval_path = train_path.parent / train_path.name.replace("train", "eval")
    with open(eval_path, "w") as f:
        json.dump(eval_data, f)
    
    logger.info(f"Created eval split: {len(eval_data)} examples")
    logger.info(f"Updated train: {len(train_data)} examples")
    
    return eval_path


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for SLM-RL-Agent training")
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Directory to save processed datasets",
    )
    parser.add_argument(
        "--sft_dataset",
        type=str,
        default="HuggingFaceH4/ultrachat_200k",
        help="SFT dataset name from HuggingFace",
    )
    parser.add_argument(
        "--preference_dataset",
        type=str,
        default="HuggingFaceH4/ultrafeedback_binarized",
        help="Preference dataset name from HuggingFace",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=50000,
        help="Maximum samples per dataset (for faster iteration)",
    )
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.05,
        help="Fraction of data to use for evaluation",
    )
    parser.add_argument(
        "--skip_sft",
        action="store_true",
        help="Skip SFT dataset preparation",
    )
    parser.add_argument(
        "--skip_preference",
        action="store_true",
        help="Skip preference dataset preparation",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare SFT dataset
    if not args.skip_sft:
        sft_path = prepare_sft_dataset(
            args.sft_dataset,
            output_dir,
            args.max_samples,
        )
        create_eval_split(sft_path, args.eval_ratio)
    
    # Prepare preference dataset
    if not args.skip_preference:
        pref_path = prepare_preference_dataset(
            args.preference_dataset,
            output_dir,
            args.max_samples,
        )
        create_eval_split(pref_path, args.eval_ratio)
    
    logger.info("Data preparation complete!")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
