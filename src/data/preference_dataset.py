#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque
"""

"""
Preference Dataset Utilities for SLM-RL-Agents

This module provides tools for creating and managing preference datasets
used in reward model training and Direct Preference Optimization (DPO).

Preference data consists of triplets (prompt, chosen_response, rejected_response)
where the chosen response is preferred over the rejected response.

Key Features:
    - Create preference pairs from various sources
    - Synthetic preference generation using AI feedback (RLAIF)
    - Dataset validation and quality checks
    - Format conversion utilities
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

from datasets import Dataset, DatasetDict
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PreferenceDataset:
    """
    A class for managing and creating preference datasets.
    
    This class provides utilities to:
    - Load preference data from various formats
    - Create preference pairs from raw data
    - Validate and clean preference data
    - Export to formats compatible with TRL trainers
    
    Example:
        >>> # Load from existing data
        >>> pref_dataset = PreferenceDataset.from_json("preferences.json")
        
        >>> # Create from comparison data
        >>> pref_dataset = PreferenceDataset.from_comparisons(
        ...     prompts=["What is AI?"],
        ...     chosen=["AI is the simulation of human intelligence..."],
        ...     rejected=["I don't know."]
        ... )
        
        >>> # Get HuggingFace Dataset
        >>> hf_dataset = pref_dataset.to_dataset()
    """
    
    def __init__(
        self,
        data: List[Dict[str, str]],
        validate: bool = True,
    ):
        """
        Initialize with a list of preference triplets.
        
        Args:
            data: List of dicts with 'prompt', 'chosen', 'rejected' keys
            validate: Whether to validate the data on initialization
        """
        self.data = data
        
        if validate:
            self._validate()
    
    def _validate(self) -> None:
        """Validate the preference data format and content."""
        required_keys = {"prompt", "chosen", "rejected"}
        
        for idx, item in enumerate(self.data):
            # Check required keys
            missing_keys = required_keys - set(item.keys())
            if missing_keys:
                raise ValueError(
                    f"Item {idx} missing required keys: {missing_keys}"
                )
            
            # Check for empty values
            for key in required_keys:
                if not item[key] or not item[key].strip():
                    logger.warning(f"Item {idx} has empty {key}")
    
    @classmethod
    def from_json(cls, path: str) -> "PreferenceDataset":
        """Load preference data from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)
    
    @classmethod
    def from_jsonl(cls, path: str) -> "PreferenceDataset":
        """Load preference data from a JSONL file."""
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return cls(data)
    
    @classmethod
    def from_comparisons(
        cls,
        prompts: List[str],
        chosen: List[str],
        rejected: List[str],
        save_path: Optional[str] = None,
    ) -> "PreferenceDataset":
        """
        Create a preference dataset from parallel lists of prompts and responses.
        
        This is the simplest way to create a preference dataset when you have
        explicit pairs of good and bad responses.
        
        Args:
            prompts: List of prompts/questions
            chosen: List of preferred responses
            rejected: List of rejected/worse responses
            save_path: Optional path to save the dataset
        
        Returns:
            PreferenceDataset instance
        
        Example:
            >>> dataset = PreferenceDataset.from_comparisons(
            ...     prompts=["What is Python?"],
            ...     chosen=["Python is a high-level programming language..."],
            ...     rejected=["Python is a snake."]
            ... )
        """
        if not (len(prompts) == len(chosen) == len(rejected)):
            raise ValueError(
                f"All lists must have the same length. "
                f"Got prompts={len(prompts)}, chosen={len(chosen)}, rejected={len(rejected)}"
            )
        
        data = [
            {"prompt": p, "chosen": c, "rejected": r}
            for p, c, r in zip(prompts, chosen, rejected)
        ]
        
        dataset = cls(data)
        
        if save_path:
            dataset.save(save_path)
        
        return dataset
    
    @classmethod
    def from_ranked_responses(
        cls,
        prompts: List[str],
        responses: List[List[str]],
        rankings: List[List[int]],
    ) -> "PreferenceDataset":
        """
        Create preference pairs from ranked responses.
        
        This is useful when you have multiple responses per prompt with rankings,
        allowing you to generate multiple preference pairs per prompt.
        
        Args:
            prompts: List of prompts
            responses: List of response lists (multiple responses per prompt)
            rankings: List of ranking lists (1 = best, higher = worse)
        
        Returns:
            PreferenceDataset with all valid preference pairs
        
        Example:
            >>> dataset = PreferenceDataset.from_ranked_responses(
            ...     prompts=["What is 2+2?"],
            ...     responses=[["4", "Four", "22", "I don't know"]],
            ...     rankings=[[1, 1, 3, 4]]  # First two are equally good
            ... )
        """
        data = []
        
        for prompt, resps, ranks in zip(prompts, responses, rankings):
            # Create pairs from rankings (chosen has lower rank number)
            for i, (resp_i, rank_i) in enumerate(zip(resps, ranks)):
                for j, (resp_j, rank_j) in enumerate(zip(resps, ranks)):
                    if rank_i < rank_j:  # resp_i is better
                        data.append({
                            "prompt": prompt,
                            "chosen": resp_i,
                            "rejected": resp_j,
                        })
        
        logger.info(f"Created {len(data)} preference pairs from ranked responses")
        return cls(data)
    
    def to_dataset(self) -> Dataset:
        """Convert to a HuggingFace Dataset."""
        return Dataset.from_list(self.data)
    
    def to_dict_list(self) -> List[Dict[str, str]]:
        """Get the raw data as a list of dictionaries."""
        return self.data.copy()
    
    def save(self, path: str) -> None:
        """
        Save the dataset to a file.
        
        Format is determined by file extension (.json or .jsonl)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix == ".jsonl":
            with open(path, "w", encoding="utf-8") as f:
                for item in self.data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.data)} preference pairs to {path}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.data[idx]
    
    def filter(
        self,
        condition: Callable[[Dict[str, str]], bool],
    ) -> "PreferenceDataset":
        """
        Filter the dataset based on a condition.
        
        Args:
            condition: Function that takes a data item and returns True to keep it
        
        Returns:
            New PreferenceDataset with filtered data
        """
        filtered_data = [item for item in self.data if condition(item)]
        return PreferenceDataset(filtered_data, validate=False)
    
    def sample(self, n: int, seed: Optional[int] = None) -> "PreferenceDataset":
        """
        Sample n items from the dataset.
        
        Args:
            n: Number of items to sample
            seed: Random seed for reproducibility
        
        Returns:
            New PreferenceDataset with sampled data
        """
        if seed is not None:
            random.seed(seed)
        
        sampled = random.sample(self.data, min(n, len(self.data)))
        return PreferenceDataset(sampled, validate=False)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Compute statistics about the dataset."""
        prompt_lengths = [len(item["prompt"]) for item in self.data]
        chosen_lengths = [len(item["chosen"]) for item in self.data]
        rejected_lengths = [len(item["rejected"]) for item in self.data]
        
        def _stats(lengths):
            import numpy as np
            arr = np.array(lengths)
            return {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": int(np.min(arr)),
                "max": int(np.max(arr)),
                "median": float(np.median(arr)),
            }
        
        return {
            "num_examples": len(self.data),
            "prompt_lengths": _stats(prompt_lengths),
            "chosen_lengths": _stats(chosen_lengths),
            "rejected_lengths": _stats(rejected_lengths),
        }


def create_preference_pairs(
    prompts: List[str],
    responses_per_prompt: List[List[str]],
    reward_model: Optional[Any] = None,
    comparison_fn: Optional[Callable[[str, str, str], Tuple[str, str]]] = None,
) -> PreferenceDataset:
    """
    Create preference pairs from multiple responses per prompt.
    
    This function can create preference pairs using either:
    1. A reward model to score responses
    2. A custom comparison function
    3. Random pairing (for baseline/testing)
    
    Args:
        prompts: List of prompts
        responses_per_prompt: List of response lists (one list per prompt)
        reward_model: Optional reward model to score responses
        comparison_fn: Optional function to compare two responses
    
    Returns:
        PreferenceDataset with generated preference pairs
    
    Example:
        >>> # Using a reward model
        >>> from src.models import RewardModel
        >>> rm = RewardModel.from_pretrained("path/to/reward_model")
        >>> dataset = create_preference_pairs(prompts, responses, reward_model=rm)
        
        >>> # Using a custom comparison function
        >>> def compare(prompt, r1, r2):
        ...     return (r1, r2) if len(r1) > len(r2) else (r2, r1)
        >>> dataset = create_preference_pairs(prompts, responses, comparison_fn=compare)
    """
    data = []
    
    for prompt, responses in tqdm(
        zip(prompts, responses_per_prompt),
        total=len(prompts),
        desc="Creating preference pairs",
    ):
        if len(responses) < 2:
            logger.warning(f"Skipping prompt with {len(responses)} responses")
            continue
        
        if reward_model is not None:
            # Score all responses and create pairs
            scores = []
            for response in responses:
                score = reward_model.score(prompt, response)
                scores.append((response, score))
            
            # Sort by score (descending)
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # Create pairs from adjacent rankings
            for i in range(len(scores) - 1):
                chosen, chosen_score = scores[i]
                rejected, rejected_score = scores[i + 1]
                
                # Only create pair if there's a meaningful difference
                if chosen_score > rejected_score:
                    data.append({
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                    })
        
        elif comparison_fn is not None:
            # Use custom comparison function
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    chosen, rejected = comparison_fn(prompt, responses[i], responses[j])
                    data.append({
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                    })
        
        else:
            # Random pairing (first response is "chosen", rest are "rejected")
            # This is only useful for testing the pipeline
            chosen = responses[0]
            for rejected in responses[1:]:
                data.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                })
    
    logger.info(f"Created {len(data)} preference pairs")
    return PreferenceDataset(data)


def generate_synthetic_preferences(
    prompts: List[str],
    model: Any,
    tokenizer: Any,
    num_responses: int = 4,
    temperature_range: Tuple[float, float] = (0.5, 1.2),
    seed: Optional[int] = None,
) -> PreferenceDataset:
    """
    Generate synthetic preference data by sampling multiple responses at different temperatures.
    
    This implements a simple version of RLAIF (RL from AI Feedback) where preferences
    are derived from generation diversity and quality metrics.
    
    The idea is that lower-temperature responses tend to be more coherent and
    focused, while higher-temperature responses may be more creative but also
    more prone to errors. This creates natural preference pairs.
    
    Args:
        prompts: List of prompts to generate responses for
        model: Language model for generation
        tokenizer: Tokenizer for the model
        num_responses: Number of responses to generate per prompt
        temperature_range: (min_temp, max_temp) for sampling
        seed: Random seed for reproducibility
    
    Returns:
        PreferenceDataset with synthetic preferences
    
    Note:
        This is a simplified approach. For higher-quality synthetic preferences,
        consider using Constitutional AI or a separate judge model.
    """
    import torch
    
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
    
    all_responses = []
    temperatures = []
    
    # Generate range of temperatures
    temp_step = (temperature_range[1] - temperature_range[0]) / (num_responses - 1)
    temps = [temperature_range[0] + i * temp_step for i in range(num_responses)]
    
    logger.info(f"Generating {num_responses} responses per prompt at temperatures {temps}")
    
    for prompt in tqdm(prompts, desc="Generating responses"):
        prompt_responses = []
        prompt_temps = []
        
        for temp in temps:
            # Encode prompt
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=temp,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            # Decode response (remove prompt)
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            
            prompt_responses.append(response)
            prompt_temps.append(temp)
        
        all_responses.append(prompt_responses)
        temperatures.append(prompt_temps)
    
    # Create preference pairs (lower temperature = preferred)
    data = []
    for prompt, responses, temps in zip(prompts, all_responses, temperatures):
        # Pair each lower-temp response with higher-temp responses
        for i, (resp_i, temp_i) in enumerate(zip(responses, temps)):
            for j, (resp_j, temp_j) in enumerate(zip(responses, temps)):
                if temp_i < temp_j:  # Lower temperature is preferred
                    data.append({
                        "prompt": prompt,
                        "chosen": resp_i,
                        "rejected": resp_j,
                    })
    
    logger.info(f"Generated {len(data)} synthetic preference pairs")
    return PreferenceDataset(data)


def merge_preference_datasets(*datasets: PreferenceDataset) -> PreferenceDataset:
    """
    Merge multiple preference datasets into one.
    
    This is useful for combining datasets from different sources
    (e.g., human annotations + synthetic data).
    
    Args:
        *datasets: PreferenceDataset instances to merge
    
    Returns:
        Merged PreferenceDataset
    """
    merged_data = []
    for dataset in datasets:
        merged_data.extend(dataset.to_dict_list())
    
    logger.info(f"Merged {len(datasets)} datasets into {len(merged_data)} total pairs")
    return PreferenceDataset(merged_data, validate=False)


def split_preference_dataset(
    dataset: PreferenceDataset,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Tuple[PreferenceDataset, PreferenceDataset]:
    """
    Split a preference dataset into train and evaluation sets.
    
    Args:
        dataset: PreferenceDataset to split
        train_ratio: Fraction of data for training
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    random.seed(seed)
    data = dataset.to_dict_list()
    random.shuffle(data)
    
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    eval_data = data[split_idx:]
    
    return (
        PreferenceDataset(train_data, validate=False),
        PreferenceDataset(eval_data, validate=False),
    )
