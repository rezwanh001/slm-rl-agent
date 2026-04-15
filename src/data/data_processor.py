#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque
"""

"""
Data Processor for SLM-RL-Agent

This module provides utilities for preprocessing and tokenizing data for training.
It handles tokenization, padding, truncation, and batching for all training stages.

Key Features:
    - Efficient tokenization with caching
    - Dynamic padding for memory efficiency
    - Support for packing multiple sequences
    - Completion-only masking for instruction tuning
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForSFT:
    """
    Data collator for Supervised Fine-Tuning.
    
    This collator handles:
    - Dynamic padding to the longest sequence in a batch
    - Creation of attention masks
    - Label preparation with -100 masking for padding tokens
    
    For instruction tuning, it can optionally mask the prompt tokens
    so the model only learns to predict the response.
    
    Args:
        tokenizer: The tokenizer used for the model
        max_length: Maximum sequence length
        pad_to_multiple_of: Pad sequence length to multiple of this value
        completion_only: If True, only compute loss on completion tokens
        instruction_template: Template string that marks the end of instruction
        response_template: Template string that marks the start of response
    """
    
    tokenizer: PreTrainedTokenizer
    max_length: int = 1024
    pad_to_multiple_of: Optional[int] = 8
    completion_only: bool = False
    instruction_template: str = "### Instruction:"
    response_template: str = "### Response:"
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of features into padded tensors.
        
        Args:
            features: List of dictionaries containing 'input_ids', 'attention_mask', etc.
        
        Returns:
            Dictionary with batched and padded tensors
        """
        # Extract the text or input_ids from features
        if "input_ids" not in features[0]:
            # Need to tokenize first
            texts = [f.get("text", "") for f in features]
            batch = self.tokenizer(
                texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
        else:
            # Already tokenized, just need to pad
            batch = self._pad_features(features)
        
        # Create labels (same as input_ids, with padding masked)
        labels = batch["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # If completion_only, mask the instruction/prompt tokens
        if self.completion_only:
            labels = self._mask_prompt_tokens(batch["input_ids"], labels)
        
        batch["labels"] = labels
        
        return batch
    
    def _pad_features(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Pad a batch of tokenized features to the same length."""
        # Find the maximum length in this batch
        max_len = max(len(f["input_ids"]) for f in features)
        
        # Round up to multiple if specified
        if self.pad_to_multiple_of is not None:
            max_len = ((max_len + self.pad_to_multiple_of - 1) 
                      // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        # Cap at max_length
        max_len = min(max_len, self.max_length)
        
        # Pad all features
        input_ids = []
        attention_mask = []
        
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        
        for feature in features:
            ids = feature["input_ids"][:max_len]
            mask = feature.get("attention_mask", [1] * len(ids))[:max_len]
            
            # Calculate padding
            padding_length = max_len - len(ids)
            
            # Pad on the right (default for causal LM)
            input_ids.append(ids + [pad_token_id] * padding_length)
            attention_mask.append(mask + [0] * padding_length)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
    
    def _mask_prompt_tokens(
        self, 
        input_ids: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Mask prompt tokens in labels so loss is only computed on responses.
        
        This finds the response_template in each sequence and masks everything before it.
        """
        response_token_ids = self.tokenizer.encode(
            self.response_template, add_special_tokens=False
        )
        
        for idx in range(input_ids.size(0)):
            # Find the start of the response
            seq = input_ids[idx].tolist()
            response_start = self._find_subsequence(seq, response_token_ids)
            
            if response_start is not None:
                # Mask everything up to and including the response template
                labels[idx, :response_start + len(response_token_ids)] = -100
        
        return labels
    
    @staticmethod
    def _find_subsequence(sequence: List[int], subsequence: List[int]) -> Optional[int]:
        """Find the starting index of a subsequence in a sequence."""
        for i in range(len(sequence) - len(subsequence) + 1):
            if sequence[i:i + len(subsequence)] == subsequence:
                return i
        return None


@dataclass
class DataCollatorForPreference:
    """
    Data collator for preference/comparison data (reward model training and DPO).
    
    This collator handles pairs of (chosen, rejected) responses for the same prompt.
    It ensures both responses are processed together for efficient training.
    
    Args:
        tokenizer: The tokenizer used for the model
        max_length: Maximum sequence length
        max_prompt_length: Maximum prompt length (responses will use remaining tokens)
        padding: Padding strategy ('longest', 'max_length', False)
    """
    
    tokenizer: PreTrainedTokenizer
    max_length: int = 512
    max_prompt_length: int = 256
    padding: Union[bool, str] = True
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate preference pairs into batched tensors.
        
        Expects features with 'prompt', 'chosen', and 'rejected' keys.
        
        Returns:
            Dictionary with input_ids and attention_mask for both chosen and rejected,
            as well as labels for training.
        """
        prompts = [f["prompt"] for f in features]
        chosen_responses = [f["chosen"] for f in features]
        rejected_responses = [f["rejected"] for f in features]
        
        # Tokenize chosen (prompt + response)
        chosen_texts = [
            self._format_preference_text(p, c) 
            for p, c in zip(prompts, chosen_responses)
        ]
        chosen_batch = self.tokenizer(
            chosen_texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=True,
            return_tensors="pt",
        )
        
        # Tokenize rejected (prompt + response)
        rejected_texts = [
            self._format_preference_text(p, r) 
            for p, r in zip(prompts, rejected_responses)
        ]
        rejected_batch = self.tokenizer(
            rejected_texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=True,
            return_tensors="pt",
        )
        
        # Create labels (mask padding)
        chosen_labels = chosen_batch["input_ids"].clone()
        chosen_labels[chosen_labels == self.tokenizer.pad_token_id] = -100
        
        rejected_labels = rejected_batch["input_ids"].clone()
        rejected_labels[rejected_labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "chosen_input_ids": chosen_batch["input_ids"],
            "chosen_attention_mask": chosen_batch["attention_mask"],
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_batch["input_ids"],
            "rejected_attention_mask": rejected_batch["attention_mask"],
            "rejected_labels": rejected_labels,
        }
    
    def _format_preference_text(self, prompt: str, response: str) -> str:
        """Format prompt and response into a single text."""
        # Try to use tokenizer's chat template
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            except Exception:
                pass
        
        # Fallback to simple format
        return f"User: {prompt}\n\nAssistant: {response}"


class DataProcessor:
    """
    High-level data processor for preparing datasets for all training stages.
    
    This class provides convenient methods for:
    - Tokenizing datasets
    - Creating data collators
    - Filtering and cleaning data
    - Computing statistics
    
    Example:
        >>> processor = DataProcessor(tokenizer, max_length=1024)
        >>> processed_dataset = processor.process_sft_dataset(raw_dataset)
        >>> collator = processor.get_sft_collator()
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024,
        max_prompt_length: int = 256,
    ):
        """
        Initialize the data processor.
        
        Args:
            tokenizer: Tokenizer for the model
            max_length: Maximum sequence length
            max_prompt_length: Maximum prompt length for preference data
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {self.tokenizer.pad_token}")
    
    def process_sft_dataset(
        self,
        dataset: Dataset,
        text_column: str = "text",
        num_proc: int = 4,
        batched: bool = True,
    ) -> Dataset:
        """
        Tokenize a dataset for SFT training.
        
        Args:
            dataset: Raw dataset with text column
            text_column: Name of the column containing text
            num_proc: Number of processes for parallel tokenization
            batched: Whether to tokenize in batches
        
        Returns:
            Tokenized dataset ready for training
        """
        def tokenize_function(examples):
            return self.tokenizer(
                examples[text_column],
                truncation=True,
                max_length=self.max_length,
                padding=False,  # Will pad during collation
            )
        
        tokenized = dataset.map(
            tokenize_function,
            batched=batched,
            num_proc=num_proc,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset",
        )
        
        return tokenized
    
    def process_preference_dataset(
        self,
        dataset: Dataset,
        num_proc: int = 4,
    ) -> Dataset:
        """
        Process a preference dataset for reward model training or DPO.
        
        Args:
            dataset: Dataset with 'prompt', 'chosen', 'rejected' columns
            num_proc: Number of processes for parallel processing
        
        Returns:
            Processed dataset
        """
        def compute_lengths(example):
            # Compute sequence lengths for filtering
            prompt_len = len(self.tokenizer.encode(example["prompt"]))
            chosen_len = len(self.tokenizer.encode(example["chosen"]))
            rejected_len = len(self.tokenizer.encode(example["rejected"]))
            
            return {
                "prompt_length": prompt_len,
                "chosen_length": chosen_len,
                "rejected_length": rejected_len,
                "total_chosen_length": prompt_len + chosen_len,
                "total_rejected_length": prompt_len + rejected_len,
            }
        
        # Add length information
        dataset = dataset.map(compute_lengths, num_proc=num_proc)
        
        # Filter out examples that are too long
        original_size = len(dataset)
        dataset = dataset.filter(
            lambda x: (x["total_chosen_length"] <= self.max_length and 
                      x["total_rejected_length"] <= self.max_length),
            num_proc=num_proc,
        )
        filtered_size = len(dataset)
        
        if filtered_size < original_size:
            logger.info(
                f"Filtered {original_size - filtered_size} examples that exceeded max_length "
                f"({filtered_size}/{original_size} remaining)"
            )
        
        return dataset
    
    def get_sft_collator(
        self,
        completion_only: bool = False,
        pad_to_multiple_of: int = 8,
    ) -> DataCollatorForSFT:
        """Get a data collator for SFT training."""
        return DataCollatorForSFT(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            completion_only=completion_only,
            pad_to_multiple_of=pad_to_multiple_of,
        )
    
    def get_preference_collator(self) -> DataCollatorForPreference:
        """Get a data collator for preference data."""
        return DataCollatorForPreference(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            max_prompt_length=self.max_prompt_length,
        )
    
    def compute_dataset_statistics(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Compute statistics about a tokenized dataset.
        
        Returns information about sequence lengths, token distribution, etc.
        """
        lengths = []
        
        for example in dataset:
            if "input_ids" in example:
                lengths.append(len(example["input_ids"]))
            elif "text" in example:
                lengths.append(len(self.tokenizer.encode(example["text"])))
        
        if not lengths:
            return {"error": "No valid examples found"}
        
        import numpy as np
        lengths = np.array(lengths)
        
        return {
            "num_examples": len(lengths),
            "mean_length": float(np.mean(lengths)),
            "std_length": float(np.std(lengths)),
            "min_length": int(np.min(lengths)),
            "max_length": int(np.max(lengths)),
            "median_length": float(np.median(lengths)),
            "percentile_90": float(np.percentile(lengths, 90)),
            "percentile_95": float(np.percentile(lengths, 95)),
            "percentile_99": float(np.percentile(lengths, 99)),
        }
