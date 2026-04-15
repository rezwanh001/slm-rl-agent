#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque
"""

"""
Dataset Loading Utilities for SLM-RL-Agent

This module provides functions to load various datasets for different training stages:
    - SFT datasets for supervised fine-tuning
    - Preference datasets for reward model training and DPO
    - Evaluation datasets for comprehensive testing

Supported datasets:
    - HuggingFace datasets (via datasets library)
    - Local JSON/JSONL files
    - Custom formats with conversion utilities
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from transformers import PreTrainedTokenizer

# Configure logging
logger = logging.getLogger(__name__)


def load_sft_dataset(
    dataset_name_or_path: str,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 1024,
    max_samples: Optional[int] = None,
    split: str = "train",
    prompt_column: str = "prompt",
    response_column: str = "response",
    chat_template: Optional[str] = None,
    streaming: bool = False,
) -> Dataset:
    """
    Load and preprocess a dataset for Supervised Fine-Tuning (SFT).
    
    This function handles multiple dataset formats:
    1. HuggingFace datasets with conversation format (messages column)
    2. Instruction datasets with prompt/response format
    3. Local JSON files with custom column names
    
    Args:
        dataset_name_or_path: HuggingFace dataset name or path to local file
        tokenizer: Tokenizer for the model being trained
        max_seq_length: Maximum sequence length for training
        max_samples: Maximum number of samples to load (for debugging)
        split: Dataset split to load ("train", "test", "validation")
        prompt_column: Name of the column containing prompts/instructions
        response_column: Name of the column containing responses
        chat_template: Chat template format ("chatml", "llama2", "zephyr", or None for auto)
        streaming: Whether to use streaming mode for large datasets
    
    Returns:
        Processed Dataset ready for SFT training
        
    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
        >>> dataset = load_sft_dataset(
        ...     "HuggingFaceH4/ultrachat_200k",
        ...     tokenizer,
        ...     max_seq_length=1024
        ... )
    """
    logger.info(f"Loading SFT dataset from: {dataset_name_or_path}")
    
    # Check if it's a local file or HuggingFace dataset
    if Path(dataset_name_or_path).exists():
        dataset = _load_local_dataset(dataset_name_or_path, split)
    else:
        # Load from HuggingFace Hub
        try:
            dataset = load_dataset(
                dataset_name_or_path,
                split=split,
                streaming=streaming,
            )
        except Exception as e:
            # Try loading with common configurations
            logger.info(f"Trying alternative loading method: {e}")
            dataset = load_dataset(
                dataset_name_or_path,
                split=f"{split}_sft" if "ultrachat" in dataset_name_or_path.lower() else split,
                streaming=streaming,
            )
    
    # Limit samples if specified (useful for debugging)
    if max_samples is not None and not streaming:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"Limited dataset to {len(dataset)} samples")
    
    # Detect and process dataset format
    column_names = dataset.column_names if hasattr(dataset, 'column_names') else list(next(iter(dataset)).keys())
    
    if "messages" in column_names:
        # Conversation format (e.g., UltraChat, OpenAssistant)
        logger.info("Detected conversation format dataset")
        dataset = _process_conversation_dataset(dataset, tokenizer, max_seq_length, chat_template)
    elif prompt_column in column_names and response_column in column_names:
        # Instruction format (e.g., Alpaca)
        logger.info("Detected instruction format dataset")
        dataset = _process_instruction_dataset(
            dataset, tokenizer, max_seq_length, prompt_column, response_column
        )
    else:
        raise ValueError(
            f"Unknown dataset format. Expected 'messages' column or '{prompt_column}'/'{response_column}' columns. "
            f"Found columns: {column_names}"
        )
    
    logger.info(f"Loaded {len(dataset) if hasattr(dataset, '__len__') else 'streaming'} samples for SFT")
    return dataset


def load_preference_dataset(
    dataset_name_or_path: str,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int = 512,
    max_samples: Optional[int] = None,
    split: str = "train",
    prompt_column: str = "prompt",
    chosen_column: str = "chosen",
    rejected_column: str = "rejected",
) -> Dataset:
    """
    Load and preprocess a preference dataset for reward model training or DPO.
    
    Preference datasets contain triplets of (prompt, chosen_response, rejected_response)
    where chosen_response is preferred over rejected_response by human annotators.
    
    Args:
        dataset_name_or_path: HuggingFace dataset name or path to local file
        tokenizer: Tokenizer for preprocessing
        max_seq_length: Maximum sequence length
        max_samples: Maximum number of samples to load
        split: Dataset split to load
        prompt_column: Name of prompt column
        chosen_column: Name of chosen response column  
        rejected_column: Name of rejected response column
    
    Returns:
        Processed Dataset with 'prompt', 'chosen', 'rejected' columns
        
    Example:
        >>> dataset = load_preference_dataset(
        ...     "HuggingFaceH4/ultrafeedback_binarized",
        ...     tokenizer,
        ...     max_seq_length=512
        ... )
    """
    logger.info(f"Loading preference dataset from: {dataset_name_or_path}")
    
    # Load the raw dataset
    if Path(dataset_name_or_path).exists():
        dataset = _load_local_dataset(dataset_name_or_path, split)
    else:
        # Handle special cases for common preference datasets
        if "ultrafeedback" in dataset_name_or_path.lower():
            dataset = load_dataset(dataset_name_or_path, split=f"{split}_prefs")
        elif "hh-rlhf" in dataset_name_or_path.lower():
            dataset = load_dataset(dataset_name_or_path, split=split)
        else:
            dataset = load_dataset(dataset_name_or_path, split=split)
    
    # Limit samples
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # Standardize column names and format
    dataset = _standardize_preference_dataset(
        dataset, tokenizer, max_seq_length,
        prompt_column, chosen_column, rejected_column
    )
    
    logger.info(f"Loaded {len(dataset)} preference pairs")
    return dataset


def load_evaluation_dataset(
    dataset_name_or_path: str,
    split: str = "test",
    max_samples: Optional[int] = None,
    task_type: str = "generation",
) -> Dataset:
    """
    Load a dataset for evaluation purposes.
    
    Args:
        dataset_name_or_path: Dataset identifier or local path
        split: Dataset split to load
        max_samples: Maximum samples for evaluation
        task_type: Type of evaluation ("generation", "classification", "qa")
    
    Returns:
        Dataset ready for evaluation
    """
    logger.info(f"Loading evaluation dataset: {dataset_name_or_path}")
    
    if Path(dataset_name_or_path).exists():
        dataset = _load_local_dataset(dataset_name_or_path, split)
    else:
        dataset = load_dataset(dataset_name_or_path, split=split)
    
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    return dataset


def _load_local_dataset(path: str, split: str = "train") -> Dataset:
    """Load a dataset from a local file (JSON, JSONL, or CSV)."""
    path = Path(path)
    
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Dataset.from_list(data)
    
    elif path.suffix == ".jsonl":
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return Dataset.from_list(data)
    
    elif path.suffix == ".csv":
        return Dataset.from_csv(str(path))
    
    elif path.is_dir():
        # Try to load as a dataset directory
        return load_dataset(str(path), split=split)
    
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def _process_conversation_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
    chat_template: Optional[str] = None,
) -> Dataset:
    """
    Process a conversation-format dataset into training format.
    
    Converts messages like:
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    into properly formatted training strings.
    """
    
    def format_conversation(example: Dict[str, Any]) -> Dict[str, str]:
        """Format a single conversation into a training string."""
        messages = example.get("messages", [])
        
        # Try to use tokenizer's chat template if available
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            try:
                text = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                return {"text": text}
            except Exception:
                pass
        
        # Fallback: Manual formatting based on chat_template
        if chat_template == "chatml":
            text = _format_chatml(messages)
        elif chat_template == "llama2":
            text = _format_llama2(messages)
        elif chat_template == "zephyr":
            text = _format_zephyr(messages)
        else:
            # Simple default format
            text = _format_simple(messages)
        
        return {"text": text}
    
    # Apply formatting
    dataset = dataset.map(
        format_conversation,
        remove_columns=dataset.column_names,
        desc="Formatting conversations",
    )
    
    return dataset


def _process_instruction_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
    prompt_column: str,
    response_column: str,
) -> Dataset:
    """Process an instruction-format dataset into training format."""
    
    def format_instruction(example: Dict[str, Any]) -> Dict[str, str]:
        """Format instruction-response pair into training string."""
        prompt = example.get(prompt_column, "")
        response = example.get(response_column, "")
        
        # Check for additional context/input column (Alpaca format)
        input_text = example.get("input", "")
        if input_text:
            prompt = f"{prompt}\n\nInput: {input_text}"
        
        # Format as a simple conversation
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        
        # Use tokenizer's chat template if available
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                return {"text": text}
            except Exception:
                pass
        
        # Fallback to simple format
        text = f"### Instruction:\n{prompt}\n\n### Response:\n{response}"
        return {"text": text}
    
    columns_to_remove = [c for c in dataset.column_names if c not in ["text"]]
    dataset = dataset.map(
        format_instruction,
        remove_columns=columns_to_remove,
        desc="Formatting instructions",
    )
    
    return dataset


def _standardize_preference_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
    prompt_column: str,
    chosen_column: str,
    rejected_column: str,
) -> Dataset:
    """Standardize preference dataset format for reward model training."""
    
    def process_preference(example: Dict[str, Any]) -> Dict[str, str]:
        """Process a single preference example."""
        # Extract prompt
        prompt = example.get(prompt_column, example.get("prompt", ""))
        
        # Handle different formats for chosen/rejected
        chosen = example.get(chosen_column, example.get("chosen", ""))
        rejected = example.get(rejected_column, example.get("rejected", ""))
        
        # If chosen/rejected are message lists, extract the content
        if isinstance(chosen, list):
            chosen = _extract_response_from_messages(chosen)
        if isinstance(rejected, list):
            rejected = _extract_response_from_messages(rejected)
        
        # If prompt is a message list, extract it
        if isinstance(prompt, list):
            prompt = _extract_prompt_from_messages(prompt)
        
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }
    
    # Process the dataset
    columns_to_keep = ["prompt", "chosen", "rejected"]
    columns_to_remove = [c for c in dataset.column_names if c not in columns_to_keep]
    
    dataset = dataset.map(
        process_preference,
        remove_columns=columns_to_remove,
        desc="Standardizing preferences",
    )
    
    return dataset


def _extract_response_from_messages(messages: List[Dict[str, str]]) -> str:
    """Extract the assistant's response from a message list."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return str(messages)


def _extract_prompt_from_messages(messages: List[Dict[str, str]]) -> str:
    """Extract the user prompt from a message list."""
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return str(messages)


def _format_chatml(messages: List[Dict[str, str]]) -> str:
    """Format messages using ChatML template."""
    text = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return text


def _format_llama2(messages: List[Dict[str, str]]) -> str:
    """Format messages using Llama 2 template."""
    text = "<s>[INST] "
    for i, msg in enumerate(messages):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "user":
            if i > 0:
                text += " [INST] "
            text += content + " [/INST] "
        else:
            text += content + " </s>"
    return text.strip()


def _format_zephyr(messages: List[Dict[str, str]]) -> str:
    """Format messages using Zephyr template."""
    text = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        text += f"<|{role}|>\n{content}</s>\n"
    return text


def _format_simple(messages: List[Dict[str, str]]) -> str:
    """Simple formatting without special tokens."""
    text = ""
    for msg in messages:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        text += f"{role}: {content}\n\n"
    return text.strip()


def create_train_eval_split(
    dataset: Dataset,
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """
    Split a dataset into training and evaluation sets.
    
    Args:
        dataset: Dataset to split
        eval_ratio: Fraction of data for evaluation
        seed: Random seed for reproducibility
    
    Returns:
        DatasetDict with 'train' and 'eval' splits
    """
    split = dataset.train_test_split(test_size=eval_ratio, seed=seed)
    return DatasetDict({
        "train": split["train"],
        "eval": split["test"],
    })
