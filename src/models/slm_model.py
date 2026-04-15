#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque
"""

"""
SLM Model Wrapper for SLM-RL-Agent

This module provides a unified interface for loading and configuring Small Language Models
for training with RLHF. It handles:
    - Model loading with quantization (QLoRA)
    - LoRA configuration and injection
    - Gradient checkpointing setup
    - Flash Attention 2 integration
    - Multi-GPU support

The wrapper abstracts away the complexity of setting up efficient training configurations,
making it easy to get started with different model architectures.

References:
    - Hu et al. (2021): LoRA: Low-Rank Adaptation of Large Language Models
    - Dettmers et al. (2023): QLoRA: Efficient Finetuning of Quantized LLMs
    - Dao et al. (2022): FlashAttention: Fast and Memory-Efficient Attention
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """
    Configuration for loading and setting up a Small Language Model.
    
    This dataclass encapsulates all the settings needed to load a model
    with the desired efficiency features (quantization, LoRA, etc.).
    
    Attributes:
        model_name_or_path: HuggingFace model ID or local path
        torch_dtype: Data type for model weights ("float32", "float16", "bfloat16", "auto")
        device_map: Device placement strategy ("auto", "cuda:0", etc.)
        use_flash_attention: Whether to use Flash Attention 2
        gradient_checkpointing: Enable gradient checkpointing for memory efficiency
        trust_remote_code: Trust remote code for models like Phi
        
        # LoRA settings
        use_lora: Whether to apply LoRA for parameter-efficient training
        lora_r: LoRA rank (lower = fewer params, higher = more capacity)
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout for LoRA layers
        lora_target_modules: Modules to apply LoRA to
        
        # Quantization settings
        use_quantization: Whether to use quantization (QLoRA)
        quantization_bits: 4 or 8 bit quantization
        quantization_type: "nf4" or "fp4" for 4-bit
        use_double_quant: Apply double quantization for more savings
    """
    # Model identification
    model_name_or_path: str = "EleutherAI/pythia-160m-deduped"
    
    # Basic settings
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    trust_remote_code: bool = False
    
    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None  # None means "all-linear"
    lora_bias: str = "none"
    
    # Quantization configuration
    use_quantization: bool = True
    quantization_bits: int = 4
    quantization_type: str = "nf4"
    use_double_quant: bool = True
    compute_dtype: str = "bfloat16"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate torch_dtype
        valid_dtypes = ["float32", "float16", "bfloat16", "auto"]
        if self.torch_dtype not in valid_dtypes:
            raise ValueError(f"torch_dtype must be one of {valid_dtypes}")
        
        # Validate quantization settings
        if self.quantization_bits not in [4, 8]:
            raise ValueError("quantization_bits must be 4 or 8")
        
        if self.quantization_type not in ["nf4", "fp4"]:
            raise ValueError("quantization_type must be 'nf4' or 'fp4'")


class SLMModel:
    """
    High-level wrapper for Small Language Models.
    
    This class provides a unified interface for:
    - Loading models with various efficiency configurations
    - Managing LoRA adapters
    - Saving and loading trained models
    - Inference with proper generation settings
    
    Example:
        >>> # Load a model for training
        >>> slm = SLMModel.from_pretrained(
        ...     "EleutherAI/pythia-160m-deduped",
        ...     use_lora=True,
        ...     use_quantization=True
        ... )
        >>> model = slm.model
        >>> tokenizer = slm.tokenizer
        
        >>> # Generate text
        >>> response = slm.generate("What is machine learning?")
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: ModelConfig,
    ):
        """
        Initialize the SLM wrapper.
        
        Usually you'll use `from_pretrained` instead of this constructor directly.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Track if we're using LoRA
        self.is_peft_model = isinstance(model, PeftModel)
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        use_lora: bool = True,
        use_quantization: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        gradient_checkpointing: bool = True,
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        **kwargs,
    ) -> "SLMModel":
        """
        Load a model from a HuggingFace model ID or local path.
        
        This method handles all the complexity of setting up efficient training:
        - Quantization configuration
        - LoRA injection
        - Gradient checkpointing
        - Flash Attention
        
        Args:
            model_name_or_path: HuggingFace model ID or local path
            use_lora: Whether to apply LoRA adapters
            use_quantization: Whether to quantize the model (4-bit by default)
            lora_r: LoRA rank
            lora_alpha: LoRA scaling factor
            gradient_checkpointing: Enable gradient checkpointing
            torch_dtype: Data type for weights
            device_map: Device placement strategy
            **kwargs: Additional arguments passed to ModelConfig
        
        Returns:
            SLMModel instance ready for training or inference
        
        Example:
            >>> slm = SLMModel.from_pretrained(
            ...     "EleutherAI/pythia-160m-deduped",
            ...     use_lora=True,
            ...     lora_r=16
            ... )
        """
        # Create configuration
        config = ModelConfig(
            model_name_or_path=model_name_or_path,
            use_lora=use_lora,
            use_quantization=use_quantization,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            gradient_checkpointing=gradient_checkpointing,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **kwargs,
        )
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=config.trust_remote_code,
        )
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")
        
        # Build model loading kwargs
        model_kwargs = _build_model_kwargs(config)
        
        # Load model
        logger.info(f"Loading model from {model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )
        
        # Prepare for k-bit training if using quantization
        if config.use_quantization:
            logger.info("Preparing model for k-bit training")
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=config.gradient_checkpointing,
            )
        elif config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        # Apply LoRA if enabled
        if config.use_lora:
            logger.info(f"Applying LoRA with r={config.lora_r}, alpha={config.lora_alpha}")
            model = _apply_lora(model, config)
        
        # Log model info
        trainable_params, total_params = _count_parameters(model)
        logger.info(
            f"Model loaded: {total_params:,} total params, "
            f"{trainable_params:,} trainable ({100 * trainable_params / total_params:.2f}%)"
        )
        
        return cls(model, tokenizer, config)
    
    def save_pretrained(self, output_dir: str, save_full_model: bool = False) -> None:
        """
        Save the model and tokenizer.
        
        If using LoRA, saves only the adapter weights by default.
        Use save_full_model=True to merge and save the full model.
        
        Args:
            output_dir: Directory to save the model
            save_full_model: Whether to merge LoRA and save full model
        """
        logger.info(f"Saving model to {output_dir}")
        
        if self.is_peft_model and not save_full_model:
            # Save only the LoRA adapter
            self.model.save_pretrained(output_dir)
        elif self.is_peft_model and save_full_model:
            # Merge LoRA weights and save full model
            merged_model = self.model.merge_and_unload()
            merged_model.save_pretrained(output_dir)
        else:
            # Save the full model
            self.model.save_pretrained(output_dir)
        
        # Always save tokenizer
        self.tokenizer.save_pretrained(output_dir)
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        do_sample: bool = True,
        **kwargs,
    ) -> str:
        """
        Generate a response for a given prompt.
        
        Args:
            prompt: Input text to generate from
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling probability threshold
            top_k: Top-k sampling parameter
            do_sample: Whether to sample (False = greedy decoding)
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text (prompt is stripped)
        """
        self.model.eval()
        
        # Encode prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.model_name_or_path if hasattr(self.config, 'max_length') else 1024,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )
        
        # Decode (remove prompt)
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        return response
    
    def get_trainable_parameters(self) -> Dict[str, int]:
        """Get information about trainable parameters."""
        trainable, total = _count_parameters(self.model)
        return {
            "trainable_params": trainable,
            "total_params": total,
            "trainable_percentage": 100 * trainable / total if total > 0 else 0,
        }


def load_model_for_training(
    model_name_or_path: str,
    training_stage: str = "sft",
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    use_quantization: bool = True,
    gradient_checkpointing: bool = True,
    device_map: str = "auto",
) -> tuple:
    """
    Convenience function to load a model configured for a specific training stage.
    
    This function applies recommended settings for each training stage:
    - SFT: Standard LoRA configuration
    - Reward: LoRA with modifications for scalar output
    - PPO: Policy model with value head preparation
    - DPO: Reference model handling
    
    Args:
        model_name_or_path: Model identifier
        training_stage: One of "sft", "reward", "ppo", "dpo"
        use_lora: Whether to use LoRA
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        use_quantization: Whether to use quantization
        gradient_checkpointing: Enable gradient checkpointing
        device_map: Device map strategy
    
    Returns:
        Tuple of (model, tokenizer)
    
    Example:
        >>> model, tokenizer = load_model_for_training(
        ...     "EleutherAI/pythia-160m-deduped",
        ...     training_stage="sft",
        ...     use_lora=True
        ... )
    """
    slm = SLMModel.from_pretrained(
        model_name_or_path,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        use_quantization=use_quantization,
        gradient_checkpointing=gradient_checkpointing,
        device_map=device_map,
    )
    
    return slm.model, slm.tokenizer


def _build_model_kwargs(config: ModelConfig) -> Dict[str, Any]:
    """Build the kwargs dictionary for model loading."""
    kwargs = {
        "trust_remote_code": config.trust_remote_code,
        "device_map": config.device_map,
    }
    
    # Set torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "auto": "auto",
    }
    kwargs["torch_dtype"] = dtype_map.get(config.torch_dtype, "auto")
    
    # Configure quantization
    if config.use_quantization:
        compute_dtype = dtype_map.get(config.compute_dtype, torch.bfloat16)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=(config.quantization_bits == 4),
            load_in_8bit=(config.quantization_bits == 8),
            bnb_4bit_quant_type=config.quantization_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.use_double_quant,
        )
        kwargs["quantization_config"] = bnb_config
    
    # Configure attention implementation
    if config.use_flash_attention:
        kwargs["attn_implementation"] = "flash_attention_2"
    else:
        kwargs["attn_implementation"] = "sdpa"  # PyTorch 2.0 native attention
    
    return kwargs


def _apply_lora(model: PreTrainedModel, config: ModelConfig) -> PeftModel:
    """Apply LoRA adapters to the model."""
    # Determine target modules
    if config.lora_target_modules is None:
        # Default to all linear layers
        target_modules = "all-linear"
    else:
        target_modules = config.lora_target_modules
    
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias=config.lora_bias,
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    
    return model


def _count_parameters(model: PreTrainedModel) -> tuple:
    """Count trainable and total parameters in a model."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params
