"""
Reward Model for SLM-RL-Agent

This module implements a Reward Model for RLHF (Reinforcement Learning from Human Feedback).
The reward model learns to predict human preferences from comparison data, providing
a scalar reward signal that can be used to fine-tune the policy model.

The reward model architecture is based on the policy model with:
1. The final language model head removed
2. A new linear projection to a single scalar output

Training uses the Bradley-Terry preference model:
    P(y_w > y_l | x) = σ(r(x, y_w) - r(x, y_l))

where r(x, y) is the reward for response y given prompt x.

References:
    - Christiano et al. (2017): Deep reinforcement learning from human preferences
    - Ouyang et al. (2022): Training language models to follow instructions with human feedback
    - Stiennon et al. (2020): Learning to summarize from human feedback
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import PeftModel, get_peft_model, LoraConfig, TaskType

logger = logging.getLogger(__name__)


@dataclass
class RewardModelOutput:
    """
    Output of the reward model forward pass.
    
    Attributes:
        rewards: Scalar rewards for each input sequence [batch_size]
        loss: Optional loss value (when training with chosen/rejected pairs)
        chosen_rewards: Rewards for chosen responses (when training)
        rejected_rewards: Rewards for rejected responses (when training)
        accuracy: Accuracy of preference prediction (when training)
    """
    rewards: torch.Tensor
    loss: Optional[torch.Tensor] = None
    chosen_rewards: Optional[torch.Tensor] = None
    rejected_rewards: Optional[torch.Tensor] = None
    accuracy: Optional[torch.Tensor] = None


class RewardModelHead(nn.Module):
    """
    The reward head that projects hidden states to a scalar reward.
    
    This can be a simple linear layer or a small MLP depending on configuration.
    The head is applied to the last non-padding token's hidden state.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 1,
        head_hidden_size: int = 256,
        dropout: float = 0.1,
    ):
        """
        Initialize the reward head.
        
        Args:
            hidden_size: Size of the input hidden states from the base model
            num_layers: Number of layers in the head (1 = linear, 2+ = MLP)
            head_hidden_size: Hidden size for MLP layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_layers = num_layers
        
        if num_layers == 1:
            # Simple linear projection
            self.head = nn.Linear(hidden_size, 1)
        else:
            # MLP head
            layers = []
            in_size = hidden_size
            
            for i in range(num_layers - 1):
                layers.extend([
                    nn.Linear(in_size, head_hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
                in_size = head_hidden_size
            
            layers.append(nn.Linear(in_size, 1))
            self.head = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights to produce near-zero outputs initially."""
        if self.num_layers == 1:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        else:
            # Initialize the final layer to produce near-zero outputs
            final_layer = self.head[-1]
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute rewards from hidden states.
        
        Args:
            hidden_states: Hidden states from the last layer [batch_size, seq_len, hidden_size]
                          or [batch_size, hidden_size] if already pooled
        
        Returns:
            Scalar rewards [batch_size]
        """
        # If we have sequence dimension, the hidden state should already be
        # the last token's representation (handled in RewardModel.forward)
        if hidden_states.dim() == 3:
            hidden_states = hidden_states[:, -1, :]
        
        rewards = self.head(hidden_states).squeeze(-1)
        return rewards


class RewardModel(nn.Module):
    """
    Reward Model for RLHF training.
    
    The reward model learns to predict human preferences by training on
    comparison data. Given a prompt and two responses (chosen and rejected),
    the model learns to assign higher reward to the preferred response.
    
    Architecture:
        - Base model (e.g., Pythia) without the LM head
        - Custom reward head that outputs a scalar
    
    Example:
        >>> # Initialize from an SFT checkpoint
        >>> rm = RewardModel.from_pretrained("./outputs/sft/final")
        
        >>> # Score a single response
        >>> reward = rm.score("What is AI?", "AI is the simulation of human intelligence...")
        
        >>> # Train on preference data
        >>> output = rm(
        ...     chosen_input_ids=chosen_ids,
        ...     chosen_attention_mask=chosen_mask,
        ...     rejected_input_ids=rejected_ids,
        ...     rejected_attention_mask=rejected_mask,
        ... )
        >>> loss = output.loss
    """
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        reward_head: RewardModelHead,
        normalize_rewards: bool = True,
        reward_scale: float = 1.0,
    ):
        """
        Initialize the reward model.
        
        Args:
            base_model: Pretrained transformer model (without LM head)
            tokenizer: Tokenizer for the model
            reward_head: The head that produces scalar rewards
            normalize_rewards: Whether to normalize rewards during inference
            reward_scale: Scaling factor for rewards
        """
        super().__init__()
        
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.reward_head = reward_head
        self.normalize_rewards = normalize_rewards
        self.reward_scale = reward_scale
        
        # For tracking reward statistics (for normalization)
        self.register_buffer("reward_mean", torch.tensor(0.0))
        self.register_buffer("reward_std", torch.tensor(1.0))
        self.register_buffer("reward_count", torch.tensor(0.0))
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        num_head_layers: int = 1,
        head_hidden_size: int = 256,
        head_dropout: float = 0.1,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        use_quantization: bool = True,
        device_map: str = "auto",
        **kwargs,
    ) -> "RewardModel":
        """
        Load a reward model from a pretrained checkpoint.
        
        The model can be initialized from:
        - An SFT model checkpoint (recommended)
        - A base pretrained model
        - An existing reward model checkpoint
        
        Args:
            model_name_or_path: Model identifier or path
            num_head_layers: Number of layers in the reward head
            head_hidden_size: Hidden size for the reward head MLP
            head_dropout: Dropout probability for the head
            use_lora: Whether to use LoRA for efficient training
            lora_r: LoRA rank
            lora_alpha: LoRA scaling factor
            use_quantization: Whether to quantize the base model
            device_map: Device placement strategy
        
        Returns:
            RewardModel instance
        """
        logger.info(f"Loading reward model from {model_name_or_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Build model loading configuration
        model_kwargs = {"device_map": device_map, "torch_dtype": torch.bfloat16}
        
        if use_quantization:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        
        # Load the base model
        # We use AutoModel (not AutoModelForCausalLM) to get hidden states without LM head
        config = AutoConfig.from_pretrained(model_name_or_path)
        
        # Try to load as a base model first
        try:
            base_model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
        except Exception:
            # If that fails, load as causal LM and extract the base
            causal_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
            # Get the base model (varies by architecture)
            if hasattr(causal_model, "model"):
                base_model = causal_model.model
            elif hasattr(causal_model, "transformer"):
                base_model = causal_model.transformer
            elif hasattr(causal_model, "gpt_neox"):
                base_model = causal_model.gpt_neox
            else:
                raise ValueError(f"Cannot extract base model from {type(causal_model)}")
        
        # Apply LoRA if enabled
        if use_lora:
            from peft import prepare_model_for_kbit_training
            
            if use_quantization:
                base_model = prepare_model_for_kbit_training(base_model)
            
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                target_modules="all-linear",
                task_type=TaskType.FEATURE_EXTRACTION,  # Not CAUSAL_LM since no LM head
            )
            base_model = get_peft_model(base_model, lora_config)
        
        # Create reward head
        hidden_size = config.hidden_size
        reward_head = RewardModelHead(
            hidden_size=hidden_size,
            num_layers=num_head_layers,
            head_hidden_size=head_hidden_size,
            dropout=head_dropout,
        )
        
        return cls(
            base_model=base_model,
            tokenizer=tokenizer,
            reward_head=reward_head,
            **kwargs,
        )
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        chosen_input_ids: Optional[torch.Tensor] = None,
        chosen_attention_mask: Optional[torch.Tensor] = None,
        rejected_input_ids: Optional[torch.Tensor] = None,
        rejected_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[RewardModelOutput, Tuple[torch.Tensor, ...]]:
        """
        Forward pass of the reward model.
        
        Can be used in two modes:
        1. Single-sequence mode: Just compute rewards for input_ids
        2. Preference mode: Compute rewards for chosen/rejected and return loss
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            chosen_input_ids: Input IDs for chosen responses
            chosen_attention_mask: Attention mask for chosen responses
            rejected_input_ids: Input IDs for rejected responses
            rejected_attention_mask: Attention mask for rejected responses
            return_dict: Whether to return a RewardModelOutput
        
        Returns:
            RewardModelOutput with rewards and optionally loss
        """
        # Preference training mode
        if chosen_input_ids is not None and rejected_input_ids is not None:
            return self._forward_preference(
                chosen_input_ids,
                chosen_attention_mask,
                rejected_input_ids,
                rejected_attention_mask,
                return_dict,
            )
        
        # Single-sequence mode
        if input_ids is None:
            raise ValueError("Must provide input_ids or chosen/rejected pairs")
        
        rewards = self._compute_rewards(input_ids, attention_mask)
        
        if return_dict:
            return RewardModelOutput(rewards=rewards)
        return (rewards,)
    
    def _forward_preference(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[RewardModelOutput, Tuple]:
        """
        Forward pass for preference training.
        
        Computes the Bradley-Terry loss:
            L = -log(σ(r_chosen - r_rejected))
        """
        # Compute rewards for both
        chosen_rewards = self._compute_rewards(chosen_input_ids, chosen_attention_mask)
        rejected_rewards = self._compute_rewards(rejected_input_ids, rejected_attention_mask)
        
        # Compute Bradley-Terry loss
        # loss = -log(sigmoid(r_chosen - r_rejected))
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
        
        # Compute accuracy (for monitoring)
        with torch.no_grad():
            accuracy = (chosen_rewards > rejected_rewards).float().mean()
        
        if return_dict:
            return RewardModelOutput(
                rewards=chosen_rewards,  # Default to chosen rewards
                loss=loss,
                chosen_rewards=chosen_rewards,
                rejected_rewards=rejected_rewards,
                accuracy=accuracy,
            )
        
        return (loss, chosen_rewards, rejected_rewards, accuracy)
    
    def _compute_rewards(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute scalar rewards for input sequences.
        
        Uses the last non-padding token's hidden state as the sequence representation.
        """
        # Get hidden states from base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        # Get the last layer's hidden states
        if hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0]
        
        # Get the hidden state for the last non-padding token
        if attention_mask is not None:
            # Find the last non-padding position for each sequence
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]
            
            # Gather the last token's hidden state for each sequence
            last_hidden = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                sequence_lengths,
            ]
        else:
            # If no attention mask, use the last position
            last_hidden = hidden_states[:, -1, :]
        
        # Compute rewards
        rewards = self.reward_head(last_hidden)
        
        # Apply scaling
        rewards = rewards * self.reward_scale
        
        return rewards
    
    def score(
        self,
        prompt: str,
        response: str,
    ) -> float:
        """
        Score a single prompt-response pair.
        
        This is a convenience method for evaluation and inference.
        
        Args:
            prompt: The input prompt
            response: The generated response
        
        Returns:
            Scalar reward value
        """
        self.eval()
        
        # Format and tokenize
        text = f"{prompt}\n\n{response}"
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.base_model.device) for k, v in inputs.items()}
        
        # Compute reward
        with torch.no_grad():
            output = self.forward(**inputs)
            reward = output.rewards.item()
        
        return reward
    
    def score_batch(
        self,
        prompts: List[str],
        responses: List[str],
        batch_size: int = 16,
    ) -> List[float]:
        """
        Score a batch of prompt-response pairs.
        
        Args:
            prompts: List of prompts
            responses: List of responses
            batch_size: Batch size for processing
        
        Returns:
            List of scalar rewards
        """
        self.eval()
        all_rewards = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_responses = responses[i:i + batch_size]
            
            texts = [f"{p}\n\n{r}" for p, r in zip(batch_prompts, batch_responses)]
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            inputs = {k: v.to(self.base_model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                output = self.forward(**inputs)
                all_rewards.extend(output.rewards.cpu().tolist())
        
        return all_rewards
    
    def save_pretrained(self, output_dir: str) -> None:
        """
        Save the reward model.
        
        Saves:
        - Base model (or LoRA adapter if using LoRA)
        - Reward head weights
        - Tokenizer
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save base model
        if isinstance(self.base_model, PeftModel):
            self.base_model.save_pretrained(output_dir)
        else:
            self.base_model.save_pretrained(output_dir)
        
        # Save reward head
        head_path = os.path.join(output_dir, "reward_head.pt")
        torch.save(self.reward_head.state_dict(), head_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Saved reward model to {output_dir}")
