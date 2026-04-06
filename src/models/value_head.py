"""
Value Head for PPO Training in SLM-RL-Agent

This module implements the Value Head component used in Proximal Policy Optimization (PPO).
The value head predicts the expected cumulative reward (value) for a given state,
which is used to compute advantages during policy optimization.

In the context of language model RLHF:
    - State = Current token sequence
    - Value V(s) = Expected cumulative reward for the rest of the generation
    - Advantage A(s,a) = Q(s,a) - V(s) ≈ R(s,a) + γV(s') - V(s)

The value head shares the base model with the policy (actor) and adds a separate
head to predict scalar values, enabling the actor-critic architecture.

References:
    - Schulman et al. (2017): Proximal Policy Optimization Algorithms
    - Ziegler et al. (2019): Fine-Tuning Language Models from Human Preferences
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = logging.getLogger(__name__)


class ValueHead(nn.Module):
    """
    A neural network head that predicts scalar values from hidden states.
    
    The value head is attached to a language model to enable actor-critic
    training. It takes the hidden states from the base model and predicts
    expected cumulative rewards for each position in the sequence.
    
    Architecture options:
        1. Linear: Single linear projection (efficient, may underfit)
        2. MLP: Multi-layer perceptron (more capacity, better value estimates)
    
    Attributes:
        head: The neural network layers
        summary_type: How to summarize sequence into a single value
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 2,
        head_hidden_size: int = 256,
        dropout: float = 0.1,
        summary_type: str = "last_token",
        zero_init: bool = True,
    ):
        """
        Initialize the value head.
        
        Args:
            hidden_size: Size of input hidden states from the base model
            num_layers: Number of layers (1 = linear, 2+ = MLP)
            head_hidden_size: Hidden size for intermediate MLP layers
            dropout: Dropout probability
            summary_type: How to get a single value from sequence
                - "last_token": Use the last token's representation
                - "mean": Average all token representations
                - "first_token": Use the first token (like BERT [CLS])
            zero_init: Whether to initialize output layer to produce zeros
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.summary_type = summary_type
        
        # Build the head architecture
        if num_layers == 1:
            self.head = nn.Linear(hidden_size, 1)
        else:
            layers = []
            in_features = hidden_size
            
            for i in range(num_layers - 1):
                layers.extend([
                    nn.Linear(in_features, head_hidden_size),
                    nn.Tanh(),  # Tanh is common for value functions
                    nn.Dropout(dropout),
                ])
                in_features = head_hidden_size
            
            layers.append(nn.Linear(in_features, 1))
            self.head = nn.Sequential(*layers)
        
        # Initialize weights
        if zero_init:
            self._zero_init_output()
    
    def _zero_init_output(self):
        """Initialize the output layer to produce near-zero values initially."""
        if isinstance(self.head, nn.Linear):
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        else:
            # Initialize the final layer in the Sequential
            final_layer = self.head[-1]
            if isinstance(final_layer, nn.Linear):
                nn.init.zeros_(final_layer.weight)
                nn.init.zeros_(final_layer.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute values from hidden states.
        
        Args:
            hidden_states: Hidden states from the model [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len] (needed for "last_token")
        
        Returns:
            Value predictions [batch_size] if summary_type is set,
            or [batch_size, seq_len] for per-token values
        """
        # Apply the value head to all positions
        # hidden_states: [batch_size, seq_len, hidden_size]
        # values: [batch_size, seq_len, 1] -> [batch_size, seq_len]
        values = self.head(hidden_states).squeeze(-1)
        
        # Summarize to a single value if needed
        if self.summary_type == "last_token":
            if attention_mask is not None:
                # Get the position of the last non-padding token
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = values.shape[0]
                values = values[
                    torch.arange(batch_size, device=values.device),
                    sequence_lengths,
                ]
            else:
                values = values[:, -1]
        
        elif self.summary_type == "mean":
            if attention_mask is not None:
                # Masked mean
                mask = attention_mask.float()
                values = (values * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                values = values.mean(dim=1)
        
        elif self.summary_type == "first_token":
            values = values[:, 0]
        
        # Otherwise, return per-token values (for PPO with per-token rewards)
        
        return values


class AutoModelForCausalLMWithValueHead(nn.Module):
    """
    A language model with both a language modeling head and a value head.
    
    This class wraps a causal language model and adds a value head for PPO training.
    The base model parameters are shared between the actor (LM head) and critic (value head),
    with separate heads for action probabilities and value estimation.
    
    This architecture is more memory-efficient than having separate actor and critic models,
    as the majority of parameters are shared.
    
    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> base_model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLMWithValueHead(base_model)
        >>> outputs = model(input_ids)
        >>> logits = outputs.logits  # For policy
        >>> values = outputs.value  # For value function
    """
    
    def __init__(
        self,
        pretrained_model: PreTrainedModel,
        value_head_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the model with value head.
        
        Args:
            pretrained_model: A pretrained causal language model
            value_head_config: Configuration for the value head
                - num_layers: Number of layers (default: 2)
                - head_hidden_size: Hidden size (default: 256)
                - dropout: Dropout rate (default: 0.1)
        """
        super().__init__()
        
        self.pretrained_model = pretrained_model
        self.config = pretrained_model.config
        
        # Get hidden size from config
        if hasattr(self.config, "hidden_size"):
            hidden_size = self.config.hidden_size
        elif hasattr(self.config, "n_embd"):
            hidden_size = self.config.n_embd
        else:
            raise ValueError("Cannot determine hidden size from model config")
        
        # Create value head
        value_head_config = value_head_config or {}
        self.value_head = ValueHead(
            hidden_size=hidden_size,
            num_layers=value_head_config.get("num_layers", 2),
            head_hidden_size=value_head_config.get("head_hidden_size", 256),
            dropout=value_head_config.get("dropout", 0.1),
            summary_type="last_token",
            zero_init=True,
        )
        
        # Track which parameters are for the value head (for separate optimization)
        self.value_head_params = list(self.value_head.parameters())
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        value_head_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> "AutoModelForCausalLMWithValueHead":
        """
        Load a pretrained model and add a value head.
        
        Args:
            model_name_or_path: HuggingFace model identifier or local path
            value_head_config: Configuration for the value head
            **kwargs: Arguments passed to AutoModelForCausalLM.from_pretrained
        
        Returns:
            AutoModelForCausalLMWithValueHead instance
        """
        pretrained_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **kwargs,
        )
        return cls(pretrained_model, value_head_config)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,  # Need hidden states for value head
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> Union[Tuple, "CausalLMOutputWithValue"]:
        """
        Forward pass with both language modeling and value prediction.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            past_key_values: Cached key-value pairs for efficient generation
            labels: Labels for language modeling loss
            use_cache: Whether to return key-value cache
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states (True by default for value head)
            return_dict: Whether to return a structured output
        
        Returns:
            CausalLMOutputWithValue containing:
                - loss: Language modeling loss (if labels provided)
                - logits: Token logits for policy
                - value: Value predictions
                - past_key_values, hidden_states, attentions: Same as base model
        """
        # Forward through the base model
        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,  # Always need hidden states for value head
            return_dict=True,
            **kwargs,
        )
        
        # Get hidden states for value head
        # Use the last layer's hidden states
        hidden_states = outputs.hidden_states[-1]
        
        # Compute values
        values = self.value_head(hidden_states, attention_mask)
        
        # Create output
        return CausalLMOutputWithValue(
            loss=outputs.loss,
            logits=outputs.logits,
            value=values,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
        )
    
    def generate(self, *args, **kwargs):
        """Forward generation to the base model."""
        return self.pretrained_model.generate(*args, **kwargs)
    
    def get_output_embeddings(self):
        """Get the output embeddings (LM head) from the base model."""
        return self.pretrained_model.get_output_embeddings()
    
    def set_output_embeddings(self, new_embeddings):
        """Set the output embeddings (LM head) on the base model."""
        self.pretrained_model.set_output_embeddings(new_embeddings)
    
    def save_pretrained(self, output_dir: str):
        """
        Save the model.
        
        Saves:
        - Base pretrained model
        - Value head weights separately
        """
        import os
        
        # Save the base model
        self.pretrained_model.save_pretrained(output_dir)
        
        # Save value head
        value_head_path = os.path.join(output_dir, "value_head.pt")
        torch.save(self.value_head.state_dict(), value_head_path)
        
        logger.info(f"Saved model with value head to {output_dir}")
    
    @property
    def device(self):
        """Get the device of the model."""
        return next(self.pretrained_model.parameters()).device
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing on the base model."""
        self.pretrained_model.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing on the base model."""
        self.pretrained_model.gradient_checkpointing_disable()


class CausalLMOutputWithValue:
    """
    Output of AutoModelForCausalLMWithValueHead.
    
    Extends the standard CausalLMOutput with value predictions.
    """
    
    def __init__(
        self,
        loss: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        hidden_states: Optional[Tuple[torch.Tensor]] = None,
        attentions: Optional[Tuple[torch.Tensor]] = None,
    ):
        self.loss = loss
        self.logits = logits
        self.value = value
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states
        self.attentions = attentions
    
    def __iter__(self):
        """Allow unpacking like a tuple."""
        return iter((self.loss, self.logits, self.value))
    
    def __getitem__(self, idx):
        """Allow indexing like a tuple."""
        return (self.loss, self.logits, self.value)[idx]


def add_value_head_to_model(
    model: PreTrainedModel,
    value_head_config: Optional[Dict[str, Any]] = None,
) -> AutoModelForCausalLMWithValueHead:
    """
    Convenience function to add a value head to an existing model.
    
    Args:
        model: A pretrained causal language model
        value_head_config: Configuration for the value head
    
    Returns:
        Model with value head attached
    
    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> model_with_value = add_value_head_to_model(model)
    """
    return AutoModelForCausalLMWithValueHead(model, value_head_config)
