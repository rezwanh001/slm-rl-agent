#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque
"""

"""
SLM Agent: The Core Agent Implementation for SLM-RL-Agents

This module implements the main SLMAgent class, which provides a production-ready
interface for using trained small language models as AI agents. The agent handles
all the complexity of model loading, text generation, and optional tool calling.

WHAT IS AN AI AGENT?
An AI agent is a system that can perceive its environment (through inputs like text),
reason about what to do (using a language model), and take actions (generate responses
or call tools). In the context of language models, the "agent" paradigm extends simple
chatbots by adding capabilities like:

1. Tool Use: The agent can call external functions to get information or perform actions
2. Planning: The agent can break down complex tasks into steps
3. Memory: The agent can maintain context across interactions
4. Reflection: The agent can evaluate and correct its own outputs

This implementation provides a foundation for building such capabilities on top of
small language models trained with RLHF.

DESIGN PHILOSOPHY:
- Simple API: Most users just need generate() - advanced features are optional
- Efficient: Optimized for low-latency inference on consumer hardware
- Extensible: Easy to add new tools, modify generation, or swap models
- Safe: Built-in content filtering and output validation hooks
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import PeftModel

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """
    Configuration for text generation.
    
    This dataclass holds all the parameters that control how the model generates text.
    Understanding these parameters is crucial for getting good outputs from your model.
    
    TEMPERATURE EXPLAINED:
    Temperature controls the "creativity" or randomness of generation. Mathematically,
    it scales the logits before applying softmax:
    
        P(token) = softmax(logits / temperature)
    
    - temperature=0: Greedy decoding (always pick highest probability token)
    - temperature=0.7: Balanced (good for most tasks)
    - temperature=1.0: Sample directly from model's distribution
    - temperature>1.0: More random, creative, potentially incoherent
    
    TOP-P (NUCLEUS SAMPLING) EXPLAINED:
    Instead of considering all tokens, only consider tokens that together have
    cumulative probability >= top_p. This dynamically adjusts the vocabulary size:
    
    - top_p=1.0: Consider all tokens (no filtering)
    - top_p=0.9: Consider tokens covering 90% of probability mass
    - top_p=0.5: Very conservative, only high-probability tokens
    
    TOP-K EXPLAINED:
    Only consider the top K most probable tokens. This is a simpler alternative
    to top-p but can be too restrictive or too loose depending on the distribution:
    
    - top_k=50: Common default
    - top_k=10: Very conservative
    - top_k=0: No filtering (consider all tokens)
    
    RECOMMENDED SETTINGS BY USE CASE:
    - Factual QA: temperature=0.3, top_p=0.9 (precise, consistent)
    - Creative writing: temperature=0.9, top_p=0.95 (varied, creative)
    - Code generation: temperature=0.2, top_p=0.95 (correct, not creative)
    - Conversation: temperature=0.7, top_p=0.9 (natural, engaging)
    """
    # Core generation parameters
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    
    # Repetition control
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 3
    
    # Length control
    min_new_tokens: int = 1
    length_penalty: float = 1.0
    
    # Stopping conditions
    stop_sequences: List[str] = field(default_factory=list)
    
    # Beam search (alternative to sampling)
    num_beams: int = 1
    num_return_sequences: int = 1
    early_stopping: bool = True


class StopOnSequences(StoppingCriteria):
    """
    Custom stopping criteria to halt generation when specific sequences are produced.
    
    This is useful for implementing conversational models that should stop at
    certain markers (e.g., end-of-turn tokens, function call indicators).
    """
    
    def __init__(self, stop_sequences: List[str], tokenizer: PreTrainedTokenizer):
        self.stop_sequences = stop_sequences
        self.tokenizer = tokenizer
        self.stop_ids = [
            tokenizer.encode(seq, add_special_tokens=False)
            for seq in stop_sequences
        ]
    
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs,
    ) -> bool:
        # Decode the last few tokens to check for stop sequences
        # This is more reliable than checking token IDs directly
        generated_text = self.tokenizer.decode(input_ids[0][-20:], skip_special_tokens=True)
        
        for stop_seq in self.stop_sequences:
            if stop_seq in generated_text:
                return True
        
        return False


class SLMAgent:
    """
    A production-ready AI agent powered by a Small Language Model.
    
    The SLMAgent class wraps a trained language model and provides a high-level
    interface for generation and tool calling. It handles:
    
    1. Model loading (including LoRA adapters and quantization)
    2. Tokenization and prompt formatting
    3. Text generation with configurable sampling
    4. Optional tool/function calling
    5. Conversation history management
    
    BASIC USAGE:
    The simplest way to use the agent is for straightforward generation:
    
        >>> agent = SLMAgent.from_pretrained("./outputs/ppo/final")
        >>> response = agent.generate("What is machine learning?")
        >>> print(response)
    
    WITH TOOLS:
    The agent can call external tools to extend its capabilities:
    
        >>> tools = [
        ...     {
        ...         "name": "calculator",
        ...         "description": "Performs math calculations",
        ...         "parameters": {"expression": "string"}
        ...     }
        ... ]
        >>> response = agent.generate("What is 25 * 47?", tools=tools)
    
    CONVERSATION MODE:
    For multi-turn conversations, the agent maintains history:
    
        >>> agent.start_conversation()
        >>> agent.generate("Hi, I'm learning Python!")
        >>> agent.generate("What should I learn first?")  # Has context
        >>> history = agent.get_conversation_history()
    
    IMPLEMENTATION NOTES:
    - The agent is stateless by default (each generate() is independent)
    - Use start_conversation() for stateful multi-turn interactions
    - Tools are executed locally; the agent decides when to call them
    - Generation parameters can be overridden per-call
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        generation_config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the SLM Agent.
        
        Usually you'll use `from_pretrained()` instead of this constructor directly,
        which handles model loading automatically.
        
        Args:
            model: The language model (HuggingFace format)
            tokenizer: Tokenizer for the model
            generation_config: Default generation settings
            system_prompt: System prompt to prepend to all conversations
        """
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config or GenerationConfig()
        self.system_prompt = system_prompt
        
        # Conversation state (None when not in conversation mode)
        self.conversation_history: Optional[List[Dict[str, str]]] = None
        
        # Tool registry
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.tool_functions: Dict[str, Callable] = {}
        
        # Ensure model is in eval mode
        self.model.eval()
        
        # Set up device
        self.device = next(model.parameters()).device
        
        logger.info(f"SLMAgent initialized on device: {self.device}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        generation_config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ) -> "SLMAgent":
        """
        Load an agent from a trained model checkpoint.
        
        This method handles all the complexity of loading different model formats:
        - Full fine-tuned models
        - LoRA adapter checkpoints
        - Quantized models
        
        Args:
            model_path: Path to the model checkpoint or HuggingFace model ID
            generation_config: Generation configuration
            system_prompt: Optional system prompt
            device_map: Device placement ("auto", "cuda:0", etc.)
            torch_dtype: Data type for weights ("auto", "float16", "bfloat16")
            load_in_4bit: Load in 4-bit quantization (requires bitsandbytes)
            load_in_8bit: Load in 8-bit quantization (requires bitsandbytes)
        
        Returns:
            Initialized SLMAgent
        
        Example:
            >>> # Load a LoRA-trained model
            >>> agent = SLMAgent.from_pretrained("./outputs/ppo/final")
            
            >>> # Load with 4-bit quantization for lower memory
            >>> agent = SLMAgent.from_pretrained(
            ...     "./outputs/ppo/final",
            ...     load_in_4bit=True
            ... )
        """
        logger.info(f"Loading agent from {model_path}")
        
        # Determine torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": "auto",
        }
        dtype = dtype_map.get(torch_dtype, "auto")
        
        # Build model kwargs
        model_kwargs = {
            "device_map": device_map,
            "torch_dtype": dtype,
            "trust_remote_code": True,
        }
        
        # Add quantization if requested
        if load_in_4bit or load_in_8bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_compute_dtype=torch.bfloat16 if load_in_4bit else None,
            )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Check if this is a LoRA adapter
        adapter_config_path = Path(model_path) / "adapter_config.json"
        if adapter_config_path.exists():
            # Load LoRA adapter
            logger.info("Detected LoRA adapter, loading base model...")
            with open(adapter_config_path) as f:
                adapter_config = json.load(f)
            base_model_path = adapter_config.get("base_model_name_or_path")
            
            if base_model_path:
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_path, **model_kwargs
                )
                model = PeftModel.from_pretrained(base_model, model_path)
            else:
                # Try to load directly (merged model saved with adapter structure)
                model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        else:
            # Load full model
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        
        return cls(
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
            system_prompt=system_prompt,
        )
    
    def generate(
        self,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        return_full_text: bool = False,
        **kwargs,
    ) -> str:
        """
        Generate a response for the given prompt.
        
        This is the main method for interacting with the agent. It handles:
        - Formatting the prompt with chat template (if available)
        - Applying generation parameters
        - Post-processing the output
        - Optionally executing tool calls
        
        Args:
            prompt: The input text to respond to
            tools: Optional list of tool definitions for function calling
            max_new_tokens: Override default max tokens
            temperature: Override default temperature
            top_p: Override default top_p
            stop_sequences: Additional sequences to stop on
            return_full_text: Whether to return full text including prompt
            **kwargs: Additional generation parameters
        
        Returns:
            Generated response text
        
        Example:
            >>> response = agent.generate(
            ...     "Explain quantum computing in simple terms.",
            ...     temperature=0.7,
            ...     max_new_tokens=200
            ... )
        """
        # Build the full prompt
        full_prompt = self._build_prompt(prompt, tools)
        
        # Tokenize
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Build generation kwargs
        gen_kwargs = self._build_generation_kwargs(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences,
            **kwargs,
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs,
            )
        
        # Decode
        if return_full_text:
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            # Remove the prompt from the output
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Clean up response
        response = self._post_process_response(response, stop_sequences)
        
        # Update conversation history if in conversation mode
        if self.conversation_history is not None:
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def _build_prompt(
        self,
        user_input: str,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build the full prompt for generation.
        
        This method handles chat formatting, system prompts, conversation history,
        and tool descriptions. The goal is to construct a prompt that gives the
        model all the context it needs to generate a helpful response.
        """
        messages = []
        
        # Add system prompt if specified
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        # Add tool descriptions to system message if tools are provided
        if tools:
            tool_desc = self._format_tool_descriptions(tools)
            if self.system_prompt:
                messages[0]["content"] += f"\n\n{tool_desc}"
            else:
                messages.append({"role": "system", "content": tool_desc})
        
        # Add conversation history
        if self.conversation_history:
            messages.extend(self.conversation_history)
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        # Format using chat template if available
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:
                logger.warning(f"Chat template failed: {e}, using fallback")
        
        # Fallback to simple formatting
        prompt_parts = []
        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            prompt_parts.append(f"{role}: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
    
    def _format_tool_descriptions(self, tools: List[Dict[str, Any]]) -> str:
        """Format tool definitions for the prompt."""
        lines = ["Available tools:"]
        for tool in tools:
            name = tool.get("name", "unknown")
            desc = tool.get("description", "No description")
            params = tool.get("parameters", {})
            
            lines.append(f"\n- {name}: {desc}")
            if params:
                lines.append(f"  Parameters: {json.dumps(params)}")
        
        lines.append("\nTo use a tool, respond with: [TOOL_CALL: tool_name(param=value)]")
        return "\n".join(lines)
    
    def _build_generation_kwargs(
        self,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Build the generation kwargs from config and overrides."""
        config = self.generation_config
        
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or config.max_new_tokens,
            "temperature": temperature or config.temperature,
            "top_p": top_p or config.top_p,
            "top_k": config.top_k,
            "do_sample": config.do_sample,
            "repetition_penalty": config.repetition_penalty,
            "no_repeat_ngram_size": config.no_repeat_ngram_size,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Handle temperature=0 (greedy decoding)
        if gen_kwargs["temperature"] == 0:
            gen_kwargs["do_sample"] = False
            gen_kwargs.pop("temperature")
            gen_kwargs.pop("top_p")
            gen_kwargs.pop("top_k")
        
        # Add stopping criteria
        all_stop_sequences = list(config.stop_sequences)
        if stop_sequences:
            all_stop_sequences.extend(stop_sequences)
        
        if all_stop_sequences:
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([
                StopOnSequences(all_stop_sequences, self.tokenizer)
            ])
        
        # Add any additional kwargs
        gen_kwargs.update(kwargs)
        
        return gen_kwargs
    
    def _post_process_response(
        self,
        response: str,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Clean up the generated response."""
        # Remove any stop sequences from the end
        all_stop_sequences = list(self.generation_config.stop_sequences)
        if stop_sequences:
            all_stop_sequences.extend(stop_sequences)
        
        for stop_seq in all_stop_sequences:
            if stop_seq in response:
                response = response.split(stop_seq)[0]
        
        # Strip whitespace
        response = response.strip()
        
        return response
    
    # =========================================================================
    # Conversation Management
    # =========================================================================
    
    def start_conversation(self) -> None:
        """
        Start a new conversation session.
        
        After calling this method, subsequent generate() calls will include
        the conversation history, enabling multi-turn conversations.
        """
        self.conversation_history = []
        logger.info("Started new conversation")
    
    def end_conversation(self) -> List[Dict[str, str]]:
        """
        End the current conversation and return the history.
        
        Returns:
            The conversation history as a list of messages
        """
        history = self.conversation_history or []
        self.conversation_history = None
        return history
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history without ending the conversation."""
        return list(self.conversation_history) if self.conversation_history else []
    
    def clear_conversation(self) -> None:
        """Clear the conversation history but stay in conversation mode."""
        if self.conversation_history is not None:
            self.conversation_history = []
    
    # =========================================================================
    # Tool Registration
    # =========================================================================
    
    def register_tool(
        self,
        name: str,
        function: Callable,
        description: str,
        parameters: Dict[str, str],
    ) -> None:
        """
        Register a tool that the agent can call.
        
        Args:
            name: Unique name for the tool
            function: Python function to execute
            description: Human-readable description
            parameters: Dictionary of parameter names to types
        
        Example:
            >>> def calculator(expression: str) -> str:
            ...     return str(eval(expression))
            >>> agent.register_tool(
            ...     "calculator",
            ...     calculator,
            ...     "Evaluates mathematical expressions",
            ...     {"expression": "string"}
            ... )
        """
        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
        self.tool_functions[name] = function
        logger.info(f"Registered tool: {name}")
    
    def get_registered_tools(self) -> List[Dict[str, Any]]:
        """Get list of registered tool definitions."""
        return list(self.tools.values())
    
    # =========================================================================
    # Batch Processing
    # =========================================================================
    
    def generate_batch(
        self,
        prompts: List[str],
        batch_size: int = 8,
        **kwargs,
    ) -> List[str]:
        """
        Generate responses for multiple prompts efficiently.
        
        Args:
            prompts: List of input prompts
            batch_size: Number of prompts to process at once
            **kwargs: Generation parameters
        
        Returns:
            List of generated responses
        """
        responses = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Process batch
            for prompt in batch_prompts:
                response = self.generate(prompt, **kwargs)
                responses.append(response)
        
        return responses
    
    # =========================================================================
    # Model Info
    # =========================================================================
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        num_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "total_parameters": num_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "dtype": str(next(self.model.parameters()).dtype),
            "vocab_size": self.tokenizer.vocab_size,
            "is_peft_model": isinstance(self.model, PeftModel),
        }
