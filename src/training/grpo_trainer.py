#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque
"""

"""
Group Relative Policy Optimization (GRPO) Trainer for SLM-RL-Agent

This module implements GRPO, an advanced alignment technique used in DeepSeek R1.
GRPO eliminates the need for a separate value network by using group-level statistics
to estimate advantages, making it more memory-efficient than standard PPO.

HOW GRPO DIFFERS FROM PPO:
In standard PPO, we need four models in memory:
1. Policy model (actor)
2. Value model (critic)
3. Reference model (for KL penalty)
4. Reward model

GRPO eliminates the value model by:
1. Generating multiple responses per prompt (a "group")
2. Computing rewards for all responses in the group
3. Using the group's mean and std to normalize advantages
4. This provides a natural baseline without explicit value estimation

THE GRPO OBJECTIVE:
For a group of K responses {y_1, ..., y_K} to prompt x:

    A_i = (r_i - mean(r)) / std(r)  # Group-relative advantage
    
    L = -E[min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)] + β * KL(π || π_ref)

This is particularly effective for tasks with verifiable rewards (math, code)
where we can use rule-based reward functions instead of learned reward models.

References:
    - DeepSeek R1 Technical Report (2024)
    - Shao et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning"
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """
    Configuration for GRPO training.
    
    Key parameters:
        num_generations: Number of responses to generate per prompt (group size)
        kl_coef: Coefficient for KL penalty (β in the objective)
        clip_range: PPO clipping parameter (ε)
        
    Larger group sizes provide better advantage estimates but require more compute.
    Typical values are 4-16 generations per prompt.
    """
    # Generation settings
    num_generations: int = 4
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    
    # Training settings
    learning_rate: float = 1e-6
    num_steps: int = 500
    batch_size: int = 16  # Number of prompts per batch
    gradient_accumulation_steps: int = 4
    
    # GRPO-specific
    kl_coef: float = 0.1
    clip_range: float = 0.2
    clip_advantage: bool = True
    advantage_clip: float = 0.2
    
    # Reward settings
    reward_type: str = "model"  # "model" or "rule"
    
    # Output
    output_dir: str = "./outputs/grpo"
    logging_steps: int = 10
    save_steps: int = 100


class GRPOTrainer:
    """
    Group Relative Policy Optimization trainer.
    
    This trainer implements the GRPO algorithm which is particularly effective for:
    1. Tasks with verifiable rewards (math, code)
    2. Situations where you want to avoid training a value network
    3. When you have good reward signal but limited compute
    
    The key insight is that by generating multiple responses to the same prompt,
    we can use the group's statistics as a baseline, eliminating the need for
    a learned value function.
    
    Example:
        >>> config = GRPOConfig(num_generations=4, num_steps=500)
        >>> trainer = GRPOTrainer(
        ...     policy_model=model,
        ...     ref_model=ref_model,
        ...     tokenizer=tokenizer,
        ...     reward_fn=reward_function,
        ...     config=config,
        ... )
        >>> trainer.train(prompts)
    """
    
    def __init__(
        self,
        policy_model: PreTrainedModel,
        ref_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        reward_fn: Any,  # Reward model or function
        config: GRPOConfig,
    ):
        """
        Initialize the GRPO trainer.
        
        Args:
            policy_model: The policy model to train
            ref_model: Reference model for KL penalty (usually SFT checkpoint)
            tokenizer: Tokenizer for both models
            reward_fn: Either a reward model or a callable that computes rewards
            config: Training configuration
        """
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.config = config
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=config.learning_rate,
        )
        
        # Device
        self.device = next(policy_model.parameters()).device
        
        # Tracking
        self.global_step = 0
        self.metrics_history = []
    
    def train(self, prompts: List[str]) -> Dict[str, Any]:
        """
        Run GRPO training.
        
        Args:
            prompts: List of training prompts
        
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Starting GRPO training with {len(prompts)} prompts")
        logger.info(f"Config: {self.config}")
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Create dataloader
        prompt_dataloader = DataLoader(
            prompts,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        
        self.policy_model.train()
        
        progress_bar = tqdm(total=self.config.num_steps, desc="GRPO Training")
        
        step = 0
        while step < self.config.num_steps:
            for batch_prompts in prompt_dataloader:
                if step >= self.config.num_steps:
                    break
                
                # Generate group of responses for each prompt
                metrics = self._train_step(batch_prompts)
                
                step += 1
                self.global_step = step
                
                # Logging
                if step % self.config.logging_steps == 0:
                    self._log_metrics(metrics, step)
                
                # Saving
                if step % self.config.save_steps == 0:
                    self._save_checkpoint(step)
                
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "reward": f"{metrics.get('mean_reward', 0):.3f}",
                    "kl": f"{metrics.get('kl_div', 0):.3f}",
                })
        
        progress_bar.close()
        
        # Save final model
        self._save_checkpoint(step, final=True)
        
        return {"final_step": step, "metrics_history": self.metrics_history}
    
    def _train_step(self, prompts: List[str]) -> Dict[str, float]:
        """
        Execute one GRPO training step.
        
        For each prompt:
        1. Generate K responses
        2. Compute rewards for each response
        3. Compute group-relative advantages
        4. Update policy using clipped objective
        """
        all_losses = []
        all_rewards = []
        all_kls = []
        
        for prompt in prompts:
            # Generate group of responses
            responses, log_probs_old = self._generate_group(prompt)
            
            # Compute rewards
            rewards = self._compute_rewards(prompt, responses)
            all_rewards.extend(rewards)
            
            # Compute group-relative advantages
            advantages = self._compute_advantages(rewards)
            
            # Compute policy loss
            loss, kl = self._compute_loss(
                prompt, responses, log_probs_old, advantages
            )
            
            all_losses.append(loss)
            all_kls.append(kl)
        
        # Aggregate and backprop
        total_loss = torch.stack(all_losses).mean()
        total_loss.backward()
        
        # Gradient step
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return {
            "loss": total_loss.item(),
            "mean_reward": sum(all_rewards) / len(all_rewards),
            "kl_div": sum(all_kls) / len(all_kls) if all_kls else 0,
        }
    
    def _generate_group(
        self,
        prompt: str,
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """
        Generate a group of K responses for a prompt.
        
        Returns:
            Tuple of (responses, log_probs) where log_probs are the
            log probabilities under the current policy at generation time.
        """
        responses = []
        log_probs = []
        
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        
        for _ in range(self.config.num_generations):
            with torch.no_grad():
                # Generate response
                outputs = self.policy_model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                
                # Extract response tokens
                response_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                responses.append(response_text)
                
                # Compute log probabilities
                # Stack scores and compute log probs
                if outputs.scores:
                    scores = torch.stack(outputs.scores, dim=1)  # [1, seq_len, vocab]
                    log_probs_seq = F.log_softmax(scores, dim=-1)
                    # Gather log probs for generated tokens
                    token_log_probs = torch.gather(
                        log_probs_seq[0],
                        dim=-1,
                        index=response_ids.unsqueeze(-1)
                    ).squeeze(-1)
                    log_probs.append(token_log_probs.sum())
                else:
                    log_probs.append(torch.tensor(0.0))
        
        return responses, log_probs
    
    def _compute_rewards(
        self,
        prompt: str,
        responses: List[str],
    ) -> List[float]:
        """
        Compute rewards for a group of responses.
        
        Supports both learned reward models and rule-based rewards.
        """
        rewards = []
        
        for response in responses:
            if self.config.reward_type == "model":
                # Use reward model
                if hasattr(self.reward_fn, "score"):
                    reward = self.reward_fn.score(prompt, response)
                else:
                    reward = self.reward_fn(prompt, response)
            else:
                # Rule-based reward (e.g., for math/code)
                reward = self.reward_fn(prompt, response)
            
            rewards.append(float(reward))
        
        return rewards
    
    def _compute_advantages(self, rewards: List[float]) -> torch.Tensor:
        """
        Compute group-relative advantages.
        
        This is the key innovation of GRPO: instead of using a value network,
        we normalize advantages within the group.
        
            A_i = (r_i - mean(r)) / (std(r) + eps)
        
        This provides a natural baseline and reduces variance.
        """
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        mean_reward = rewards_tensor.mean()
        std_reward = rewards_tensor.std()
        
        # Normalize
        advantages = (rewards_tensor - mean_reward) / (std_reward + 1e-8)
        
        # Optionally clip advantages
        if self.config.clip_advantage:
            advantages = torch.clamp(
                advantages,
                -self.config.advantage_clip,
                self.config.advantage_clip,
            )
        
        return advantages
    
    def _compute_loss(
        self,
        prompt: str,
        responses: List[str],
        log_probs_old: List[torch.Tensor],
        advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute the GRPO policy loss.
        
        Uses the clipped PPO objective with group-relative advantages:
        
            L = -E[min(ratio * A, clip(ratio) * A)]
        
        Plus a KL penalty to prevent drift from the reference policy.
        """
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        total_kl = 0.0
        
        for i, (response, log_prob_old, advantage) in enumerate(
            zip(responses, log_probs_old, advantages)
        ):
            # Encode full sequence
            full_text = prompt + response
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(self.device)
            
            # Get current policy log probs
            with torch.enable_grad():
                outputs = self.policy_model(**inputs, labels=inputs["input_ids"])
                # Note: This is simplified - in practice you'd compute
                # log probs only for the response tokens
                log_prob_new = -outputs.loss * inputs["input_ids"].shape[1]
            
            # Get reference log probs
            with torch.no_grad():
                ref_outputs = self.ref_model(**inputs, labels=inputs["input_ids"])
                log_prob_ref = -ref_outputs.loss * inputs["input_ids"].shape[1]
            
            # Compute ratio
            log_prob_old_val = log_prob_old.to(self.device) if isinstance(log_prob_old, torch.Tensor) else torch.tensor(log_prob_old, device=self.device)
            ratio = torch.exp(log_prob_new - log_prob_old_val)
            
            # Clipped objective
            advantage_val = advantage.to(self.device)
            unclipped = ratio * advantage_val
            clipped = torch.clamp(
                ratio,
                1 - self.config.clip_range,
                1 + self.config.clip_range,
            ) * advantage_val
            
            policy_loss = -torch.min(unclipped, clipped)
            
            # KL penalty
            kl = log_prob_new - log_prob_ref
            total_kl += kl.item()
            
            # Total loss
            loss = policy_loss + self.config.kl_coef * kl
            total_loss = total_loss + loss
        
        avg_loss = total_loss / len(responses)
        avg_kl = total_kl / len(responses)
        
        return avg_loss, avg_kl
    
    def _log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log training metrics."""
        logger.info(
            f"Step {step}: "
            f"loss={metrics.get('loss', 0):.4f}, "
            f"reward={metrics.get('mean_reward', 0):.4f}, "
            f"kl={metrics.get('kl_div', 0):.4f}"
        )
        self.metrics_history.append({"step": step, **metrics})
    
    def _save_checkpoint(self, step: int, final: bool = False) -> None:
        """Save model checkpoint."""
        if final:
            save_path = os.path.join(self.config.output_dir, "final")
        else:
            save_path = os.path.join(self.config.output_dir, f"checkpoint-{step}")
        
        os.makedirs(save_path, exist_ok=True)
        
        self.policy_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"Saved checkpoint to {save_path}")


def create_verifiable_reward_function(task_type: str = "math"):
    """
    Create a rule-based reward function for verifiable tasks.
    
    These are particularly useful with GRPO because they provide
    accurate, deterministic rewards without a learned reward model.
    
    Args:
        task_type: Type of task ("math", "code", "format")
    
    Returns:
        Callable that computes rewards
    """
    
    if task_type == "math":
        def math_reward(prompt: str, response: str) -> float:
            """
            Reward function for math problems.
            
            Extracts the answer from the response and checks correctness.
            Returns 1.0 for correct, 0.0 for incorrect.
            """
            import re
            
            # Try to extract expected answer from prompt
            # This assumes format like "What is 2+2? Answer: "
            # You'd customize this for your actual format
            
            # Extract number from response
            numbers = re.findall(r'-?\d+\.?\d*', response)
            if not numbers:
                return 0.0
            
            # Simple heuristic: longer, well-formatted responses get partial credit
            base_reward = 0.1 if len(response) > 10 else 0.0
            
            # Check for reasoning indicators
            if any(word in response.lower() for word in ["because", "therefore", "so", "thus"]):
                base_reward += 0.2
            
            return min(base_reward + 0.5, 1.0)
        
        return math_reward
    
    elif task_type == "format":
        def format_reward(prompt: str, response: str) -> float:
            """
            Reward function for format compliance.
            
            Checks if response follows requested format (e.g., JSON, XML).
            """
            import json
            
            reward = 0.0
            
            # Check if valid JSON when requested
            if "json" in prompt.lower():
                try:
                    json.loads(response)
                    reward = 1.0
                except json.JSONDecodeError:
                    # Partial credit for attempting JSON-like structure
                    if "{" in response and "}" in response:
                        reward = 0.3
            
            # Check length requirements
            if "brief" in prompt.lower() and len(response) < 200:
                reward += 0.2
            elif "detailed" in prompt.lower() and len(response) > 500:
                reward += 0.2
            
            return min(reward, 1.0)
        
        return format_reward
    
    else:
        # Default: simple length-based reward
        def default_reward(prompt: str, response: str) -> float:
            # Reward reasonable length responses
            length = len(response)
            if length < 10:
                return 0.1
            elif length < 50:
                return 0.3
            elif length < 500:
                return 0.7
            else:
                return 0.5  # Penalize very long responses
        
        return default_reward
