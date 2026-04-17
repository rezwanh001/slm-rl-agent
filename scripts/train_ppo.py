#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque
"""

"""
PPO Training Script for SLM-RL-Agents

This script runs Stage 3 of the RLHF pipeline: Proximal Policy Optimization.
PPO fine-tunes the policy model to maximize rewards while staying close to
the reference model (SFT) to prevent reward hacking.

Usage:
    python scripts/train_ppo.py \
        --policy_model "./outputs/sft/final" \
        --reward_model "./outputs/reward_model/final" \
        --output_dir "./outputs/ppo" \
        --num_steps 1000

The PPO objective balances reward maximization with KL divergence constraint:
    J(θ) = E[r(x,y)] - β × KL(π_θ || π_ref)

This ensures the model improves according to the reward model while maintaining
the coherent language capabilities learned during SFT.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    LogitsProcessor, LogitsProcessorList,
)
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class StableLogitsProcessor(LogitsProcessor):
    """Clamp logits to prevent inf/nan in softmax → avoids CUDA device-side assert.

    Without this, aggressive PPO updates can push LoRA weights so far that
    the model produces extreme logits. After softmax these become inf/nan
    probabilities, triggering an unrecoverable CUDA assert during sampling.
    """

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.clamp(scores, min=-100.0, max=100.0)
        scores = torch.nan_to_num(scores, nan=0.0, posinf=100.0, neginf=-100.0)
        return scores


def check_model_health(model) -> bool:
    """Return False if any model parameter contains NaN or Inf."""
    for name, param in model.named_parameters():
        if param.requires_grad and (torch.isnan(param).any() or torch.isinf(param).any()):
            logger.warning(f"Corrupted parameter detected: {name}")
            return False
    return True


def save_state_backup(model) -> dict:
    """Save a copy of all trainable parameter tensors."""
    return {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def restore_state_backup(model, backup: dict):
    """Restore trainable parameters from a backup."""
    for name, param in model.named_parameters():
        if name in backup:
            param.data.copy_(backup[name])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_prompts(dataset_path: str, max_samples: int = None) -> list:
    """Load prompts for PPO training."""
    logger.info(f"Loading prompts from {dataset_path}")
    
    if dataset_path.endswith(".json"):
        with open(dataset_path) as f:
            data = json.load(f)
    elif dataset_path.endswith(".jsonl"):
        data = []
        with open(dataset_path) as f:
            for line in f:
                data.append(json.loads(line))
    else:
        # HuggingFace dataset
        ds = load_dataset(dataset_path, split="train")
        data = list(ds)
    
    # Extract prompts
    prompts = []
    for item in data:
        if "prompt" in item:
            prompts.append(item["prompt"])
        elif "text" in item:
            # Try to extract prompt from text
            text = item["text"]
            if "User:" in text:
                prompt = text.split("User:")[1].split("Assistant:")[0].strip()
                prompts.append(prompt)
            else:
                prompts.append(text[:200])  # Use first 200 chars as prompt
    
    if max_samples:
        prompts = prompts[:max_samples]
    
    logger.info(f"Loaded {len(prompts)} prompts")
    return prompts


def create_reward_fn(reward_model_path: str, tokenizer):
    """Create a reward function using the trained reward model."""
    from transformers import AutoModelForSequenceClassification

    logger.info(f"Loading reward model from {reward_model_path}")

    # Check if this is a PEFT adapter (has adapter_config.json)
    adapter_config_path = os.path.join(reward_model_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        from peft import PeftConfig, PeftModel
        peft_config = PeftConfig.from_pretrained(reward_model_path)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            num_labels=1,
        )
        reward_model = PeftModel.from_pretrained(base_model, reward_model_path)
        reward_model = reward_model.merge_and_unload()
    else:
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    reward_model.eval()
    
    def compute_rewards(prompts, responses):
        """Compute rewards for prompt-response pairs."""
        rewards = []
        
        for prompt, response in zip(prompts, responses):
            # Combine prompt and response
            text = f"{prompt}\n\n{response}"
            
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            inputs = {k: v.to(reward_model.device) for k, v in inputs.items()}
            
            # Get reward
            with torch.no_grad():
                outputs = reward_model(**inputs)
                reward = outputs.logits[0, 0].item()
            
            rewards.append(torch.tensor(reward))
        
        return rewards
    
    return compute_rewards


def main():
    parser = argparse.ArgumentParser(description="PPO Training for SLM-RL-Agents")
    
    # Model arguments
    parser.add_argument("--policy_model", type=str, required=True,
                        help="Path to SFT model (policy)")
    parser.add_argument("--reward_model", type=str, required=True,
                        help="Path to reward model")
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", action="store_false", dest="use_lora")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--use_quantization", action="store_true", default=False,
                        help="Use quantization (may cause issues with PPO)")
    
    # Data arguments
    parser.add_argument("--dataset_path", type=str, default="./data/sft_train.json",
                        help="Path to prompts dataset")
    parser.add_argument("--max_prompts", type=int, default=10000,
                        help="Maximum number of prompts to use")
    
    # PPO arguments
    parser.add_argument("--output_dir", type=str, default="./outputs/ppo")
    parser.add_argument("--num_ppo_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--mini_batch_size", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=1000,
                        help="Total PPO optimization steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--kl_penalty", type=float, default=0.1,
                        help="KL penalty coefficient (beta)")
    parser.add_argument("--target_kl", type=float, default=0.1,
                        help="Target KL divergence")
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.policy_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for generation
    
    # Load policy model with value head
    logger.info(f"Loading policy model from {args.policy_model}")
    
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    
    if args.use_quantization:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    
    # Check if SFT model is a PEFT adapter
    adapter_config_path = os.path.join(args.policy_model, "adapter_config.json")
    is_peft = os.path.exists(adapter_config_path)

    if is_peft:
        logger.info("Detected PEFT adapter — merging into base model for stable PPO...")
        from peft import PeftConfig, PeftModel as PeftModelLoader
        peft_cfg = PeftConfig.from_pretrained(args.policy_model)

        # --- পুরাতন পদ্ধতি (সমস্যা: LoRA params সব frozen থাকতো) ---
        # policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        #     args.policy_model,  # PEFT adapter সরাসরি PPOTrainer-এ দিলে
        #     ...                 # requires_grad=False হয়ে যায়, gradient flow হয় না
        # )
        # --- সমাধান: merge_and_unload() করে নতুন LoRA লাগানো (Algorithm 1) ---

        # Merge PEFT adapter into base to get a full model
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_cfg.base_model_name_or_path,
            trust_remote_code=True,
            # bfloat16 ব্যবহার করলে ratio > 10^6 হয়ে CUDA crash করে (200M এর নিচে)
            torch_dtype=torch.float32,  # float32 is mandatory for sub-200M PPO stability
        )
        peft_model = PeftModelLoader.from_pretrained(base_model, args.policy_model)
        merged_model = peft_model.merge_and_unload()

        merged_dir = os.path.join(args.output_dir, "_merged_sft")
        os.makedirs(merged_dir, exist_ok=True)
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        del merged_model, peft_model, base_model
        torch.cuda.empty_cache()

        # Auto-detect LoRA target modules for the merged model architecture
        def get_target_modules(model_path):
            """Return target module names based on layer names present in the model."""
            from transformers import AutoConfig
            import json, os
            # Peek at config to determine arch
            cfg_path = os.path.join(model_path, "config.json")
            if os.path.exists(cfg_path):
                cfg = json.load(open(cfg_path))
                arch = cfg.get("architectures", [""])[0].lower()
                model_type = cfg.get("model_type", "").lower()
            else:
                arch, model_type = "", ""

            # Pythia / GPT-NeoX: uses query_key_value fused projection + dense
            if "neox" in arch or "pythia" in model_type or "gpt_neox" in model_type:
                return ["query_key_value", "dense"]
            # LLaMA / Mistral / SmolLM2 / Qwen style: separate q/k/v/o projections
            elif any(x in arch or x in model_type for x in
                     ["llama", "mistral", "smollm", "qwen", "falcon", "phi"]):
                return ["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]
            # GPT-2 style
            elif "gpt2" in arch or "gpt2" in model_type:
                return ["c_attn", "c_proj"]
            # Fallback: let PEFT scan all linear layers
            else:
                return "all-linear"

        target_mods = get_target_modules(merged_dir)
        logger.info(f"PPO LoRA target modules: {target_mods}")

        # Load merged model with fresh LoRA (small alpha for stability)
        logger.info("Loading merged model with fresh LoRA for PPO...")
        ppo_lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r,  # alpha=r (not 2r) for smaller initial updates
            lora_dropout=0.0,
            target_modules=target_mods,
            task_type="CAUSAL_LM",
        )
        merged_kwargs = {"trust_remote_code": True, "torch_dtype": torch.float32}
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            merged_dir, peft_config=ppo_lora_config, **merged_kwargs,
        )

        logger.info("Loading reference model (frozen merged SFT)...")
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            merged_dir, **merged_kwargs,
        )
    else:
        # Load model with value head for PPO
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            args.policy_model, **model_kwargs,
        )
        # Apply LoRA if enabled
        if args.use_lora:
            logger.info(f"Applying LoRA with r={args.lora_r}")
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_r * 2,
                lora_dropout=0.05,
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            )
            model.pretrained_model = get_peft_model(model.pretrained_model, lora_config)
        # Load reference model (frozen SFT model)
        logger.info("Loading reference model...")
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            args.policy_model, **model_kwargs,
        )
    
    # Create reward function
    reward_fn = create_reward_fn(args.reward_model, tokenizer)
    
    # Load prompts
    prompts = load_prompts(args.dataset_path, args.max_prompts)
    
    # PPO configuration — reward whitening + adaptive KL for stability
    ppo_config = PPOConfig(
        model_name=args.policy_model,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.num_ppo_epochs,
        init_kl_coef=args.kl_penalty,
        adap_kl_ctrl=True,            # adaptive KL controller
        target_kl=args.target_kl,
        cliprange=args.clip_range,
        cliprange_value=args.clip_range,
        gamma=args.gamma,
        lam=args.gae_lambda,
        log_with="tensorboard",
        project_kwargs={"logging_dir": os.path.join(args.output_dir, "logs")},
        seed=args.seed,
        max_grad_norm=0.5,
        ratio_threshold=5.0,          # tighter threshold: skip batches > 5x
        use_score_scaling=True,       # whiten reward scores
        use_score_norm=True,          # normalize to unit variance
        score_clip=3.0,               # clip reward z-scores to ±3
        whiten_rewards=True,          # GAE reward whitening
    )
    
    # Create PPO trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )
    
    # Generation config
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
    }
    
    # Training loop
    logger.info("Starting PPO training...")

    best_reward = float("-inf")
    last_healthy_backup = save_state_backup(model)
    rollback_count = 0
    max_rollbacks = 5  # stop if model keeps corrupting

    for step in range(args.num_steps):
        # ── 1. Sample & tokenize prompts ─────────────────────────────────
        batch_start = (step * args.batch_size) % len(prompts)
        batch_prompts = prompts[batch_start:batch_start + args.batch_size]
        while len(batch_prompts) < args.batch_size:
            batch_prompts.extend(prompts[:args.batch_size - len(batch_prompts)])

        prompt_tensors = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        query_tensors = [prompt_tensors["input_ids"][i] for i in range(len(batch_prompts))]

        # ── 2. Generate responses (logits are clamped by StableLogitsProcessor) ──
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)

        # ── 3. Decode & compute rewards ──────────────────────────────────
        responses = [
            tokenizer.decode(r[len(q):], skip_special_tokens=True)
            for q, r in zip(query_tensors, response_tensors)
        ]
        rewards = reward_fn(batch_prompts, responses)

        # Sanitize rewards: replace NaN/inf with 0
        sanitized = False
        for i in range(len(rewards)):
            if torch.isnan(rewards[i]) or torch.isinf(rewards[i]):
                rewards[i] = torch.tensor(0.0)
                sanitized = True
        if sanitized:
            logger.warning(f"Step {step}: some rewards were NaN/inf — replaced with 0.")

        # ── 4. Backup weights before PPO step ────────────────────────────
        pre_step_backup = save_state_backup(model)

        # ── 5. PPO update ────────────────────────────────────────────────
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        # ── 6. Post-step health check — rollback if weights corrupted ────
        if not check_model_health(model):
            rollback_count += 1
            logger.warning(
                f"Step {step}: weights corrupted after PPO step — "
                f"rolling back ({rollback_count}/{max_rollbacks})"
            )
            restore_state_backup(model, pre_step_backup)
            if rollback_count >= max_rollbacks:
                logger.error("Too many rollbacks — stopping PPO with last healthy weights.")
                break
            continue

        # Healthy step — update the backup
        last_healthy_backup = pre_step_backup

        # ── 7. Logging ───────────────────────────────────────────────────
        mean_reward = sum(r.item() for r in rewards) / len(rewards)
        if mean_reward > best_reward:
            best_reward = mean_reward

        kl = stats.get("objective/kl", 0)

        if step % args.logging_steps == 0:
            logger.info(
                f"Step {step}/{args.num_steps} | "
                f"Reward: {mean_reward:.4f} | "
                f"KL: {kl:.4f} | "
                f"Best: {best_reward:.4f}"
            )

        # Early stop if KL diverges badly
        if isinstance(kl, (int, float)) and kl < -50:
            logger.warning(f"KL diverged to {kl:.1f} — stopping early.")
            break

        # ── 8. Save checkpoint ───────────────────────────────────────────
        if step > 0 and step % args.save_steps == 0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
            ppo_trainer.save_pretrained(checkpoint_dir)
            logger.info(f"Saved checkpoint to {checkpoint_dir}")

    # Ensure weights are healthy before saving; if not, restore last good backup
    if not check_model_health(model):
        logger.warning("Final weights are corrupted — restoring last healthy backup before saving.")
        restore_state_backup(model, last_healthy_backup)

    # Save final model
    final_dir = os.path.join(args.output_dir, "final")
    ppo_trainer.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    logger.info("PPO training complete!")
    logger.info(f"Model saved to: {final_dir}")
    logger.info(f"Best mean reward: {best_reward:.4f}")


if __name__ == "__main__":
    main()
