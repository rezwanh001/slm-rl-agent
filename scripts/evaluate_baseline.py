#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque
"""

"""
Evaluate external SOTA SLM baselines using the same metrics as our models.

This script loads a HuggingFace instruct-tuned SLM, runs generation on the same
evaluation prompts we use for our models, and computes perplexity + diversity +
reward scores against our own reward models. This lets us compare against
published SOTA at comparable parameter counts.

Usage:
    python scripts/evaluate_baseline.py \
        --model_name HuggingFaceTB/SmolLM2-135M-Instruct \
        --eval_dataset data/tinystories/sft_eval.json \
        --reward_model_path outputs/smollm2-135m/tinystories/reward_model/final \
        --output_dir outputs/baselines/smollm2-135m-instruct/tinystories
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_eval_data(path: str, max_samples: int = 200) -> List[Dict]:
    with open(path) as f:
        data = json.load(f)
    return data[:max_samples]


def extract_prompts_references(data: List[Dict]):
    prompts, refs = [], []
    for item in data:
        text = item.get("text", "")
        if "User:" in text and "Assistant:" in text:
            parts = text.split("Assistant:")
            prompts.append(parts[0].replace("User:", "").strip())
            refs.append(parts[1].strip() if len(parts) > 1 else "")
        else:
            prompts.append(text[:200])
            refs.append(text[200:])
    return prompts, refs


def compute_perplexity(model, tokenizer, texts: List[str], batch_size: int = 8) -> float:
    model.eval()
    device = next(model.parameters()).device
    total_loss, total_tokens = 0.0, 0
    for i in tqdm(range(0, len(texts), batch_size), desc="Perplexity"):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            out = model(**enc, labels=labels)
            n = enc["attention_mask"].sum().item()
            if not (torch.isnan(out.loss) or torch.isinf(out.loss)):
                total_loss += out.loss.item() * n
                total_tokens += n
    avg_loss = total_loss / max(total_tokens, 1)
    return float(torch.exp(torch.tensor(avg_loss)).item())


def generate_responses(model, tokenizer, prompts: List[str],
                       max_new_tokens: int = 128, temperature: float = 0.8,
                       batch_size: int = 8) -> List[str]:
    model.eval()
    device = next(model.parameters()).device
    out_texts = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch = prompts[i:i + batch_size]
        # For instruct models, wrap in chat template if available
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            wrapped = [tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            ) for p in batch]
        else:
            wrapped = batch
        enc = tokenizer(wrapped, return_tensors="pt", padding=True, truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        for j, g in enumerate(gen):
            prompt_len = enc["input_ids"][j].shape[0]
            out_texts.append(tokenizer.decode(g[prompt_len:], skip_special_tokens=True).strip())
    return out_texts


def compute_distinct_n(texts: List[str], n: int) -> float:
    ngrams = []
    for t in texts:
        toks = t.lower().split()
        ngrams.extend([tuple(toks[i:i+n]) for i in range(len(toks) - n + 1)])
    return len(set(ngrams)) / len(ngrams) if ngrams else 0.0


def compute_rewards(reward_model_path: str, our_tokenizer, prompts, responses) -> Dict:
    """Score baseline outputs with OUR reward model (fair comparison on our reward scale)."""
    from transformers import AutoModelForSequenceClassification
    adapter = os.path.join(reward_model_path, "adapter_config.json")
    if os.path.exists(adapter):
        from peft import PeftConfig, PeftModel
        cfg = PeftConfig.from_pretrained(reward_model_path)
        base = AutoModelForSequenceClassification.from_pretrained(
            cfg.base_model_name_or_path, torch_dtype=torch.float32,
            device_map="auto", num_labels=1,
        )
        rm = PeftModel.from_pretrained(base, reward_model_path).merge_and_unload()
    else:
        rm = AutoModelForSequenceClassification.from_pretrained(
            reward_model_path, torch_dtype=torch.float32, device_map="auto")
    rm.eval()

    scores = []
    for p, r in tqdm(zip(prompts, responses), desc="Rewarding", total=len(prompts)):
        text = f"{p}\n\n{r}"
        enc = our_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        enc = {k: v.to(rm.device) for k, v in enc.items()}
        with torch.no_grad():
            out = rm(**enc)
            scores.append(out.logits[0, 0].item())
    return {
        "reward_mean": float(np.mean(scores)),
        "reward_std": float(np.std(scores)),
        "reward_min": float(np.min(scores)),
        "reward_max": float(np.max(scores)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--eval_dataset", required=True)
    ap.add_argument("--reward_model_path", required=True)
    ap.add_argument("--our_tokenizer_path", required=True,
                    help="Path to our SFT checkpoint — used for scoring with reward model")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_samples", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Loading baseline model: {args.model_name}")
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32,
        device_map="auto", trust_remote_code=True,
    )

    # Load our tokenizer for reward scoring (must match reward model's vocab)
    our_tok = AutoTokenizer.from_pretrained(args.our_tokenizer_path, trust_remote_code=True)
    if our_tok.pad_token is None:
        our_tok.pad_token = our_tok.eos_token

    data = load_eval_data(args.eval_dataset, args.max_samples)
    prompts, refs = extract_prompts_references(data)
    logger.info(f"Loaded {len(prompts)} prompts")

    metrics = {}

    # Perplexity on reference text
    texts_for_ppl = [f"{p}\n\n{r}" for p, r in zip(prompts, refs) if r]
    if texts_for_ppl:
        metrics["perplexity"] = compute_perplexity(model, tok, texts_for_ppl, args.batch_size)
        logger.info(f"Perplexity: {metrics['perplexity']:.2f}")

    # Generation
    gens = generate_responses(model, tok, prompts, args.max_new_tokens, args.temperature, args.batch_size)

    # Diversity
    metrics["distinct_1"] = compute_distinct_n(gens, 1)
    metrics["distinct_2"] = compute_distinct_n(gens, 2)
    metrics["distinct_3"] = compute_distinct_n(gens, 3)
    metrics["avg_response_length"] = sum(len(g.split()) for g in gens) / len(gens)

    # ROUGE
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge = {'rouge1_f1': [], 'rouge2_f1': [], 'rougeL_f1': []}
        for g, r in zip(gens, refs):
            s = scorer.score(r, g)
            rouge['rouge1_f1'].append(s['rouge1'].fmeasure)
            rouge['rouge2_f1'].append(s['rouge2'].fmeasure)
            rouge['rougeL_f1'].append(s['rougeL'].fmeasure)
        for k, v in rouge.items():
            metrics[k] = float(np.mean(v))
    except ImportError:
        pass

    # Free baseline model before loading reward model
    del model
    torch.cuda.empty_cache()

    # Reward scoring (using our tokenizer for reward model's vocab)
    metrics.update(compute_rewards(args.reward_model_path, our_tok, prompts, gens))

    # Save
    result = {
        "model_name": args.model_name,
        "num_samples": len(prompts),
        "metrics": metrics,
    }
    with open(os.path.join(args.output_dir, "evaluation_results.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(args.output_dir, "sample_generations.json"), "w") as f:
        json.dump([{"prompt": p, "generated": g, "reference": r}
                   for p, g, r in zip(prompts[:50], gens[:50], refs[:50])], f, indent=2)

    logger.info("=" * 60)
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
