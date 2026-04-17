#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque
"""

"""
Evaluation Script for SLM-RL-Agents

This script runs comprehensive evaluation on trained models, computing metrics across
multiple dimensions: language modeling quality, generation quality, reward scores,
and diversity. It helps diagnose training issues and compare model variants.

Usage:
    python scripts/evaluate.py \
        --model_path "./outputs/ppo/final" \
        --eval_dataset "./data/sft_eval.json" \
        --output_dir "./outputs/evaluation"

The evaluation covers several complementary perspectives. Perplexity tells you how well
the model predicts text (fundamental capability). Generation metrics like BLEU and ROUGE
compare outputs against references. Reward scores indicate alignment quality. And
diversity metrics ensure the model hasn't collapsed to repetitive outputs.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_eval_data(dataset_path: str, max_samples: int = None) -> Dataset:
    """Load evaluation dataset."""
    logger.info(f"Loading evaluation data from {dataset_path}")
    
    if dataset_path.endswith(".json"):
        with open(dataset_path) as f:
            data = json.load(f)
        dataset = Dataset.from_list(data)
    else:
        dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    logger.info(f"Loaded {len(dataset)} evaluation examples")
    return dataset


def compute_perplexity(model, tokenizer, texts: List[str], batch_size: int = 8) -> float:
    """Compute perplexity on a list of texts."""
    model.eval()
    device = next(model.parameters()).device
    
    total_loss = 0.0
    total_tokens = 0
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing perplexity"):
        batch_texts = texts[i:i + batch_size]
        
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Mask padding tokens in labels (-100 is ignored by CrossEntropyLoss)
            labels = inputs["input_ids"].clone()
            labels[inputs["attention_mask"] == 0] = -100

            outputs = model(**inputs, labels=labels)

            # Count actual tokens (non-padding)
            mask = inputs["attention_mask"]
            num_tokens = mask.sum().item()
            loss_val = outputs.loss.item()
            if not (torch.isnan(outputs.loss) or torch.isinf(outputs.loss)):
                total_loss += loss_val * num_tokens
                total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity


def generate_responses(
    model, tokenizer, prompts: List[str],
    max_new_tokens: int = 128, temperature: float = 0.7, batch_size: int = 8
) -> List[str]:
    """Generate responses for a list of prompts."""
    model.eval()
    device = next(model.parameters()).device
    
    responses = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating responses"):
        batch_prompts = prompts[i:i + batch_size]
        
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode, removing the prompt
        for j, output in enumerate(outputs):
            prompt_len = inputs["input_ids"][j].shape[0]
            response = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            responses.append(response.strip())
    
    return responses


def compute_distinct_n(texts: List[str], n: int = 2) -> float:
    """Compute Distinct-n metric (unique n-grams / total n-grams)."""
    all_ngrams = []
    
    for text in texts:
        tokens = text.lower().split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)
    
    if len(all_ngrams) == 0:
        return 0.0
    
    unique_ngrams = len(set(all_ngrams))
    return unique_ngrams / len(all_ngrams)


def compute_bleu(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute BLEU scores."""
    try:
        from sacrebleu.metrics import BLEU
        bleu = BLEU()
        result = bleu.corpus_score(predictions, [references])
        return {
            "bleu": result.score / 100,
            "bleu_1": result.precisions[0] / 100 if len(result.precisions) > 0 else 0,
            "bleu_2": result.precisions[1] / 100 if len(result.precisions) > 1 else 0,
            "bleu_4": result.precisions[3] / 100 if len(result.precisions) > 3 else 0,
        }
    except ImportError:
        logger.warning("sacrebleu not installed, skipping BLEU")
        return {}


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE scores."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        for pred, ref in zip(predictions, references):
            result = scorer.score(ref, pred)
            for key in scores:
                scores[key].append(result[key].fmeasure)
        
        return {
            "rouge1_f1": sum(scores['rouge1']) / len(scores['rouge1']),
            "rouge2_f1": sum(scores['rouge2']) / len(scores['rouge2']),
            "rougeL_f1": sum(scores['rougeL']) / len(scores['rougeL']),
        }
    except ImportError:
        logger.warning("rouge_score not installed, skipping ROUGE")
        return {}


def compute_reward_scores(
    reward_model_path: str, tokenizer, prompts: List[str], responses: List[str]
) -> Dict[str, float]:
    """Compute reward scores using a trained reward model."""
    try:
        from transformers import AutoModelForSequenceClassification

        logger.info(f"Loading reward model from {reward_model_path}")
        adapter_config_path = os.path.join(reward_model_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            from peft import PeftConfig, PeftModel
            peft_config = PeftConfig.from_pretrained(reward_model_path)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                peft_config.base_model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                num_labels=1,
            )
            reward_model = PeftModel.from_pretrained(base_model, reward_model_path)
            reward_model = reward_model.merge_and_unload()
        else:
            reward_model = AutoModelForSequenceClassification.from_pretrained(
                reward_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        reward_model.eval()
        
        rewards = []
        for prompt, response in tqdm(zip(prompts, responses), desc="Computing rewards", total=len(prompts)):
            text = f"{prompt}\n\n{response}"
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(reward_model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = reward_model(**inputs)
                reward = outputs.logits[0, 0].item()
            rewards.append(reward)
        
        import numpy as np
        return {
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "reward_min": float(np.min(rewards)),
            "reward_max": float(np.max(rewards)),
        }
    except Exception as e:
        logger.warning(f"Could not compute reward scores: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Evaluate SLM-RL-Agents models")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model to evaluate")
    parser.add_argument("--eval_dataset", type=str, required=True,
                        help="Path to evaluation dataset")
    parser.add_argument("--output_dir", type=str, default="./outputs/evaluation",
                        help="Directory to save evaluation results")
    parser.add_argument("--reward_model_path", type=str, default=None,
                        help="Path to reward model (optional)")
    parser.add_argument("--reference_model_path", type=str, default=None,
                        help="Path to reference model for comparison (optional)")
    parser.add_argument("--max_samples", type=int, default=500,
                        help="Maximum samples to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Generation temperature")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for evaluation")
    parser.add_argument("--metrics", type=str, default="all",
                        help="Metrics to compute (comma-separated or 'all')")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Required for decoder-only generation

    # Check if model is a PEFT adapter
    # Use float32 for small models — bfloat16 causes inf/nan with PPO-trained weights
    adapter_config_path = os.path.join(args.model_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        from peft import PeftConfig, PeftModel
        peft_config = PeftConfig.from_pretrained(args.model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, args.model_path)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

    # Verify model weights are healthy
    has_nan = any(
        torch.isnan(p).any() or torch.isinf(p).any()
        for p in model.parameters()
    )
    if has_nan:
        logger.warning("Model contains NaN/Inf weights — results may be degraded.")
    
    # Load evaluation data
    dataset = load_eval_data(args.eval_dataset, args.max_samples)
    
    # Extract prompts and references
    if "text" in dataset.column_names:
        # SFT format - extract prompts
        texts = dataset["text"]
        prompts = []
        references = []
        for text in texts:
            if "User:" in text and "Assistant:" in text:
                parts = text.split("Assistant:")
                prompt = parts[0].replace("User:", "").strip()
                reference = parts[1].strip() if len(parts) > 1 else ""
                prompts.append(prompt)
                references.append(reference)
            else:
                prompts.append(text[:200])
                references.append(text[200:])
    elif "prompt" in dataset.column_names:
        prompts = dataset["prompt"]
        references = dataset.get("chosen", dataset.get("response", [""] * len(prompts)))
    else:
        raise ValueError("Dataset format not recognized")
    
    # Initialize results
    results = {
        "model_path": args.model_path,
        "num_samples": len(prompts),
        "metrics": {},
    }
    
    # Determine which metrics to compute
    if args.metrics == "all":
        metrics_to_compute = ["perplexity", "generation", "diversity", "bleu", "rouge", "reward"]
    else:
        metrics_to_compute = args.metrics.split(",")
    
    # Compute perplexity
    if "perplexity" in metrics_to_compute:
        logger.info("Computing perplexity...")
        texts_for_ppl = [f"{p}\n\n{r}" for p, r in zip(prompts, references) if r]
        if texts_for_ppl:
            ppl = compute_perplexity(model, tokenizer, texts_for_ppl, args.batch_size)
            results["metrics"]["perplexity"] = ppl
            logger.info(f"Perplexity: {ppl:.2f}")
    
    # Generate responses
    generated_responses = None
    if any(m in metrics_to_compute for m in ["generation", "diversity", "bleu", "rouge", "reward"]):
        logger.info("Generating responses...")
        generated_responses = generate_responses(
            model, tokenizer, prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
        )
    
    # Compute diversity metrics
    if "diversity" in metrics_to_compute and generated_responses:
        logger.info("Computing diversity metrics...")
        results["metrics"]["distinct_1"] = compute_distinct_n(generated_responses, 1)
        results["metrics"]["distinct_2"] = compute_distinct_n(generated_responses, 2)
        results["metrics"]["distinct_3"] = compute_distinct_n(generated_responses, 3)
        
        # Average response length
        avg_length = sum(len(r.split()) for r in generated_responses) / len(generated_responses)
        results["metrics"]["avg_response_length"] = avg_length
        
        logger.info(f"Distinct-1: {results['metrics']['distinct_1']:.4f}")
        logger.info(f"Distinct-2: {results['metrics']['distinct_2']:.4f}")
    
    # Compute BLEU (if references available)
    if "bleu" in metrics_to_compute and generated_responses and references and references[0]:
        logger.info("Computing BLEU scores...")
        bleu_results = compute_bleu(generated_responses, references)
        results["metrics"].update(bleu_results)
        if "bleu" in bleu_results:
            logger.info(f"BLEU: {bleu_results['bleu']:.4f}")
    
    # Compute ROUGE
    if "rouge" in metrics_to_compute and generated_responses and references and references[0]:
        logger.info("Computing ROUGE scores...")
        rouge_results = compute_rouge(generated_responses, references)
        results["metrics"].update(rouge_results)
        if "rougeL_f1" in rouge_results:
            logger.info(f"ROUGE-L F1: {rouge_results['rougeL_f1']:.4f}")
    
    # Compute reward scores
    if "reward" in metrics_to_compute and args.reward_model_path and generated_responses:
        logger.info("Computing reward scores...")
        reward_results = compute_reward_scores(
            args.reward_model_path, tokenizer, prompts, generated_responses
        )
        results["metrics"].update(reward_results)
        if "reward_mean" in reward_results:
            logger.info(f"Mean Reward: {reward_results['reward_mean']:.4f}")
    
    # Save results
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")
    
    # Save sample generations
    if generated_responses:
        samples_path = os.path.join(args.output_dir, "sample_generations.json")
        samples = [
            {"prompt": p, "generated": g, "reference": r}
            for p, g, r in zip(prompts[:50], generated_responses[:50], references[:50])
        ]
        with open(samples_path, "w") as f:
            json.dump(samples, f, indent=2)
        logger.info(f"Saved sample generations to {samples_path}")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    for metric, value in results["metrics"].items():
        logger.info(f"{metric}: {value:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
