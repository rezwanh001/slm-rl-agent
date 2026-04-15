#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque
"""

"""
Comprehensive Evaluation Metrics for SLM-RL-Agent

This module implements a complete suite of evaluation metrics for measuring the quality
of small language models throughout the RLHF training process. Understanding these metrics
is crucial for diagnosing training issues and comparing model variants.

The metrics are organized into categories based on what aspect of model quality they measure:

1. LANGUAGE MODELING METRICS
   These measure how well the model predicts text, which is the fundamental capability
   of language models. Perplexity is the primary metric here.

2. GENERATION QUALITY METRICS
   These compare generated text against reference outputs. They're useful for tasks
   where there are "correct" answers (summarization, translation, QA).

3. RLHF-SPECIFIC METRICS
   These track alignment-specific quantities like reward scores and policy drift,
   which are essential for monitoring PPO/DPO training.

4. DIVERSITY METRICS
   These ensure the model maintains output variety and doesn't collapse to
   repetitive or degenerate outputs (a common failure mode in RLHF).

Each metric includes references to the original papers and practical guidance on
interpretation and typical value ranges.
"""

import logging
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


# =============================================================================
# LANGUAGE MODELING METRICS
# =============================================================================

def compute_perplexity(
    model: Any,
    tokenizer: Any,
    texts: List[str],
    batch_size: int = 8,
    max_length: int = 1024,
    stride: int = 512,
) -> Dict[str, float]:
    """
    Compute perplexity of a language model on a set of texts.
    
    WHAT IS PERPLEXITY?
    Perplexity measures how "surprised" the model is by the text. Mathematically,
    it's the exponentiated average cross-entropy loss:
    
        PPL = exp(avg_loss) = exp(-1/N * sum(log P(token_i | context)))
    
    INTERPRETATION:
    - Lower perplexity = better model (less surprised by text)
    - A perplexity of K means the model is as uncertain as if it had K equally
      likely choices for each token
    - Typical ranges:
      * Well-trained LLMs on in-domain data: 10-30
      * Small models or out-of-domain: 50-200
      * Random model (untrained): ~vocabulary_size
    
    WHY STRIDE?
    For texts longer than max_length, we use a sliding window approach with stride.
    This ensures all tokens are scored while maintaining reasonable context windows.
    The stride parameter controls overlap - larger stride is faster but may miss
    some context dependencies.
    
    Args:
        model: Language model (HuggingFace format)
        tokenizer: Tokenizer for the model
        texts: List of texts to evaluate
        batch_size: Batch size for processing
        max_length: Maximum sequence length for the model
        stride: Stride for sliding window (smaller = more overlap = better but slower)
    
    Returns:
        Dictionary with:
        - 'perplexity': Overall perplexity across all texts
        - 'loss': Average cross-entropy loss
        - 'num_tokens': Total tokens evaluated
    
    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> results = compute_perplexity(model, tokenizer, ["Hello world!"])
        >>> print(f"Perplexity: {results['perplexity']:.2f}")
    """
    model.eval()
    device = next(model.parameters()).device
    
    total_loss = 0.0
    total_tokens = 0
    
    # Process each text with sliding window for long sequences
    for text in tqdm(texts, desc="Computing perplexity"):
        # Tokenize the text
        encodings = tokenizer(
            text,
            return_tensors="pt",
            truncation=False,  # We'll handle truncation with sliding window
            return_attention_mask=True,
        )
        
        seq_len = encodings.input_ids.size(1)
        
        # Handle sequences that fit within max_length
        if seq_len <= max_length:
            input_ids = encodings.input_ids.to(device)
            attention_mask = encodings.attention_mask.to(device)
            
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                # outputs.loss is already the mean loss over all tokens
                total_loss += outputs.loss.item() * (seq_len - 1)  # -1 because we don't predict first token
                total_tokens += seq_len - 1
        else:
            # Sliding window for long sequences
            prev_end_loc = 0
            for begin_loc in range(0, seq_len, stride):
                end_loc = min(begin_loc + max_length, seq_len)
                
                # Extract window
                input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
                
                # Create target labels
                # We only want to compute loss on new tokens (not seen in previous window)
                target_len = end_loc - prev_end_loc
                labels = input_ids.clone()
                labels[:, :-target_len] = -100  # Mask tokens we've already scored
                
                with torch.no_grad():
                    outputs = model(input_ids, labels=labels)
                    # Count only the non-masked tokens
                    num_valid = (labels != -100).sum().item()
                    total_loss += outputs.loss.item() * num_valid
                    total_tokens += num_valid
                
                prev_end_loc = end_loc
                
                if end_loc >= seq_len:
                    break
    
    # Compute final metrics
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')  # Avoid overflow
    
    return {
        "perplexity": perplexity,
        "loss": avg_loss,
        "num_tokens": total_tokens,
    }


# =============================================================================
# GENERATION QUALITY METRICS
# =============================================================================

def compute_bleu(
    predictions: List[str],
    references: List[str],
    max_n: int = 4,
    smooth: bool = True,
) -> Dict[str, float]:
    """
    Compute BLEU (Bilingual Evaluation Understudy) score.
    
    WHAT IS BLEU?
    BLEU measures how many n-grams in the generated text match the reference text.
    It was originally designed for machine translation but is widely used for
    any text generation task with reference outputs.
    
    THE BLEU FORMULA:
    BLEU = BP × exp(sum(w_n × log(p_n)))
    
    Where:
    - BP = brevity penalty (penalizes short outputs)
    - p_n = precision of n-grams (what fraction of generated n-grams appear in reference)
    - w_n = weights (typically uniform: 1/n for BLEU-n)
    
    INTERPRETATION:
    - Range: 0 to 1 (often reported as 0-100)
    - Higher is better
    - BLEU-4 is most commonly reported
    - Typical ranges:
      * Excellent: > 0.40
      * Good: 0.30-0.40
      * Moderate: 0.20-0.30
      * Poor: < 0.20
    
    LIMITATIONS:
    - Penalizes valid paraphrases (synonyms don't match)
    - Sensitive to tokenization/normalization
    - Doesn't capture semantic meaning, only surface overlap
    
    Args:
        predictions: List of generated texts
        references: List of reference texts
        max_n: Maximum n-gram size (4 for BLEU-4)
        smooth: Use smoothing for short sentences
    
    Returns:
        Dictionary with BLEU-1 through BLEU-n scores
    """
    try:
        from sacrebleu.metrics import BLEU
        bleu = BLEU(max_ngram_order=max_n, smooth_method="exp" if smooth else "none")
        
        # sacrebleu expects references as list of lists (multiple references per prediction)
        refs = [[ref] for ref in references]
        
        result = bleu.corpus_score(predictions, [references])
        
        return {
            "bleu": result.score / 100,  # Convert to 0-1 scale
            "bleu_1": result.precisions[0] / 100 if len(result.precisions) > 0 else 0,
            "bleu_2": result.precisions[1] / 100 if len(result.precisions) > 1 else 0,
            "bleu_3": result.precisions[2] / 100 if len(result.precisions) > 2 else 0,
            "bleu_4": result.precisions[3] / 100 if len(result.precisions) > 3 else 0,
            "brevity_penalty": result.bp,
        }
    except ImportError:
        logger.warning("sacrebleu not installed, using basic BLEU computation")
        return _compute_bleu_basic(predictions, references, max_n)


def _compute_bleu_basic(
    predictions: List[str],
    references: List[str],
    max_n: int = 4,
) -> Dict[str, float]:
    """Basic BLEU implementation when sacrebleu is not available."""
    
    def get_ngrams(text: str, n: int) -> Counter:
        tokens = text.lower().split()
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    precisions = []
    for n in range(1, max_n + 1):
        total_matches = 0
        total_count = 0
        
        for pred, ref in zip(predictions, references):
            pred_ngrams = get_ngrams(pred, n)
            ref_ngrams = get_ngrams(ref, n)
            
            # Count matches (clipped by reference count)
            for ngram, count in pred_ngrams.items():
                total_matches += min(count, ref_ngrams.get(ngram, 0))
            total_count += sum(pred_ngrams.values())
        
        precision = total_matches / total_count if total_count > 0 else 0
        precisions.append(precision)
    
    # Geometric mean of precisions
    if all(p > 0 for p in precisions):
        log_bleu = sum(math.log(p) for p in precisions) / len(precisions)
        bleu = math.exp(log_bleu)
    else:
        bleu = 0.0
    
    return {
        "bleu": bleu,
        "bleu_1": precisions[0] if len(precisions) > 0 else 0,
        "bleu_2": precisions[1] if len(precisions) > 1 else 0,
        "bleu_3": precisions[2] if len(precisions) > 2 else 0,
        "bleu_4": precisions[3] if len(precisions) > 3 else 0,
    }


def compute_rouge(
    predictions: List[str],
    references: List[str],
    rouge_types: List[str] = ["rouge1", "rouge2", "rougeL"],
) -> Dict[str, float]:
    """
    Compute ROUGE (Recall-Oriented Understudy for Gisting Evaluation) scores.
    
    WHAT IS ROUGE?
    ROUGE measures recall: what fraction of the reference text appears in the generated text.
    This makes it complementary to BLEU (which measures precision). ROUGE was designed
    for summarization, where we want to ensure important content is included.
    
    ROUGE VARIANTS:
    - ROUGE-1: Unigram overlap (individual words)
    - ROUGE-2: Bigram overlap (two-word phrases)
    - ROUGE-L: Longest Common Subsequence (captures sentence structure)
    - ROUGE-Lsum: LCS for multi-sentence summaries
    
    INTERPRETATION:
    - Range: 0 to 1
    - Higher is better
    - F1 score (harmonic mean of precision and recall) is typically reported
    - Typical ranges depend on task:
      * Summarization: ROUGE-L F1 of 0.30-0.50 is good
      * ROUGE-2 is harder (requires exact bigram matches)
    
    Args:
        predictions: List of generated texts
        references: List of reference texts
        rouge_types: Which ROUGE variants to compute
    
    Returns:
        Dictionary with ROUGE scores (precision, recall, F1 for each type)
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
        
        results = {rt: {"precision": [], "recall": [], "fmeasure": []} for rt in rouge_types}
        
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            for rt in rouge_types:
                results[rt]["precision"].append(scores[rt].precision)
                results[rt]["recall"].append(scores[rt].recall)
                results[rt]["fmeasure"].append(scores[rt].fmeasure)
        
        # Average across all examples
        final_results = {}
        for rt in rouge_types:
            final_results[f"{rt}_precision"] = np.mean(results[rt]["precision"])
            final_results[f"{rt}_recall"] = np.mean(results[rt]["recall"])
            final_results[f"{rt}_f1"] = np.mean(results[rt]["fmeasure"])
        
        return final_results
        
    except ImportError:
        logger.warning("rouge_score not installed, returning empty ROUGE scores")
        return {f"{rt}_{m}": 0.0 for rt in rouge_types for m in ["precision", "recall", "f1"]}


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    lang: str = "en",
    rescale_with_baseline: bool = True,
) -> Dict[str, float]:
    """
    Compute BERTScore for semantic similarity evaluation.
    
    WHAT IS BERTSCORE?
    BERTScore uses contextual embeddings from BERT-like models to compute similarity
    between generated and reference texts. Unlike BLEU/ROUGE which require exact
    token matches, BERTScore can recognize synonyms and paraphrases.
    
    HOW IT WORKS:
    1. Encode both texts using a pretrained transformer (like BERT or DeBERTa)
    2. Compute cosine similarity between all token pairs
    3. Find optimal matching using greedy alignment
    4. Aggregate into precision, recall, and F1
    
    INTERPRETATION:
    - Range: typically 0.5-1.0 (before rescaling) or 0-1 (after rescaling)
    - Higher is better
    - Rescaling with baseline normalizes against random text pairs
    - F1 is the primary metric
    - Correlates better with human judgments than BLEU/ROUGE
    
    Args:
        predictions: List of generated texts
        references: List of reference texts  
        model_type: Which transformer model to use for embeddings
        lang: Language code
        rescale_with_baseline: Whether to rescale scores
    
    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    try:
        from bert_score import score
        
        P, R, F1 = score(
            predictions,
            references,
            model_type=model_type,
            lang=lang,
            rescale_with_baseline=rescale_with_baseline,
            verbose=False,
        )
        
        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item(),
        }
        
    except ImportError:
        logger.warning("bert_score not installed, returning empty BERTScore")
        return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}


# =============================================================================
# RLHF-SPECIFIC METRICS
# =============================================================================

def compute_reward_metrics(
    reward_model: Any,
    prompts: List[str],
    responses: List[str],
    batch_size: int = 16,
) -> Dict[str, float]:
    """
    Compute reward-based metrics using a trained reward model.
    
    WHAT ARE REWARD METRICS?
    In RLHF, the reward model provides the training signal for policy optimization.
    Monitoring reward scores helps detect:
    - Reward hacking (score increases but quality decreases)
    - Training instability
    - Overfitting to reward model quirks
    
    KEY METRICS:
    - Mean reward: Average reward across all examples
    - Reward std: Variance in rewards (too low might indicate collapse)
    - Min/Max reward: Detect outliers and degenerate outputs
    
    INTERPRETATION:
    - Reward should increase during PPO training
    - But rapid/extreme increases may indicate reward hacking
    - Compare against baseline (SFT model) rewards
    
    Args:
        reward_model: Trained reward model
        prompts: List of prompts
        responses: List of generated responses
        batch_size: Batch size for scoring
    
    Returns:
        Dictionary with reward statistics
    """
    all_rewards = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_responses = responses[i:i + batch_size]
        
        # Score the batch
        rewards = reward_model.score_batch(batch_prompts, batch_responses)
        all_rewards.extend(rewards)
    
    rewards_array = np.array(all_rewards)
    
    return {
        "reward_mean": float(np.mean(rewards_array)),
        "reward_std": float(np.std(rewards_array)),
        "reward_min": float(np.min(rewards_array)),
        "reward_max": float(np.max(rewards_array)),
        "reward_median": float(np.median(rewards_array)),
    }


def compute_kl_divergence(
    policy_model: Any,
    reference_model: Any,
    tokenizer: Any,
    prompts: List[str],
    responses: List[str],
    batch_size: int = 8,
) -> Dict[str, float]:
    """
    Compute KL divergence between policy and reference model.
    
    WHAT IS KL DIVERGENCE IN RLHF?
    KL divergence measures how much the policy model's distribution has drifted
    from the reference (usually SFT) model. In RLHF, we optimize:
    
        J(θ) = E[r(x,y)] - β × KL(π_θ || π_ref)
    
    The KL penalty prevents the policy from deviating too far from the reference,
    which helps maintain coherent language and prevents reward hacking.
    
    INTERPRETATION:
    - KL = 0 means policy equals reference (no learning)
    - Small KL (0.1-0.5) is typical for well-tuned RLHF
    - Large KL (>1.0) may indicate:
      * The policy is drifting too far
      * Risk of reward hacking
      * May need to increase β (KL penalty)
    
    HOW IT'S COMPUTED:
    For each token position:
        KL = sum(π_θ(a) × log(π_θ(a) / π_ref(a)))
    
    Then averaged across all positions.
    
    Args:
        policy_model: The trained policy model
        reference_model: The reference model (usually SFT checkpoint)
        tokenizer: Tokenizer for both models
        prompts: List of prompts
        responses: List of responses
        batch_size: Batch size for computation
    
    Returns:
        Dictionary with KL divergence statistics
    """
    policy_model.eval()
    reference_model.eval()
    
    device = next(policy_model.parameters()).device
    
    all_kls = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_responses = responses[i:i + batch_size]
        
        # Tokenize prompt + response
        texts = [p + " " + r for p, r in zip(batch_prompts, batch_responses)]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get logits from both models
            policy_outputs = policy_model(**inputs)
            ref_outputs = reference_model(**inputs)
            
            # Convert to log probabilities
            policy_logprobs = F.log_softmax(policy_outputs.logits, dim=-1)
            ref_logprobs = F.log_softmax(ref_outputs.logits, dim=-1)
            
            # Compute KL divergence per token
            # KL(P||Q) = sum(P * (log P - log Q))
            policy_probs = policy_logprobs.exp()
            kl = (policy_probs * (policy_logprobs - ref_logprobs)).sum(dim=-1)
            
            # Mask padding tokens
            mask = inputs["attention_mask"]
            kl = (kl * mask).sum(dim=1) / mask.sum(dim=1)
            
            all_kls.extend(kl.cpu().tolist())
    
    kl_array = np.array(all_kls)
    
    return {
        "kl_divergence_mean": float(np.mean(kl_array)),
        "kl_divergence_std": float(np.std(kl_array)),
        "kl_divergence_max": float(np.max(kl_array)),
    }


# =============================================================================
# DIVERSITY METRICS
# =============================================================================

def compute_distinct_n(
    texts: List[str],
    n_values: List[int] = [1, 2, 3],
) -> Dict[str, float]:
    """
    Compute Distinct-n metrics for output diversity.
    
    WHAT IS DISTINCT-N?
    Distinct-n measures the ratio of unique n-grams to total n-grams in generated text.
    It helps detect mode collapse, where the model produces repetitive or templated outputs.
    
    FORMULA:
        Distinct-n = |unique n-grams| / |total n-grams|
    
    INTERPRETATION:
    - Range: 0 to 1
    - Higher is better (more diverse)
    - Typical values:
      * Distinct-1: 0.1-0.4 (many repeated common words is normal)
      * Distinct-2: 0.3-0.7 
      * Distinct-3: 0.5-0.9
    - Very low values indicate repetitive outputs
    - Compare across training checkpoints to detect degradation
    
    WHY IT MATTERS IN RLHF:
    Reward hacking often manifests as decreased diversity - the model finds
    a specific pattern that scores high and repeats it. Monitoring Distinct-n
    helps catch this early.
    
    Args:
        texts: List of generated texts to analyze
        n_values: Which n-gram sizes to compute (typically 1, 2, 3)
    
    Returns:
        Dictionary with Distinct-n values for each n
    """
    results = {}
    
    for n in n_values:
        all_ngrams = []
        
        for text in texts:
            tokens = text.lower().split()
            ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
            all_ngrams.extend(ngrams)
        
        if len(all_ngrams) > 0:
            unique_ngrams = len(set(all_ngrams))
            total_ngrams = len(all_ngrams)
            distinct = unique_ngrams / total_ngrams
        else:
            distinct = 0.0
        
        results[f"distinct_{n}"] = distinct
    
    return results


def compute_self_bleu(
    texts: List[str],
    sample_size: int = 100,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Compute Self-BLEU for measuring output diversity.
    
    WHAT IS SELF-BLEU?
    Self-BLEU measures the BLEU score of each generated text against all other
    generated texts. High Self-BLEU means generations are similar to each other
    (low diversity), while low Self-BLEU indicates diverse outputs.
    
    INTERPRETATION:
    - Range: 0 to 1
    - LOWER is better (more diverse)
    - Typical values:
      * Diverse generation: 0.2-0.4
      * Moderate repetition: 0.4-0.6
      * Highly repetitive: 0.6+
    
    This metric complements Distinct-n by capturing longer-range patterns
    and structural similarity, not just n-gram repetition.
    
    Args:
        texts: List of generated texts
        sample_size: Number of texts to use (for efficiency)
        seed: Random seed for sampling
    
    Returns:
        Dictionary with Self-BLEU score
    """
    import random
    random.seed(seed)
    
    # Sample if too many texts
    if len(texts) > sample_size:
        texts = random.sample(texts, sample_size)
    
    if len(texts) < 2:
        return {"self_bleu": 0.0}
    
    bleu_scores = []
    
    for i, text in enumerate(texts):
        # Use all other texts as references
        references = texts[:i] + texts[i+1:]
        
        # Compute BLEU against each reference and average
        text_scores = []
        for ref in references[:10]:  # Limit references for speed
            score = compute_bleu([text], [ref])
            text_scores.append(score.get("bleu", 0))
        
        if text_scores:
            bleu_scores.append(np.mean(text_scores))
    
    return {"self_bleu": float(np.mean(bleu_scores)) if bleu_scores else 0.0}


# =============================================================================
# EVALUATION SUITE
# =============================================================================

class EvaluationSuite:
    """
    Comprehensive evaluation suite that runs all metrics.
    
    This class provides a unified interface for evaluating language models across
    all relevant metrics. It handles the complexity of running different metrics
    with different requirements and aggregates results into a single report.
    
    Example:
        >>> suite = EvaluationSuite(model, tokenizer, reward_model)
        >>> results = suite.evaluate(
        ...     prompts=prompts,
        ...     predictions=generated_texts,
        ...     references=reference_texts,
        ... )
        >>> print(results)
    """
    
    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        reward_model: Any = None,
        reference_model: Any = None,
    ):
        """
        Initialize the evaluation suite.
        
        Args:
            model: The model to evaluate (for perplexity)
            tokenizer: Tokenizer for the model
            reward_model: Trained reward model (optional)
            reference_model: Reference model for KL divergence (optional)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.reference_model = reference_model
    
    def evaluate(
        self,
        prompts: List[str],
        predictions: List[str],
        references: Optional[List[str]] = None,
        compute_perplexity_flag: bool = True,
        compute_generation_metrics: bool = True,
        compute_reward_flag: bool = True,
        compute_diversity: bool = True,
        compute_kl: bool = True,
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation.
        
        Args:
            prompts: List of input prompts
            predictions: List of generated responses
            references: Optional reference responses (for BLEU/ROUGE/BERTScore)
            compute_perplexity_flag: Whether to compute perplexity
            compute_generation_metrics: Whether to compute BLEU/ROUGE/BERTScore
            compute_reward_flag: Whether to compute reward metrics
            compute_diversity: Whether to compute diversity metrics
            compute_kl: Whether to compute KL divergence
        
        Returns:
            Dictionary with all computed metrics
        """
        results = {}
        
        # Perplexity (requires model)
        if compute_perplexity_flag and self.model is not None:
            logger.info("Computing perplexity...")
            texts = [f"{p} {r}" for p, r in zip(prompts, predictions)]
            ppl_results = compute_perplexity(self.model, self.tokenizer, texts)
            results.update(ppl_results)
        
        # Generation quality (requires references)
        if compute_generation_metrics and references is not None:
            logger.info("Computing generation quality metrics...")
            
            bleu_results = compute_bleu(predictions, references)
            results.update(bleu_results)
            
            rouge_results = compute_rouge(predictions, references)
            results.update(rouge_results)
            
            bert_results = compute_bertscore(predictions, references)
            results.update(bert_results)
        
        # Reward metrics (requires reward model)
        if compute_reward_flag and self.reward_model is not None:
            logger.info("Computing reward metrics...")
            reward_results = compute_reward_metrics(
                self.reward_model, prompts, predictions
            )
            results.update(reward_results)
        
        # Diversity metrics
        if compute_diversity:
            logger.info("Computing diversity metrics...")
            distinct_results = compute_distinct_n(predictions)
            results.update(distinct_results)
            
            self_bleu_results = compute_self_bleu(predictions)
            results.update(self_bleu_results)
        
        # KL divergence (requires reference model)
        if compute_kl and self.reference_model is not None and self.model is not None:
            logger.info("Computing KL divergence...")
            kl_results = compute_kl_divergence(
                self.model, self.reference_model, self.tokenizer,
                prompts, predictions
            )
            results.update(kl_results)
        
        return results
    
    def generate_report(
        self,
        results: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate a human-readable evaluation report.
        
        Args:
            results: Results dictionary from evaluate()
            output_path: Optional path to save the report
        
        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "EVALUATION REPORT",
            "=" * 60,
            "",
        ]
        
        # Group metrics by category
        categories = {
            "Language Modeling": ["perplexity", "loss", "num_tokens"],
            "Generation Quality (BLEU)": ["bleu", "bleu_1", "bleu_2", "bleu_3", "bleu_4"],
            "Generation Quality (ROUGE)": [k for k in results if k.startswith("rouge")],
            "Generation Quality (BERTScore)": [k for k in results if k.startswith("bertscore")],
            "Reward Metrics": [k for k in results if k.startswith("reward")],
            "Diversity": ["distinct_1", "distinct_2", "distinct_3", "self_bleu"],
            "KL Divergence": [k for k in results if k.startswith("kl")],
        }
        
        for category, keys in categories.items():
            category_results = {k: results.get(k) for k in keys if k in results}
            if category_results:
                lines.append(f"\n{category}")
                lines.append("-" * 40)
                for key, value in category_results.items():
                    if isinstance(value, float):
                        lines.append(f"  {key}: {value:.4f}")
                    else:
                        lines.append(f"  {key}: {value}")
        
        lines.append("\n" + "=" * 60)
        
        report = "\n".join(lines)
        
        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            logger.info(f"Saved evaluation report to {output_path}")
        
        return report
