"""
Evaluation Metrics for SLM-RL-Agents

This module provides comprehensive evaluation tools for measuring model quality
across all stages of RLHF training. Metrics include:

Language Modeling:
    - Perplexity: Measures prediction uncertainty
    - Cross-entropy loss: Training loss on held-out data

Generation Quality:
    - BLEU: N-gram precision for translation/generation
    - ROUGE: N-gram recall for summarization
    - BERTScore: Semantic similarity using embeddings
    - METEOR: Alignment-based metric

RLHF-Specific:
    - Reward score: Average reward from reward model
    - KL divergence: Policy drift from reference
    - Win rate: Preference comparison results

Diversity:
    - Distinct-n: Unique n-gram ratios
    - Self-BLEU: Inter-generation similarity

References:
    - Papineni et al. (2002): BLEU
    - Lin (2004): ROUGE
    - Zhang et al. (2020): BERTScore
"""

from src.evaluation.metrics import (
    compute_perplexity,
    compute_bleu,
    compute_rouge,
    compute_bertscore,
    compute_distinct_n,
    compute_reward_metrics,
    compute_kl_divergence,
    EvaluationSuite,
)
from src.evaluation.benchmark import run_benchmarks, BenchmarkRunner

__all__ = [
    "compute_perplexity",
    "compute_bleu",
    "compute_rouge",
    "compute_bertscore",
    "compute_distinct_n",
    "compute_reward_metrics",
    "compute_kl_divergence",
    "EvaluationSuite",
    "run_benchmarks",
    "BenchmarkRunner",
]
