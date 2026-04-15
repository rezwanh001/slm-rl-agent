#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque
"""

"""
Benchmark Evaluation for SLM-RL-Agent

This module provides tools for evaluating models on standard NLP benchmarks.
Benchmarks provide standardized comparison points across different models and
training configurations.

Supported Benchmarks:
    - HellaSwag: Commonsense reasoning (sentence completion)
    - ARC: Science reasoning (multiple choice questions)
    - TruthfulQA: Truthfulness evaluation
    - Custom benchmarks: Load your own evaluation datasets

These benchmarks help answer the question: "How does my trained model compare
to other models on established tasks?" They complement the RLHF-specific metrics
by measuring general language understanding capabilities.
"""

import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark evaluation."""
    name: str
    accuracy: float
    num_correct: int
    num_total: int
    per_category_accuracy: Optional[Dict[str, float]] = None
    predictions: Optional[List[Dict[str, Any]]] = None


class BenchmarkRunner:
    """
    Runs standard NLP benchmarks on a language model.
    
    This class provides a unified interface for evaluating models on various
    benchmarks. Each benchmark tests different capabilities:
    
    - HellaSwag: Tests commonsense reasoning through sentence completion.
      The model must choose the most plausible continuation of a scenario.
      Good performance indicates understanding of everyday situations.
    
    - ARC (AI2 Reasoning Challenge): Tests science knowledge through
      multiple-choice questions from grade-school science exams.
      ARC-Easy and ARC-Challenge test different difficulty levels.
    
    - TruthfulQA: Tests whether the model generates truthful answers
      rather than plausible-sounding but false information.
    
    Example:
        >>> runner = BenchmarkRunner(model, tokenizer)
        >>> results = runner.run_hellaswag(num_samples=1000)
        >>> print(f"HellaSwag accuracy: {results.accuracy:.2%}")
    """
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: Optional[str] = None,
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            model: Language model (HuggingFace format)
            tokenizer: Tokenizer for the model
            device: Device to run on (defaults to model's device)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        
        # Ensure model is in eval mode
        self.model.eval()
    
    def run_hellaswag(
        self,
        num_samples: Optional[int] = None,
        batch_size: int = 8,
    ) -> BenchmarkResult:
        """
        Evaluate on HellaSwag benchmark.
        
        HellaSwag (Harder Endings, Longer contexts, and Low-shot Activities for
        Situations With Adversarial Generations) tests commonsense reasoning
        by asking the model to complete sentences describing everyday situations.
        
        The benchmark is challenging because incorrect options were generated
        by a language model and filtered to be adversarially difficult.
        
        Format:
            Context: "A woman is sitting at a piano..."
            Options: [A, B, C, D] (one correct, three adversarial)
            Task: Choose the most plausible continuation
        
        Args:
            num_samples: Number of examples to evaluate (None = all)
            batch_size: Batch size for evaluation
        
        Returns:
            BenchmarkResult with accuracy and detailed predictions
        """
        logger.info("Loading HellaSwag dataset...")
        
        try:
            from datasets import load_dataset
            dataset = load_dataset("hellaswag", split="validation")
        except Exception as e:
            logger.error(f"Failed to load HellaSwag: {e}")
            return BenchmarkResult(
                name="hellaswag",
                accuracy=0.0,
                num_correct=0,
                num_total=0,
            )
        
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        correct = 0
        total = 0
        predictions = []
        
        for example in tqdm(dataset, desc="HellaSwag"):
            # Extract the context and endings
            ctx = example["ctx"]
            endings = example["endings"]
            label = int(example["label"])
            
            # Score each ending
            scores = []
            for ending in endings:
                full_text = ctx + " " + ending
                score = self._compute_sequence_score(full_text)
                scores.append(score)
            
            # Prediction is the ending with highest score
            pred = int(torch.tensor(scores).argmax().item())
            
            if pred == label:
                correct += 1
            total += 1
            
            predictions.append({
                "context": ctx,
                "endings": endings,
                "label": label,
                "prediction": pred,
                "scores": scores,
                "correct": pred == label,
            })
        
        accuracy = correct / total if total > 0 else 0.0
        
        return BenchmarkResult(
            name="hellaswag",
            accuracy=accuracy,
            num_correct=correct,
            num_total=total,
            predictions=predictions,
        )
    
    def run_arc(
        self,
        subset: str = "easy",
        num_samples: Optional[int] = None,
    ) -> BenchmarkResult:
        """
        Evaluate on ARC (AI2 Reasoning Challenge) benchmark.
        
        ARC contains science questions from grade-school exams. It tests
        whether the model has acquired factual knowledge and can reason
        about scientific concepts.
        
        Subsets:
            - "easy": Questions most models can answer correctly
            - "challenge": Questions that are difficult for retrieval methods
        
        Args:
            subset: "easy" or "challenge"
            num_samples: Number of examples to evaluate
        
        Returns:
            BenchmarkResult with accuracy breakdown by question type
        """
        logger.info(f"Loading ARC-{subset} dataset...")
        
        try:
            from datasets import load_dataset
            config = "ARC-Easy" if subset == "easy" else "ARC-Challenge"
            dataset = load_dataset("ai2_arc", config, split="validation")
        except Exception as e:
            logger.error(f"Failed to load ARC: {e}")
            return BenchmarkResult(
                name=f"arc_{subset}",
                accuracy=0.0,
                num_correct=0,
                num_total=0,
            )
        
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        correct = 0
        total = 0
        predictions = []
        
        for example in tqdm(dataset, desc=f"ARC-{subset}"):
            question = example["question"]
            choices = example["choices"]
            answer_key = example["answerKey"]
            
            # Map answer key (A, B, C, D or 1, 2, 3, 4) to index
            if answer_key.isdigit():
                label = int(answer_key) - 1
            else:
                label = ord(answer_key) - ord('A')
            
            # Score each choice
            scores = []
            for choice_text in choices["text"]:
                prompt = f"Question: {question}\nAnswer: {choice_text}"
                score = self._compute_sequence_score(prompt)
                scores.append(score)
            
            pred = int(torch.tensor(scores).argmax().item())
            
            if pred == label:
                correct += 1
            total += 1
            
            predictions.append({
                "question": question,
                "choices": choices["text"],
                "label": label,
                "prediction": pred,
                "correct": pred == label,
            })
        
        accuracy = correct / total if total > 0 else 0.0
        
        return BenchmarkResult(
            name=f"arc_{subset}",
            accuracy=accuracy,
            num_correct=correct,
            num_total=total,
            predictions=predictions,
        )
    
    def run_truthfulqa(
        self,
        num_samples: Optional[int] = None,
    ) -> BenchmarkResult:
        """
        Evaluate on TruthfulQA benchmark.
        
        TruthfulQA tests whether models generate truthful answers to questions
        where humans might be misled by common misconceptions or false beliefs.
        
        This is particularly relevant for RLHF-trained models, as we want to
        verify that alignment training improves truthfulness rather than just
        making the model sound more confident about incorrect information.
        
        Args:
            num_samples: Number of examples to evaluate
        
        Returns:
            BenchmarkResult with truthfulness metrics
        """
        logger.info("Loading TruthfulQA dataset...")
        
        try:
            from datasets import load_dataset
            dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
        except Exception as e:
            logger.error(f"Failed to load TruthfulQA: {e}")
            return BenchmarkResult(
                name="truthfulqa",
                accuracy=0.0,
                num_correct=0,
                num_total=0,
            )
        
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        correct = 0
        total = 0
        predictions = []
        
        for example in tqdm(dataset, desc="TruthfulQA"):
            question = example["question"]
            choices = example["mc1_targets"]["choices"]
            labels = example["mc1_targets"]["labels"]
            
            # Find the correct answer (label = 1)
            correct_idx = labels.index(1) if 1 in labels else 0
            
            # Score each choice
            scores = []
            for choice in choices:
                prompt = f"Q: {question}\nA: {choice}"
                score = self._compute_sequence_score(prompt)
                scores.append(score)
            
            pred = int(torch.tensor(scores).argmax().item())
            
            if pred == correct_idx:
                correct += 1
            total += 1
            
            predictions.append({
                "question": question,
                "choices": choices,
                "label": correct_idx,
                "prediction": pred,
                "correct": pred == correct_idx,
            })
        
        accuracy = correct / total if total > 0 else 0.0
        
        return BenchmarkResult(
            name="truthfulqa",
            accuracy=accuracy,
            num_correct=correct,
            num_total=total,
            predictions=predictions,
        )
    
    def _compute_sequence_score(self, text: str) -> float:
        """
        Compute the log probability of a text sequence.
        
        This is used for multiple-choice evaluation where we select
        the option with the highest likelihood under the model.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            # Return negative loss (higher is better)
            return -outputs.loss.item()
    
    def run_all(
        self,
        num_samples: Optional[int] = 500,
    ) -> Dict[str, BenchmarkResult]:
        """
        Run all available benchmarks.
        
        Args:
            num_samples: Number of samples per benchmark
        
        Returns:
            Dictionary mapping benchmark names to results
        """
        results = {}
        
        # HellaSwag
        results["hellaswag"] = self.run_hellaswag(num_samples)
        
        # ARC
        results["arc_easy"] = self.run_arc("easy", num_samples)
        results["arc_challenge"] = self.run_arc("challenge", num_samples)
        
        # TruthfulQA
        results["truthfulqa"] = self.run_truthfulqa(num_samples)
        
        return results
    
    def generate_report(
        self,
        results: Dict[str, BenchmarkResult],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate a summary report of benchmark results.
        
        Args:
            results: Dictionary of benchmark results
            output_path: Optional path to save the report
        
        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "BENCHMARK EVALUATION REPORT",
            "=" * 60,
            "",
        ]
        
        for name, result in results.items():
            lines.append(f"{result.name.upper()}")
            lines.append("-" * 40)
            lines.append(f"  Accuracy: {result.accuracy:.2%}")
            lines.append(f"  Correct: {result.num_correct}/{result.num_total}")
            lines.append("")
        
        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 40)
        avg_accuracy = sum(r.accuracy for r in results.values()) / len(results)
        lines.append(f"  Average Accuracy: {avg_accuracy:.2%}")
        
        lines.append("\n" + "=" * 60)
        
        report = "\n".join(lines)
        
        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            logger.info(f"Saved benchmark report to {output_path}")
        
        return report


def run_benchmarks(
    model: Any,
    tokenizer: Any,
    benchmarks: List[str] = ["hellaswag", "arc_easy"],
    num_samples: int = 500,
    output_dir: Optional[str] = None,
) -> Dict[str, BenchmarkResult]:
    """
    Convenience function to run benchmarks on a model.
    
    Args:
        model: Language model to evaluate
        tokenizer: Tokenizer for the model
        benchmarks: List of benchmark names to run
        num_samples: Number of samples per benchmark
        output_dir: Directory to save results
    
    Returns:
        Dictionary of benchmark results
    """
    runner = BenchmarkRunner(model, tokenizer)
    results = {}
    
    for benchmark in benchmarks:
        if benchmark == "hellaswag":
            results[benchmark] = runner.run_hellaswag(num_samples)
        elif benchmark == "arc_easy":
            results[benchmark] = runner.run_arc("easy", num_samples)
        elif benchmark == "arc_challenge":
            results[benchmark] = runner.run_arc("challenge", num_samples)
        elif benchmark == "truthfulqa":
            results[benchmark] = runner.run_truthfulqa(num_samples)
        else:
            logger.warning(f"Unknown benchmark: {benchmark}")
    
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save results as JSON
        results_dict = {
            name: {
                "accuracy": r.accuracy,
                "num_correct": r.num_correct,
                "num_total": r.num_total,
            }
            for name, r in results.items()
        }
        with open(Path(output_dir) / "benchmark_results.json", "w") as f:
            json.dump(results_dict, f, indent=2)
        
        # Save report
        runner.generate_report(results, str(Path(output_dir) / "benchmark_report.txt"))
    
    return results
