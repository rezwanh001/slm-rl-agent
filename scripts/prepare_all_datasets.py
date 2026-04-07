#!/usr/bin/env python3
"""
Dataset Preparation for SLM-RL-Agent Experiments

Prepares three datasets for the RLHF pipeline:
1. TinyStories - Short children's stories
2. CNN/DailyMail - News summarization
3. Wikitext-103 - Wikipedia language modeling

For each dataset, creates:
- SFT training data (text field)
- SFT evaluation data
- Preference pairs (prompt, chosen, rejected) for reward model & DPO
- Preference evaluation data

Usage:
    python scripts/prepare_all_datasets.py --output_dir ./data --max_samples 10000
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def prepare_tinystories(output_dir: Path, max_samples: int = 10000, eval_ratio: float = 0.05):
    """Prepare TinyStories dataset for SFT and preference training."""
    ds_dir = output_dir / "tinystories"
    ds_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    # Subsample
    indices = list(range(len(dataset)))
    random.seed(42)
    random.shuffle(indices)
    indices = indices[:max_samples]
    dataset = dataset.select(indices)

    # SFT data: use text field directly
    sft_data = []
    for ex in dataset:
        text = ex.get("text", "").strip()
        if len(text) > 50:
            sft_data.append({"text": text})

    # Split eval
    num_eval = min(int(len(sft_data) * eval_ratio), 500)
    random.shuffle(sft_data)
    eval_data = sft_data[:num_eval]
    train_data = sft_data[num_eval:]

    with open(ds_dir / "sft_train.json", "w") as f:
        json.dump(train_data, f)
    with open(ds_dir / "sft_eval.json", "w") as f:
        json.dump(eval_data, f)

    logger.info(f"TinyStories SFT: {len(train_data)} train, {len(eval_data)} eval")

    # Preference data: use pairs of stories as chosen/rejected
    # We rank by length and coherence heuristic (longer, complete stories = better)
    pref_data = create_preference_pairs(sft_data, dataset_type="tinystories")

    num_pref_eval = min(int(len(pref_data) * eval_ratio), 300)
    random.shuffle(pref_data)
    pref_eval = pref_data[:num_pref_eval]
    pref_train = pref_data[num_pref_eval:]

    with open(ds_dir / "preference_train.json", "w") as f:
        json.dump(pref_train, f)
    with open(ds_dir / "preference_eval.json", "w") as f:
        json.dump(pref_eval, f)

    logger.info(f"TinyStories preferences: {len(pref_train)} train, {len(pref_eval)} eval")
    return ds_dir


def prepare_cnn_dailymail(output_dir: Path, max_samples: int = 10000, eval_ratio: float = 0.05):
    """Prepare CNN/DailyMail dataset for SFT and preference training."""
    ds_dir = output_dir / "cnn_dailymail"
    ds_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading CNN/DailyMail dataset...")
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")

    indices = list(range(len(dataset)))
    random.seed(42)
    random.shuffle(indices)
    indices = indices[:max_samples]
    dataset = dataset.select(indices)

    # SFT data: article + summary as text
    sft_data = []
    for ex in dataset:
        article = ex.get("article", "").strip()
        highlights = ex.get("highlights", "").strip()
        if len(article) > 100 and len(highlights) > 20:
            text = f"Article: {article[:1500]}\n\nSummary: {highlights}"
            sft_data.append({"text": text, "prompt": f"Summarize: {article[:800]}", "reference": highlights})

    num_eval = min(int(len(sft_data) * eval_ratio), 500)
    random.shuffle(sft_data)
    eval_data = sft_data[:num_eval]
    train_data = sft_data[num_eval:]

    with open(ds_dir / "sft_train.json", "w") as f:
        json.dump(train_data, f)
    with open(ds_dir / "sft_eval.json", "w") as f:
        json.dump(eval_data, f)

    logger.info(f"CNN/DailyMail SFT: {len(train_data)} train, {len(eval_data)} eval")

    # Preference data
    pref_data = create_preference_pairs(sft_data, dataset_type="cnn_dailymail")

    num_pref_eval = min(int(len(pref_data) * eval_ratio), 300)
    random.shuffle(pref_data)
    pref_eval = pref_data[:num_pref_eval]
    pref_train = pref_data[num_pref_eval:]

    with open(ds_dir / "preference_train.json", "w") as f:
        json.dump(pref_train, f)
    with open(ds_dir / "preference_eval.json", "w") as f:
        json.dump(pref_eval, f)

    logger.info(f"CNN/DailyMail preferences: {len(pref_train)} train, {len(pref_eval)} eval")
    return ds_dir


def prepare_wikitext(output_dir: Path, max_samples: int = 10000, eval_ratio: float = 0.05):
    """Prepare Wikitext-103 dataset for SFT and preference training."""
    ds_dir = output_dir / "wikitext"
    ds_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Wikitext-103 dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    # Filter non-empty paragraphs
    paragraphs = []
    for ex in dataset:
        text = ex.get("text", "").strip()
        if len(text) > 100 and not text.startswith("="):
            paragraphs.append(text)

    random.seed(42)
    random.shuffle(paragraphs)
    paragraphs = paragraphs[:max_samples]

    sft_data = [{"text": p} for p in paragraphs]

    num_eval = min(int(len(sft_data) * eval_ratio), 500)
    eval_data = sft_data[:num_eval]
    train_data = sft_data[num_eval:]

    with open(ds_dir / "sft_train.json", "w") as f:
        json.dump(train_data, f)
    with open(ds_dir / "sft_eval.json", "w") as f:
        json.dump(eval_data, f)

    logger.info(f"Wikitext SFT: {len(train_data)} train, {len(eval_data)} eval")

    # Preference data
    pref_data = create_preference_pairs(sft_data, dataset_type="wikitext")

    num_pref_eval = min(int(len(pref_data) * eval_ratio), 300)
    random.shuffle(pref_data)
    pref_eval = pref_data[:num_pref_eval]
    pref_train = pref_data[num_pref_eval:]

    with open(ds_dir / "preference_train.json", "w") as f:
        json.dump(pref_train, f)
    with open(ds_dir / "preference_eval.json", "w") as f:
        json.dump(pref_eval, f)

    logger.info(f"Wikitext preferences: {len(pref_train)} train, {len(pref_eval)} eval")
    return ds_dir


def create_preference_pairs(sft_data: list, dataset_type: str) -> list:
    """
    Create preference pairs from SFT data.

    Strategy: For each sample, create a "chosen" (high quality) and "rejected" (degraded) version.
    The rejected version is created by truncating, shuffling sentences, or adding noise.
    """
    pref_pairs = []
    random.seed(42)

    for i in range(0, len(sft_data) - 1, 2):
        text_a = sft_data[i]["text"]
        text_b = sft_data[i + 1]["text"] if i + 1 < len(sft_data) else text_a

        if dataset_type == "tinystories":
            # Prompt: beginning of story; chosen: full story; rejected: truncated/corrupted
            sentences = text_a.split(". ")
            if len(sentences) < 3:
                continue
            prompt = sentences[0] + "."
            chosen = ". ".join(sentences[1:])
            # Rejected: truncated or sentence-shuffled version
            rejected_sentences = sentences[1:]
            random.shuffle(rejected_sentences)
            rejected = ". ".join(rejected_sentences[:max(1, len(rejected_sentences) // 2)])

        elif dataset_type == "cnn_dailymail":
            prompt = sft_data[i].get("prompt", text_a[:300])
            chosen = sft_data[i].get("reference", text_a[300:600])
            # Rejected: use a mismatched summary from another article
            rejected = sft_data[min(i + 1, len(sft_data) - 1)].get("reference", text_b[:300])
            if chosen == rejected:
                continue

        elif dataset_type == "wikitext":
            sentences = text_a.split(". ")
            if len(sentences) < 3:
                continue
            prompt = sentences[0] + "."
            chosen = ". ".join(sentences[1:])
            # Rejected: corrupted version with word drops and shuffling
            words = chosen.split()
            if len(words) > 10:
                # Drop 30% of words randomly
                kept = [w for w in words if random.random() > 0.3]
                rejected = " ".join(kept) if kept else chosen[:50]
            else:
                rejected = chosen[:len(chosen) // 2]
        else:
            continue

        if len(chosen) > 20 and len(rejected) > 10 and chosen != rejected:
            pref_pairs.append({
                "prompt": prompt[:500],
                "chosen": chosen[:1000],
                "rejected": rejected[:1000],
            })

    logger.info(f"Created {len(pref_pairs)} preference pairs for {dataset_type}")
    return pref_pairs


def main():
    parser = argparse.ArgumentParser(description="Prepare all datasets for SLM-RL experiments")
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--max_samples", type=int, default=10000,
                        help="Max samples per dataset")
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--datasets", type=str, default="all",
                        help="Comma-separated: tinystories,cnn_dailymail,wikitext or 'all'")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_prepare = args.datasets.split(",") if args.datasets != "all" else [
        "tinystories", "cnn_dailymail", "wikitext"
    ]

    for ds_name in datasets_to_prepare:
        ds_name = ds_name.strip()
        if ds_name == "tinystories":
            prepare_tinystories(output_dir, args.max_samples, args.eval_ratio)
        elif ds_name == "cnn_dailymail":
            prepare_cnn_dailymail(output_dir, args.max_samples, args.eval_ratio)
        elif ds_name == "wikitext":
            prepare_wikitext(output_dir, args.max_samples, args.eval_ratio)
        else:
            logger.warning(f"Unknown dataset: {ds_name}")

    logger.info("All datasets prepared!")
    logger.info(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
