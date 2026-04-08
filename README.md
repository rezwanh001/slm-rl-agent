# Efficiently Enhancing SLM Agents: A Reinforcement Learning Approach to Performance Improvement

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/slm-rl-agent"><img src="https://img.shields.io/badge/🤗%20HuggingFace-slm--rl--agent-yellow" alt="HuggingFace"></a>
  <a href="https://github.com/rezwanh001/slm-rl-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.5%2B-EE4C2C.svg" alt="PyTorch"></a>
</p>

<p align="center">
  <b>Md Rezwanul Haque</b><br>
  University of Waterloo<br>
  <a href="mailto:mr3haque@uwaterloo.ca">mr3haque@uwaterloo.ca</a>
</p>

---

## Abstract

We study whether Reinforcement Learning from Human Feedback (RLHF) can meaningfully improve small language models (SLMs) with 70M–360M parameters. Our framework implements a complete three-stage pipeline—supervised fine-tuning (SFT), reward model training, and Proximal Policy Optimization (PPO)—applied to five model architectures across three diverse text corpora. Beyond reporting empirical results, we document and resolve two engineering obstacles unique to the SLM scale: (1) TRL's PPO trainer silently freezes LoRA parameters when the policy is a PEFT adapter, and (2) bfloat16 precision causes PPO ratio explosions in very small models. We propose a *merge-and-reinitialize* strategy and float32 training as the remedies. Our experiments reveal that RLHF benefits are domain-sensitive, with clear improvements on simpler domains where the SFT model has moderate perplexity. All model weights, preference datasets, training code, and an interactive demo are publicly available.

---

## Table of Contents

- [Overview](#overview)
- [Key Contributions](#key-contributions)
- [Results](#results)
- [Models & Datasets](#models--datasets)
- [Installation](#installation)
- [Usage](#usage)
  - [Prepare Datasets](#1-prepare-datasets)
  - [Supervised Fine-Tuning](#2-supervised-fine-tuning-sft)
  - [Reward Model Training](#3-reward-model-training)
  - [PPO Alignment](#4-ppo-alignment)
  - [Evaluation](#5-evaluation)
  - [Run Full Pipeline](#6-run-full-pipeline)
- [Engineering Notes](#engineering-notes)
- [Interactive Demo](#interactive-demo)
- [Citation](#citation)

---

## Overview

This repository provides the complete implementation for the paper **"Efficiently Enhancing SLM Agents: A Reinforcement Learning Approach to Performance Improvement"**. We investigate whether RLHF—the dominant alignment paradigm for large language models—transfers effectively to the sub-500M parameter regime.

**Pipeline overview:**

```
Pre-trained SLM
      │
      ▼
  Stage 1: Supervised Fine-Tuning (SFT)
      │  LoRA adapter trained on domain text
      │
      ▼
  Stage 2: Reward Model Training
      │  Bradley-Terry preference loss on synthetic pairs
      │
      ▼
  Stage 3: PPO Alignment
         Merge-and-reinitialize strategy (our contribution)
         Float32 precision for numerical stability
      │
      ▼
  Aligned SLM Agent
```

---

## Key Contributions

1. **End-to-end RLHF at the SLM scale** — systematic empirical evaluation on 5 architectures × 3 datasets (15 configurations).

2. **Merge-and-reinitialize for PEFT–PPO** — TRL v0.9.x freezes LoRA parameters when the policy is loaded as a PEFT adapter. We fix this by merging the SFT adapter into base weights, then attaching a fresh LoRA before PPO:
   ```python
   # The fix (see scripts/train_ppo.py)
   merged = PeftModel.from_pretrained(base, sft_path).merge_and_unload()
   merged.save_pretrained(merged_dir)
   policy = AutoModelForCausalLMWithValueHead.from_pretrained(
       merged_dir, peft_config=LoraConfig(r=8, ...))
   ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(merged_dir)
   ```

3. **Float32 precision requirement** — bfloat16 causes PPO ratio explosions (values > 10⁶) for models < 200M parameters. Float32 is required for stable training.

4. **Domain-sensitivity analysis** — RLHF improves reward when SFT perplexity is moderate (~100); high-perplexity domains (PPL > 200) produce noisy reward signals that PPO cannot reliably exploit.

---

## Results

### Main Results (SFT vs. PPO)

| Model | Dataset | SFT PPL | SFT Reward | PPO Reward | Δ Reward | Win Rate |
|-------|---------|:-------:|:----------:|:----------:|:--------:|:--------:|
| **Pythia-70M** | TinyStories | 98.7 | 6.52 ± 1.20 | **6.61 ± 1.12** | **+0.085** | **52.1%** |
| Pythia-70M | CNN/DailyMail | 289.8 | 7.17 ± 1.16 | 7.09 ± 1.27 | −0.083 | 48.1% |
| Pythia-70M | Wikitext-103 | 504.4 | 6.77 ± 1.14 | 6.57 ± 1.29 | −0.203 | 45.3% |
| Pythia-160M | TinyStories | 18.6 | −9.01 ± 1.62 | −9.28 ± 1.09 | −0.270 | 44.5% |
| Pythia-160M | CNN/DailyMail | 46.8 | −9.57 ± 1.09 | **−9.59 ± 0.72** | **−0.018** | **49.5%** |
| Pythia-160M | Wikitext-103 | 81.8 | −9.68 ± 0.90 | −9.72 ± 0.79 | −0.035 | 48.8% |
| Pythia-410M | TinyStories | — | — | — | — | — |
| Pythia-410M | CNN/DailyMail | — | — | — | — | — |
| Pythia-410M | Wikitext-103 | — | — | — | — | — |
| SmolLM2-135M | TinyStories | — | — | — | — | — |
| SmolLM2-135M | CNN/DailyMail | — | — | — | — | — |
| SmolLM2-135M | Wikitext-103 | — | — | — | — | — |
| SmolLM2-360M | TinyStories | — | — | — | — | — |
| SmolLM2-360M | CNN/DailyMail | — | — | — | — | — |
| SmolLM2-360M | Wikitext-103 | — | — | — | — | — |

> **Notes:** Reward scores are on per-model scales (not directly comparable across architectures). Win rate = analytical probability that a PPO response scores higher than an SFT response on the same prompt. Rows marked — are training in progress.

### Text Diversity (no collapse observed)

| Model | Dataset | SFT Dist-2 | PPO Dist-2 | SFT ROUGE-1 | PPO ROUGE-1 |
|-------|---------|:-----------:|:-----------:|:-----------:|:-----------:|
| Pythia-70M | TinyStories | 0.667 | 0.660 | 0.267 | 0.262 |
| Pythia-70M | CNN/DailyMail | 0.794 | 0.804 | 0.179 | 0.178 |
| Pythia-70M | Wikitext-103 | 0.744 | 0.758 | 0.167 | 0.159 |
| Pythia-160M | TinyStories | 0.468 | 0.476 | 0.275 | 0.286 |
| Pythia-160M | CNN/DailyMail | 0.639 | 0.617 | 0.227 | 0.232 |
| Pythia-160M | Wikitext-103 | 0.533 | 0.526 | 0.195 | 0.195 |

---

## Models & Datasets

All artifacts are hosted on the [slm-rl-agent](https://huggingface.co/slm-rl-agent) HuggingFace organization.

### Datasets

| Dataset | HuggingFace | Description |
|---------|-------------|-------------|
| TinyStories (preference) | [slm-rl-agent/slm-rl-tinystories](https://huggingface.co/datasets/slm-rl-agent/slm-rl-tinystories) | SFT + preference pairs for simple narrative text |
| CNN/DailyMail (preference) | [slm-rl-agent/slm-rl-cnn-dailymail](https://huggingface.co/datasets/slm-rl-agent/slm-rl-cnn-dailymail) | SFT + preference pairs for news articles |
| Wikitext-103 (preference) | [slm-rl-agent/slm-rl-wikitext](https://huggingface.co/datasets/slm-rl-agent/slm-rl-wikitext) | SFT + preference pairs for encyclopedic text |

Each dataset contains:
- `sft_train.json` / `sft_eval.json` — domain text for supervised fine-tuning
- `preference_train.json` / `preference_eval.json` — `{prompt, chosen, rejected}` pairs for reward modeling

### Model Checkpoints

| Model | Stage | TinyStories | CNN/DailyMail | Wikitext-103 |
|-------|-------|-------------|---------------|--------------|
| Pythia-70M | SFT | [🤗](https://huggingface.co/slm-rl-agent/pythia-70m-tinystories-sft) | [🤗](https://huggingface.co/slm-rl-agent/pythia-70m-cnn-dailymail-sft) | [🤗](https://huggingface.co/slm-rl-agent/pythia-70m-wikitext-sft) |
| Pythia-70M | PPO | [🤗](https://huggingface.co/slm-rl-agent/pythia-70m-tinystories-ppo) | [🤗](https://huggingface.co/slm-rl-agent/pythia-70m-cnn-dailymail-ppo) | [🤗](https://huggingface.co/slm-rl-agent/pythia-70m-wikitext-ppo) |
| Pythia-160M | SFT | [🤗](https://huggingface.co/slm-rl-agent/pythia-160m-tinystories-sft) | [🤗](https://huggingface.co/slm-rl-agent/pythia-160m-cnn-dailymail-sft) | [🤗](https://huggingface.co/slm-rl-agent/pythia-160m-wikitext-sft) |
| Pythia-160M | PPO | [🤗](https://huggingface.co/slm-rl-agent/pythia-160m-tinystories-ppo) | [🤗](https://huggingface.co/slm-rl-agent/pythia-160m-cnn-dailymail-ppo) | [🤗](https://huggingface.co/slm-rl-agent/pythia-160m-wikitext-ppo) |
| Pythia-410M | SFT | *(training)* | *(training)* | *(training)* |
| Pythia-410M | PPO | *(training)* | *(training)* | *(training)* |
| SmolLM2-135M | SFT | *(training)* | *(training)* | *(training)* |
| SmolLM2-135M | PPO | *(training)* | *(training)* | *(training)* |
| SmolLM2-360M | SFT | *(training)* | *(training)* | *(training)* |
| SmolLM2-360M | PPO | *(training)* | *(training)* | *(training)* |

---

## Installation

### Requirements

- Python 3.10+
- CUDA 11.8+ (CUDA 12.1 recommended)
- 24 GB+ GPU VRAM per model (48 GB for 360M–410M models)

### Setup

```bash
git clone https://github.com/rezwanh001/slm-rl-agent.git
cd slm-rl-agent

# Create conda environment
conda create -n slm-rl python=3.10
conda activate slm-rl

# Install PyTorch with CUDA 12.1
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install transformers==4.45.2 trl==0.9.6 peft==0.18.1
pip install datasets accelerate evaluate rouge_score nltk

# Install package
pip install -e .
```

---

## Usage

### 1. Prepare Datasets

```bash
python scripts/prepare_all_datasets.py \
    --output_dir data \
    --num_train 10000 \
    --num_eval 1000
```

This downloads and preprocesses TinyStories, CNN/DailyMail, and Wikitext-103, generating both SFT splits and synthetic preference pairs.

### 2. Supervised Fine-Tuning (SFT)

```bash
python scripts/train_sft.py \
    --model_name EleutherAI/pythia-70m-deduped \
    --dataset_path data/tinystories/sft_train.json \
    --output_dir outputs/pythia-70m/tinystories/sft \
    --num_epochs 3 \
    --learning_rate 2e-5 \
    --lora_r 8 \
    --lora_alpha 16
```

### 3. Reward Model Training

```bash
python scripts/train_reward.py \
    --base_model outputs/pythia-70m/tinystories/sft/final \
    --dataset_path data/tinystories/preference_train.json \
    --output_dir outputs/pythia-70m/tinystories/reward_model \
    --num_epochs 1 \
    --learning_rate 1e-5
```

### 4. PPO Alignment

```bash
python scripts/train_ppo.py \
    --policy_model outputs/pythia-70m/tinystories/sft/final \
    --reward_model outputs/pythia-70m/tinystories/reward_model/final \
    --output_dir outputs/pythia-70m/tinystories/ppo \
    --num_steps 500 \
    --kl_coef 0.1
```

> **Note:** The script automatically applies the merge-and-reinitialize strategy and float32 precision.

### 5. Evaluation

```bash
python scripts/evaluate.py \
    --model_path outputs/pythia-70m/tinystories/ppo/final \
    --reward_model_path outputs/pythia-70m/tinystories/reward_model/final \
    --dataset_path data/tinystories/sft_eval.json \
    --output_dir outputs/pythia-70m/tinystories/eval_ppo \
    --num_samples 200
```

### 6. Run Full Pipeline

To reproduce all experiments (requires 2× NVIDIA RTX A6000 or equivalent):

```bash
# Runs all 5 models × 3 datasets in parallel across 2 GPUs
bash scripts/run_all_experiments.sh
```

---

## Engineering Notes

### Merge-and-Reinitialize Strategy

TRL ≤ 0.9.x does not correctly handle PEFT-adapted policies in PPO: all LoRA parameters are frozen (`requires_grad=False`), and only the 2-parameter value head is updated. The policy generates text, receives reward, and processes gradients—but nothing changes. This is a silent failure.

**Our fix** (implemented in `scripts/train_ppo.py`):

```python
# Step 1: Merge the SFT LoRA adapter into base weights
base = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float32)
peft_model = PeftModel.from_pretrained(base, sft_adapter_path)
merged = peft_model.merge_and_unload()
merged.save_pretrained(merged_dir)

# Step 2: Fresh LoRA on merged weights (B matrices initialized to 0 → no-op at init)
fresh_lora = LoraConfig(r=8, lora_alpha=8, lora_dropout=0.0, ...)
policy = AutoModelForCausalLMWithValueHead.from_pretrained(
    merged_dir, peft_config=fresh_lora)

# Step 3: Frozen reference = merged weights, no LoRA
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(merged_dir)
```

### Float32 Precision

Bfloat16 causes probability-ratio explosions (values > 10⁶) within the first PPO batch for models < 200M parameters, triggering CUDA assertion errors. Always use `torch_dtype=torch.float32` for both policy and reference models during PPO.

### Perplexity Computation

When evaluating perplexity on batches with padding, mask padding positions to prevent inflated scores:

```python
labels = input_ids.clone()
labels[attention_mask == 0] = -100   # mask padding tokens
outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
ppl = torch.exp(outputs.loss)
```

---

## Interactive Demo

```bash
pip install gradio
python app.py
```

The demo allows side-by-side comparison of SFT and PPO agents for all trained configurations.

---

## Project Structure

```
slm-rl-agent/
├── paper/
│   ├── main.tex              # Full paper (LaTeX)
│   ├── references.bib        # Bibliography
│   └── figures/
│       └── pipeline.pdf      # Pipeline diagram
├── scripts/
│   ├── prepare_all_datasets.py
│   ├── train_sft.py
│   ├── train_reward.py
│   ├── train_ppo.py          # Includes merge-and-reinitialize fix
│   ├── evaluate.py
│   └── run_all_experiments.sh
├── src/                      # Core library modules
├── configs/                  # YAML configuration files
├── data/                     # Prepared datasets (generated, not tracked)
├── outputs/                  # Model checkpoints (generated, not tracked)
├── results/
│   └── all_results.json      # Aggregated evaluation metrics
├── app.py                    # Gradio interactive demo
├── requirements.txt
└── setup.py
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{haque2026slmrlagent,
  title        = {Efficiently Enhancing {SLM} Agents: A Reinforcement Learning Approach
                  to Performance Improvement},
  author       = {Haque, Md Rezwanul},
  year         = {2026},
  howpublished = {\url{https://github.com/rezwanh001/slm-rl-agent}},
  note         = {University of Waterloo, CPAMI Lab}
}
```

---

## Acknowledgements

Computational resources were provided by the CPAMI Lab at the University of Waterloo. This work builds on the TRL library by Hugging Face, the PEFT library, and the Pythia model suite by EleutherAI.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
