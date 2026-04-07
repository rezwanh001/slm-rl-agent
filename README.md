# SLM-RL-Agent: Efficient Small Language Model Alignment via RLHF

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/pytorch-2.5+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-mr3haque-yellow)](https://huggingface.co/mr3haque)

A research framework for applying Reinforcement Learning from Human Feedback (RLHF) to small language models (70M–160M parameters). Implements the full SFT → Reward Model → PPO pipeline with key engineering fixes for stable training on PEFT-adapted models.

## Results

| Model | Dataset | SFT PPL | SFT Reward | PPO Reward | Δ Reward | Win Rate |
|-------|---------|---------|------------|------------|----------|----------|
| Pythia-70M | TinyStories | 98.65 | 6.52±1.20 | 6.61±1.12 | **+0.085** | **52.1%** |
| Pythia-70M | CNN/DailyMail | 289.84 | 7.17±1.16 | 7.09±1.27 | −0.083 | 48.1% |
| Pythia-70M | Wikitext | 504.40 | 6.77±1.14 | 6.57±1.29 | −0.203 | 45.3% |
| Pythia-160M | TinyStories | 18.61 | −9.01±1.62 | −9.28±1.09 | −0.270 | 44.5% |
| Pythia-160M | CNN/DailyMail | 46.77 | −9.57±1.09 | −9.59±0.72 | −0.018 | 49.5% |
| Pythia-160M | Wikitext | 81.79 | −9.68±0.90 | −9.72±0.79 | −0.035 | 48.8% |

> **Note**: Reward scores are on per-model scales (not directly comparable across models). Win rate is the analytical probability that a PPO response scores higher than an SFT response.

## Key Engineering Findings

### 1. PPO with PEFT: Merge-and-Reinitialize Strategy
TRL 0.9.6 freezes LoRA parameters when loading a PEFT-adapted policy, causing PPO gradients to have no effect. The fix:
```python
# Step 1: Merge SFT LoRA adapter into base model weights
peft_model = PeftModel.from_pretrained(base_model, sft_adapter_path)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(merged_dir)

# Step 2: Apply a fresh LoRA on the merged model for PPO
fresh_lora_config = LoraConfig(r=8, lora_alpha=8, lora_dropout=0.0, ...)
policy = AutoModelForCausalLMWithValueHead.from_pretrained(
    merged_dir, peft_config=fresh_lora_config
)
# Step 3: Use merged model (no LoRA) as frozen reference
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(merged_dir)
```

### 2. Float32 for Numerical Stability
BFloat16 causes PPO ratio explosions for models with large reward variance. Always use `torch_dtype=torch.float32` for the policy and reference models during PPO.

### 3. Perplexity with Padding Tokens
When computing perplexity on padded batches, mask padding positions:
```python
labels = inputs["input_ids"].clone()
labels[inputs["attention_mask"] == 0] = -100  # mask padding
outputs = model(**inputs, labels=labels)
perplexity = torch.exp(outputs.loss)
```

## HuggingFace Resources

### Datasets
- [mr3haque/slm-rl-tinystories](https://huggingface.co/datasets/mr3haque/slm-rl-tinystories)
- [mr3haque/slm-rl-cnn_dailymail](https://huggingface.co/datasets/mr3haque/slm-rl-cnn_dailymail)
- [mr3haque/slm-rl-wikitext](https://huggingface.co/datasets/mr3haque/slm-rl-wikitext)
- [mr3haque/slm-rl-agent-results](https://huggingface.co/datasets/mr3haque/slm-rl-agent-results) — aggregated metrics JSON

### Models (SFT checkpoints)
- [mr3haque/slm-rl-pythia-70m-tinystories-sft](https://huggingface.co/mr3haque/slm-rl-pythia-70m-tinystories-sft)
- [mr3haque/slm-rl-pythia-70m-cnn-dailymail-sft](https://huggingface.co/mr3haque/slm-rl-pythia-70m-cnn-dailymail-sft)
- [mr3haque/slm-rl-pythia-70m-wikitext-sft](https://huggingface.co/mr3haque/slm-rl-pythia-70m-wikitext-sft)
- [mr3haque/slm-rl-pythia-160m-tinystories-sft](https://huggingface.co/mr3haque/slm-rl-pythia-160m-tinystories-sft)
- [mr3haque/slm-rl-pythia-160m-cnn-dailymail-sft](https://huggingface.co/mr3haque/slm-rl-pythia-160m-cnn-dailymail-sft)
- [mr3haque/slm-rl-pythia-160m-wikitext-sft](https://huggingface.co/mr3haque/slm-rl-pythia-160m-wikitext-sft)

### Models (PPO checkpoints)
- [mr3haque/slm-rl-pythia-70m-tinystories-ppo](https://huggingface.co/mr3haque/slm-rl-pythia-70m-tinystories-ppo)
- [mr3haque/slm-rl-pythia-70m-cnn-dailymail-ppo](https://huggingface.co/mr3haque/slm-rl-pythia-70m-cnn-dailymail-ppo)
- [mr3haque/slm-rl-pythia-70m-wikitext-ppo](https://huggingface.co/mr3haque/slm-rl-pythia-70m-wikitext-ppo)
- [mr3haque/slm-rl-pythia-160m-tinystories-ppo](https://huggingface.co/mr3haque/slm-rl-pythia-160m-tinystories-ppo)
- [mr3haque/slm-rl-pythia-160m-cnn-dailymail-ppo](https://huggingface.co/mr3haque/slm-rl-pythia-160m-cnn-dailymail-ppo)
- [mr3haque/slm-rl-pythia-160m-wikitext-ppo](https://huggingface.co/mr3haque/slm-rl-pythia-160m-wikitext-ppo)

## Quick Start

### Requirements
```bash
conda create -n slm-rl python=3.10
conda activate slm-rl
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.45.2 trl==0.9.6 peft==0.18.1
pip install datasets accelerate evaluate rouge_score nltk
```

### Run Full Pipeline (Single Configuration)

```bash
# 1. Prepare datasets
python scripts/prepare_all_datasets.py

# 2. SFT
python scripts/train_sft.py \
    --model_name EleutherAI/pythia-70m-deduped \
    --dataset_path data/tinystories/sft_train.json \
    --output_dir outputs/pythia-70m/tinystories/sft

# 3. Reward Model
python scripts/train_reward.py \
    --base_model outputs/pythia-70m/tinystories/sft/final \
    --dataset_path data/tinystories/preference_train.json \
    --output_dir outputs/pythia-70m/tinystories/reward_model

# 4. PPO
python scripts/train_ppo.py \
    --policy_model outputs/pythia-70m/tinystories/sft/final \
    --reward_model outputs/pythia-70m/tinystories/reward_model/final \
    --output_dir outputs/pythia-70m/tinystories/ppo \
    --num_steps 500

# 5. Evaluate
python scripts/evaluate.py \
    --model_path outputs/pythia-70m/tinystories/ppo/final \
    --reward_model_path outputs/pythia-70m/tinystories/reward_model/final \
    --dataset_path data/tinystories/sft_eval.json \
    --output_dir outputs/pythia-70m/tinystories/eval_ppo
```

### Run All Experiments
```bash
# Runs all 6 configurations (2 models × 3 datasets) on 2 GPUs in parallel
bash scripts/run_all_experiments.sh
```

### Interactive Demo
```bash
pip install gradio
python app.py
```

## Project Structure

```
slm-rl-agent/
├── scripts/
│   ├── prepare_all_datasets.py   # Dataset preparation
│   ├── train_sft.py              # SFT training
│   ├── train_reward.py           # Reward model training
│   ├── train_ppo.py              # PPO training (with merge-and-reinitialize)
│   ├── evaluate.py               # Evaluation
│   └── run_all_experiments.sh    # End-to-end runner
├── data/                         # Prepared datasets (generated)
├── outputs/                      # Training outputs (generated)
├── results/
│   └── all_results.json          # Aggregated metrics
├── paper/
│   ├── main.tex                  # Academic paper
│   ├── references.bib
│   └── figures/
│       └── pipeline.pdf          # Pipeline diagram
├── app.py                        # Gradio demo
└── requirements.txt
```

## Citation

```bibtex
@misc{haque2026slmrl,
  title   = {Efficiently Enhancing SLM Agents: A Reinforcement Learning Approach to Performance Improvement},
  author  = {Haque, Md Rezwanul},
  year    = {2026},
  url     = {https://github.com/rezwanh001/slm-rl-agent}
}
```
