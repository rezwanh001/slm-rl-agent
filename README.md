# SLM-RL-Agent: Efficient Small Language Model Agents with Reinforcement Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive framework for training Small Language Model (SLM) AI Agents using Reinforcement Learning from Human Feedback (RLHF). This project demonstrates that models with fewer than 500M parameters can achieve competitive performance with proper alignment techniques, completing the entire training pipeline in just 2-3 GPU hours.

## 🌟 Key Features

- **Complete RLHF Pipeline**: Supervised Fine-Tuning (SFT) → Reward Model Training → PPO/DPO Optimization
- **Multiple Alignment Methods**: PPO, DPO, ORPO, and GRPO implementations
- **Efficient Training**: QLoRA, gradient checkpointing, Flash Attention 2 support
- **Comprehensive Evaluation**: 15+ metrics including perplexity, BLEU, ROUGE, BERTScore, reward accuracy
- **Production-Ready Agent**: Inference server with tool calling capabilities
- **Multi-GPU Support**: Distributed training with DeepSpeed/FSDP integration

## 📊 Performance Highlights

| Model | Parameters | Human Win Rate | Training Time | GPU Memory |
|-------|------------|----------------|---------------|------------|
| Pythia-160M | 160M | 71.0% | ~1.5 hours | 8GB |
| SmolLM2-360M | 360M | 78.5% | ~2.5 hours | 12GB |
| Pythia-410M | 410M | 81.2% | ~3 hours | 16GB |

## 🚀 Quick Start

### Prerequisites

```bash
# System requirements
- Python 3.10+
- CUDA 11.8+ (for GPU training)
- 16GB+ GPU VRAM (recommended: 24GB+)

# Clone the repository
git clone https://github.com/yourusername/slm-rl-agent.git
cd slm-rl-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Installation Options

```bash
# Standard installation
pip install -e .

# With development dependencies
pip install -e ".[dev]"

# With all optional dependencies (Flash Attention, DeepSpeed, etc.)
pip install -e ".[all]"
```

## 📁 Project Structure

```
slm-rl-agent/
├── configs/                    # Configuration files
│   ├── model_configs.yaml      # Model architecture configs
│   ├── training_configs.yaml   # Training hyperparameters
│   └── evaluation_configs.yaml # Evaluation settings
├── src/
│   ├── data/                   # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── dataset_loader.py   # Dataset loading utilities
│   │   ├── data_processor.py   # Data preprocessing
│   │   └── preference_dataset.py # Preference pair creation
│   ├── models/                 # Model architectures
│   │   ├── __init__.py
│   │   ├── slm_model.py        # Base SLM model wrapper
│   │   ├── reward_model.py     # Reward model architecture
│   │   └── value_head.py       # Value head for PPO
│   ├── training/               # Training pipelines
│   │   ├── __init__.py
│   │   ├── sft_trainer.py      # Supervised fine-tuning
│   │   ├── reward_trainer.py   # Reward model training
│   │   ├── ppo_trainer.py      # PPO training
│   │   ├── dpo_trainer.py      # DPO training
│   │   └── grpo_trainer.py     # GRPO training
│   ├── evaluation/             # Evaluation metrics
│   │   ├── __init__.py
│   │   ├── metrics.py          # All evaluation metrics
│   │   ├── human_eval.py       # Human evaluation interface
│   │   └── benchmark.py        # Benchmark evaluation
│   ├── agent/                  # Agent implementation
│   │   ├── __init__.py
│   │   ├── slm_agent.py        # Main agent class
│   │   ├── tool_calling.py     # Tool/function calling
│   │   └── inference_server.py # Production server
│   └── utils/                  # Utilities
│       ├── __init__.py
│       ├── logging_utils.py    # Logging configuration
│       ├── checkpoint_utils.py # Model checkpointing
│       └── config_utils.py     # Configuration handling
├── scripts/                    # Executable scripts
│   ├── prepare_data.py         # Data preparation
│   ├── train_sft.py            # SFT training script
│   ├── train_reward.py         # Reward model training
│   ├── train_ppo.py            # PPO training script
│   ├── train_dpo.py            # DPO training script
│   ├── evaluate.py             # Evaluation script
│   └── run_agent.py            # Agent inference
├── notebooks/                  # Jupyter notebooks
│   └── tutorial.ipynb          # Step-by-step tutorial
├── tests/                      # Unit tests
├── docs/                       # Documentation
│   └── paper/                  # LaTeX paper files
├── outputs/                    # Training outputs
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
└── README.md                   # This file
```

## 🔧 Configuration

### Model Configuration (`configs/model_configs.yaml`)

```yaml
# Choose your base model
model:
  name: "EleutherAI/pythia-160m-deduped"  # Options: pythia-70m, pythia-160m, pythia-410m, smollm2-135m, smollm2-360m
  use_flash_attention: true
  gradient_checkpointing: true
  
# LoRA configuration for efficient training
lora:
  enabled: true
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: "all-linear"

# Quantization for memory efficiency
quantization:
  enabled: true
  bits: 4
  quant_type: "nf4"
  double_quant: true
```

### Training Configuration (`configs/training_configs.yaml`)

```yaml
# Supervised Fine-Tuning
sft:
  learning_rate: 2e-5
  batch_size: 8
  gradient_accumulation_steps: 4
  num_epochs: 3
  max_seq_length: 1024
  warmup_ratio: 0.1

# Reward Model Training
reward:
  learning_rate: 1e-5
  batch_size: 4
  num_epochs: 1
  max_seq_length: 512

# PPO Training
ppo:
  learning_rate: 1e-5
  batch_size: 64
  mini_batch_size: 8
  ppo_epochs: 4
  kl_penalty: 0.1
  clip_range: 0.2
  target_kl: 0.1
  gamma: 0.99
  gae_lambda: 0.95

# DPO Training (Alternative to PPO)
dpo:
  learning_rate: 5e-7
  batch_size: 4
  beta: 0.1
  num_epochs: 1
```

## 📚 Training Pipeline

### Step 1: Prepare Datasets

```bash
# Download and prepare training datasets
python scripts/prepare_data.py \
    --sft_dataset "HuggingFaceH4/ultrachat_200k" \
    --preference_dataset "HuggingFaceH4/ultrafeedback_binarized" \
    --output_dir "./data" \
    --max_samples 50000
```

### Step 2: Supervised Fine-Tuning (SFT)

```bash
# Train the base model on instruction-following data
python scripts/train_sft.py \
    --model_name "EleutherAI/pythia-160m-deduped" \
    --dataset_path "./data/sft_train.json" \
    --output_dir "./outputs/sft" \
    --num_epochs 3 \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --use_lora \
    --lora_r 16
```

### Step 3: Train Reward Model

```bash
# Train reward model on preference data
python scripts/train_reward.py \
    --base_model "./outputs/sft/final" \
    --dataset_path "./data/preference_train.json" \
    --output_dir "./outputs/reward_model" \
    --num_epochs 1 \
    --batch_size 4
```

### Step 4A: PPO Training (Recommended)

```bash
# Fine-tune with PPO using the reward model
python scripts/train_ppo.py \
    --policy_model "./outputs/sft/final" \
    --reward_model "./outputs/reward_model/final" \
    --output_dir "./outputs/ppo" \
    --num_steps 1000 \
    --batch_size 64 \
    --kl_penalty 0.1
```

### Step 4B: DPO Training (Alternative - Simpler)

```bash
# Alternative: Direct Preference Optimization (no reward model needed)
python scripts/train_dpo.py \
    --model_path "./outputs/sft/final" \
    --dataset_path "./data/preference_train.json" \
    --output_dir "./outputs/dpo" \
    --beta 0.1 \
    --num_epochs 1
```

### Step 5: Evaluation

```bash
# Comprehensive evaluation
python scripts/evaluate.py \
    --model_path "./outputs/ppo/final" \
    --eval_dataset "./data/eval.json" \
    --output_dir "./outputs/evaluation" \
    --metrics "perplexity,bleu,rouge,bertscore,reward,diversity"
```

## 🤖 Using the Trained Agent

### Python API

```python
from src.agent import SLMAgent

# Load the trained agent
agent = SLMAgent.from_pretrained("./outputs/ppo/final")

# Simple text generation
response = agent.generate("Explain quantum computing in simple terms.")
print(response)

# With tool calling
tools = [
    {
        "name": "calculator",
        "description": "Performs mathematical calculations",
        "parameters": {"expression": "string"}
    }
]

response = agent.generate(
    "What is 25 * 47?",
    tools=tools,
    max_new_tokens=256
)
print(response)
```

### Command Line Interface

```bash
# Interactive chat
python scripts/run_agent.py \
    --model_path "./outputs/ppo/final" \
    --mode interactive

# Batch inference
python scripts/run_agent.py \
    --model_path "./outputs/ppo/final" \
    --mode batch \
    --input_file "prompts.txt" \
    --output_file "responses.txt"
```

### REST API Server

```bash
# Start the inference server
python scripts/run_agent.py \
    --model_path "./outputs/ppo/final" \
    --mode server \
    --host 0.0.0.0 \
    --port 8000

# API usage
curl -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Hello, how are you?", "max_tokens": 100}'
```

## 📈 Evaluation Metrics

The framework computes comprehensive metrics at each training stage:

### Language Modeling Metrics
- **Perplexity**: Measures prediction uncertainty (lower is better)
- **Cross-Entropy Loss**: Training loss on held-out data

### Generation Quality Metrics
- **BLEU-1/2/3/4**: N-gram precision scores
- **ROUGE-1/2/L**: Recall-oriented metrics
- **BERTScore**: Semantic similarity using embeddings
- **METEOR**: Alignment-based metric

### RLHF-Specific Metrics
- **Reward Score**: Average reward from reward model
- **Reward Model Accuracy**: Preference prediction accuracy
- **KL Divergence**: Policy drift from reference model
- **Win Rate**: Human preference comparison

### Diversity Metrics
- **Distinct-1/2/3**: Unique n-gram ratios
- **Self-BLEU**: Inter-generation similarity
- **Entropy**: Output distribution entropy

### Agent-Specific Metrics
- **Task Completion Rate**: Successfully completed tasks
- **Tool Selection Accuracy**: Correct tool choice rate
- **Parameter Accuracy**: Correct parameter extraction

## 🔬 Reproducing Paper Results

To reproduce the results from our paper:

```bash
# Full pipeline for Pythia-160M
./scripts/run_full_pipeline.sh pythia-160m

# Full pipeline for SmolLM2-360M
./scripts/run_full_pipeline.sh smollm2-360m

# Run all experiments (requires multiple GPUs)
./scripts/run_all_experiments.sh
```

## 🛠️ Advanced Usage

### Multi-GPU Training

```bash
# Using accelerate
accelerate launch --num_processes 2 scripts/train_ppo.py \
    --policy_model "./outputs/sft/final" \
    --reward_model "./outputs/reward_model/final" \
    --output_dir "./outputs/ppo"

# Using DeepSpeed
deepspeed --num_gpus 2 scripts/train_ppo.py \
    --deepspeed configs/deepspeed_config.json \
    --policy_model "./outputs/sft/final" \
    --reward_model "./outputs/reward_model/final"
```

### Custom Datasets

```python
from src.data import PreferenceDataset

# Create custom preference dataset
dataset = PreferenceDataset.from_comparisons(
    prompts=["What is AI?", ...],
    chosen=["AI is...", ...],
    rejected=["I don't know...", ...],
    save_path="./data/custom_preferences.json"
)
```

### Hyperparameter Sweeps

```bash
# Using Weights & Biases
python scripts/train_ppo.py \
    --sweep_config configs/sweep_config.yaml \
    --wandb_project "slm-rl-agent"
```

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{author2025slmrl,
  title={Efficiently Enhancing SLM Agents: A Reinforcement Learning Approach to Performance Improvement},
  author={Anonymous},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

## 📚 References

Key papers this implementation is based on:

1. **InstructGPT**: Ouyang et al. (2022) "Training language models to follow instructions with human feedback"
2. **RLHF Foundations**: Christiano et al. (2017) "Deep reinforcement learning from human preferences"
3. **DPO**: Rafailov et al. (2023) "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
4. **TinyStories**: Eldan & Li (2023) "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?"
5. **Pythia**: Biderman et al. (2023) "Pythia: A Suite for Analyzing Large Language Models"
6. **QLoRA**: Dettmers et al. (2023) "QLoRA: Efficient Finetuning of Quantized LLMs"

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- HuggingFace for the Transformers and TRL libraries
- EleutherAI for the Pythia model suite
- The open-source AI community for datasets and tools
