#!/usr/bin/env python3
"""
Setup file for SLM-RL-Agent package.

This package provides a comprehensive framework for training Small Language Model (SLM)
AI Agents using Reinforcement Learning from Human Feedback (RLHF).

Installation:
    pip install -e .                    # Standard installation
    pip install -e ".[dev]"             # With development dependencies
    pip install -e ".[all]"             # With all optional dependencies
"""

from setuptools import setup, find_packages

# Read the README for the long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Core dependencies required for basic functionality
INSTALL_REQUIRES = [
    "torch>=2.1.0",
    "transformers>=4.36.0",
    "datasets>=2.16.0",
    "accelerate>=0.25.0",
    "peft>=0.7.0",
    "trl>=0.7.4",
    "bitsandbytes>=0.41.0",
    "tokenizers>=0.15.0",
    "safetensors>=0.4.0",
    "huggingface-hub>=0.20.0",
    "pyyaml>=6.0",
    "tqdm>=4.66.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
]

# Development dependencies for testing and code quality
DEV_REQUIRES = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.12.0",
    "isort>=5.13.0",
    "flake8>=6.1.0",
    "mypy>=1.7.0",
    "pre-commit>=3.6.0",
]

# Evaluation metrics dependencies
EVAL_REQUIRES = [
    "evaluate>=0.4.0",
    "rouge-score>=0.1.2",
    "nltk>=3.8.0",
    "bert-score>=0.3.13",
    "sacrebleu>=2.3.0",
    "scikit-learn>=1.3.0",
]

# Experiment tracking and logging
LOGGING_REQUIRES = [
    "wandb>=0.16.0",
    "tensorboard>=2.15.0",
    "rich>=13.0.0",
]

# Server and API dependencies
SERVER_REQUIRES = [
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    "pydantic>=2.5.0",
]

# Jupyter notebook support
NOTEBOOK_REQUIRES = [
    "jupyter>=1.0.0",
    "ipywidgets>=8.1.0",
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
]

# All optional dependencies combined
ALL_REQUIRES = (
    DEV_REQUIRES + 
    EVAL_REQUIRES + 
    LOGGING_REQUIRES + 
    SERVER_REQUIRES + 
    NOTEBOOK_REQUIRES
)

setup(
    name="slm-rl-agent",
    version="1.0.0",
    author="SLM-RL-Agent Team",
    author_email="",
    description="Efficient Small Language Model Agents with Reinforcement Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/slm-rl-agent",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/slm-rl-agent/issues",
        "Documentation": "https://github.com/yourusername/slm-rl-agent#readme",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(exclude=["tests", "notebooks", "docs"]),
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "dev": DEV_REQUIRES,
        "eval": EVAL_REQUIRES,
        "logging": LOGGING_REQUIRES,
        "server": SERVER_REQUIRES,
        "notebook": NOTEBOOK_REQUIRES,
        "all": ALL_REQUIRES,
    },
    entry_points={
        "console_scripts": [
            "slm-train-sft=scripts.train_sft:main",
            "slm-train-reward=scripts.train_reward:main",
            "slm-train-ppo=scripts.train_ppo:main",
            "slm-train-dpo=scripts.train_dpo:main",
            "slm-evaluate=scripts.evaluate:main",
            "slm-agent=scripts.run_agent:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
