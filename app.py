#!/usr/bin/env python3
"""
Interactive Gradio Demo for SLM-RL-Agent

Provides a web interface to:
1. Compare SFT vs PPO outputs side-by-side
2. Select different models and datasets
3. View reward scores for generated text
4. Try custom prompts

Usage:
    python app.py [--share] [--port 7860]
"""

import argparse
import glob
import json
import os
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Global model cache
_loaded_models = {}


def get_available_models(outputs_dir="./outputs"):
    """Scan outputs directory for trained models."""
    models = []
    for model_dir in sorted(glob.glob(os.path.join(outputs_dir, "*/*"))):
        model_name = os.path.basename(os.path.dirname(model_dir))
        dataset = os.path.basename(model_dir)
        sft_path = os.path.join(model_dir, "sft", "final")
        ppo_path = os.path.join(model_dir, "ppo", "final")

        if os.path.isdir(sft_path):
            models.append({
                "label": f"{model_name} / {dataset}",
                "model_name": model_name,
                "dataset": dataset,
                "sft_path": sft_path,
                "ppo_path": ppo_path if os.path.isdir(ppo_path) else None,
                "reward_path": os.path.join(model_dir, "reward_model", "final"),
            })
    return models


def load_model(model_path):
    """Load a model and tokenizer, with caching."""
    if model_path in _loaded_models:
        return _loaded_models[model_path]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    _loaded_models[model_path] = (model, tokenizer)
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_new_tokens=200, temperature=0.7, top_p=0.9):
    """Generate text from a model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.1,
        )

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()


def compute_reward(reward_path, tokenizer, prompt, response):
    """Compute reward score for a response."""
    try:
        from transformers import AutoModelForSequenceClassification

        if not os.path.isdir(reward_path):
            return "N/A"

        if reward_path not in _loaded_models:
            rm = AutoModelForSequenceClassification.from_pretrained(
                reward_path, torch_dtype=torch.bfloat16, device_map="auto"
            )
            rm.eval()
            _loaded_models[reward_path] = rm
        else:
            rm = _loaded_models[reward_path]

        text = f"{prompt}\n\n{response}"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(rm.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = rm(**inputs)
            score = outputs.logits[0, 0].item()

        return f"{score:.4f}"
    except Exception as e:
        return f"Error: {e}"


def compare_models(model_selection, prompt, max_tokens, temperature):
    """Generate and compare SFT vs PPO outputs."""
    available = get_available_models()

    if not available:
        return "No models found", "No models found", "N/A", "N/A"

    # Find selected model config
    selected = None
    for m in available:
        if m["label"] == model_selection:
            selected = m
            break

    if not selected:
        return "Model not found", "Model not found", "N/A", "N/A"

    # Generate SFT output
    try:
        sft_model, sft_tokenizer = load_model(selected["sft_path"])
        sft_output = generate_text(sft_model, sft_tokenizer, prompt, max_tokens, temperature)
    except Exception as e:
        sft_output = f"Error loading SFT model: {e}"

    # Generate PPO output
    if selected["ppo_path"]:
        try:
            ppo_model, ppo_tokenizer = load_model(selected["ppo_path"])
            ppo_output = generate_text(ppo_model, ppo_tokenizer, prompt, max_tokens, temperature)
        except Exception as e:
            ppo_output = f"Error loading PPO model: {e}"
    else:
        ppo_output = "PPO model not yet available for this configuration."

    # Compute reward scores
    sft_reward = compute_reward(selected["reward_path"], sft_tokenizer, prompt, sft_output)
    ppo_reward = compute_reward(selected["reward_path"], sft_tokenizer, prompt, ppo_output) if selected["ppo_path"] else "N/A"

    return sft_output, ppo_output, sft_reward, ppo_reward


def single_generate(model_selection, stage, prompt, max_tokens, temperature):
    """Generate text from a single model."""
    available = get_available_models()

    selected = None
    for m in available:
        if m["label"] == model_selection:
            selected = m
            break

    if not selected:
        return "Model not found"

    path = selected["sft_path"] if stage == "SFT" else selected.get("ppo_path", selected["sft_path"])
    if not path or not os.path.isdir(path):
        return f"{stage} model not available for this configuration."

    model, tokenizer = load_model(path)
    return generate_text(model, tokenizer, prompt, max_tokens, temperature)


# Example prompts per dataset
EXAMPLES = {
    "tinystories": [
        "Once upon a time, there was a little girl named Lily.",
        "Tom went to the park with his dog.",
        "The cat sat on the mat and looked at the bird.",
    ],
    "cnn_dailymail": [
        "Summarize: The president announced a new policy today regarding climate change.",
        "Summarize: Scientists have discovered a new species of fish in the deep ocean.",
        "Summarize: The technology company reported record earnings for the quarter.",
    ],
    "wikitext": [
        "The history of artificial intelligence began in",
        "Quantum computing is a type of computation that",
        "The Renaissance was a cultural movement that",
    ],
}


def build_demo():
    """Build the Gradio interface."""
    available = get_available_models()
    model_choices = [m["label"] for m in available] if available else ["No models found"]

    with gr.Blocks(
        title="SLM-RL-Agent Demo",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("""
        # SLM-RL-Agent: Interactive Demo
        Compare SFT (baseline) vs PPO (RL-enhanced) small language model agents.

        **Paper:** *Efficiently Enhancing SLM Agents: A Reinforcement Learning Approach*
        **Code:** [GitHub](https://github.com/rezwanh001/slm-rl-agent)
        """)

        with gr.Tab("Side-by-Side Comparison"):
            gr.Markdown("### Compare SFT vs PPO outputs for the same prompt")

            with gr.Row():
                model_select = gr.Dropdown(
                    choices=model_choices,
                    value=model_choices[0] if model_choices else None,
                    label="Select Model & Dataset",
                )
                max_tokens = gr.Slider(50, 500, value=200, step=10, label="Max Tokens")
                temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")

            prompt_input = gr.Textbox(
                label="Input Prompt",
                placeholder="Enter your prompt here...",
                lines=3,
            )
            compare_btn = gr.Button("Generate & Compare", variant="primary")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### SFT Output (Baseline)")
                    sft_output = gr.Textbox(label="SFT Response", lines=10)
                    sft_reward = gr.Textbox(label="SFT Reward Score")
                with gr.Column():
                    gr.Markdown("#### PPO Output (RL-Enhanced)")
                    ppo_output = gr.Textbox(label="PPO Response", lines=10)
                    ppo_reward = gr.Textbox(label="PPO Reward Score")

            compare_btn.click(
                compare_models,
                inputs=[model_select, prompt_input, max_tokens, temperature],
                outputs=[sft_output, ppo_output, sft_reward, ppo_reward],
            )

            gr.Markdown("### Example Prompts")
            with gr.Row():
                for dataset_name, examples in EXAMPLES.items():
                    with gr.Column():
                        gr.Markdown(f"**{dataset_name}**")
                        for ex in examples:
                            gr.Button(ex[:50] + "...").click(
                                lambda x=ex: x, outputs=[prompt_input]
                            )

        with gr.Tab("Single Model Generation"):
            gr.Markdown("### Generate text from a specific model")

            with gr.Row():
                model_select2 = gr.Dropdown(choices=model_choices, value=model_choices[0], label="Model")
                stage_select = gr.Radio(["SFT", "PPO"], value="PPO", label="Stage")
                max_tokens2 = gr.Slider(50, 500, value=200, step=10, label="Max Tokens")
                temp2 = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")

            prompt_input2 = gr.Textbox(label="Prompt", lines=3)
            gen_btn = gr.Button("Generate", variant="primary")
            output2 = gr.Textbox(label="Generated Output", lines=10)

            gen_btn.click(
                single_generate,
                inputs=[model_select2, stage_select, prompt_input2, max_tokens2, temp2],
                outputs=[output2],
            )

        with gr.Tab("About"):
            gr.Markdown("""
            ## About This Project

            This demo accompanies the paper *"Efficiently Enhancing SLM Agents:
            A Reinforcement Learning Approach to Performance Improvement"*.

            ### Pipeline
            1. **Supervised Fine-Tuning (SFT)**: Adapt base model to target domain
            2. **Reward Model Training**: Learn preference-based reward function
            3. **PPO Optimization**: Align policy to maximize reward while staying close to SFT

            ### Models
            - Pythia (70M, 160M, 410M parameters)
            - SmolLM2 (135M, 360M parameters)

            ### Datasets
            - TinyStories (fictional narratives)
            - CNN/DailyMail (news summarization)
            - Wikitext-103 (encyclopedic text)

            ### Citation
            ```bibtex
            @article{haque2025slm,
              title={Efficiently Enhancing SLM Agents: A Reinforcement Learning Approach},
              author={Haque, Md Rezwanul},
              year={2025}
            }
            ```
            """)

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--outputs_dir", type=str, default="./outputs")
    args = parser.parse_args()

    demo = build_demo()
    demo.launch(share=args.share, server_port=args.port)
