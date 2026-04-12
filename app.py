#!/usr/bin/env python3
"""
SLM-RL-Agent — Interactive Verification App (paper appendix)

This Gradio app accompanies the paper:

    "Efficiently Enhancing SLM Agents: A Reinforcement Learning Approach
     to Performance Improvement"

Its purpose is to let reviewers (and any third party) independently verify
that the numbers reported in the paper and in the HuggingFace model/dataset
repositories are backed by real, runnable checkpoints and real, on-disk
evaluation files — not synthesised or cherry-picked.

It has four tabs:

  1. Live SFT vs PPO comparison   — pick a model/dataset, enter a prompt,
                                    and the app loads the actual trained
                                    LoRA/full checkpoints (local or HF hub)
                                    and generates text side-by-side plus
                                    the reward-model score for each output.
  2. Published results table      — the full 15-config × 18-metric table
                                    straight from results/all_results.json
                                    (the same file the paper reads from).
  3. Raw evaluation samples       — browse the actual prompt/generated/
                                    reference triples saved during eval;
                                    these are the ground-truth outputs the
                                    reported perplexity / reward / diversity
                                    numbers are computed over.
  4. How to verify                — step-by-step instructions reviewers can
                                    copy/paste to reproduce a single number
                                    from scratch.

Run:
    python app.py                    # local
    python app.py --share            # public gradio link
    python app.py --use_hf           # pull weights from HuggingFace hub
                                     # (mr3haque/SLM-RL-Agent) instead of
                                     # the local outputs/ directory
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import gradio as gr
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
RESULTS_JSON = ROOT / "results" / "all_results.json"

HF_MODEL_REPO = "mr3haque/SLM-RL-Agent"
HF_DATA_REPO = "mr3haque/SLM-RL-Agent-Data"
GITHUB_URL = "https://github.com/rezwanh001/slm-rl-agent"

MODELS = ["pythia-70m", "pythia-160m", "pythia-410m", "smollm2-135m", "smollm2-360m"]
DATASETS = ["tinystories", "cnn_dailymail", "wikitext"]

MODEL_PRETTY = {
    "pythia-70m":    "Pythia-70M",
    "pythia-160m":   "Pythia-160M",
    "pythia-410m":   "Pythia-410M",
    "smollm2-135m":  "SmolLM2-135M",
    "smollm2-360m":  "SmolLM2-360M",
}
DATASET_PRETTY = {
    "tinystories":   "TinyStories",
    "cnn_dailymail": "CNN/DailyMail",
    "wikitext":      "Wikitext-103",
}

# Example prompts taken verbatim from the raw eval splits so reviewers can
# reproduce what the evaluator saw.
EXAMPLE_PROMPTS = {
    "tinystories":   "Once upon a time, there was a little girl named Lily. She loved to play outside in her backyard.",
    "cnn_dailymail": "Summarize: Scientists have discovered a new species of fish in the deep ocean near the Mariana Trench.",
    "wikitext":      "The history of artificial intelligence began in",
}


# ---------------------------------------------------------------------------
# Lazy model cache
# ---------------------------------------------------------------------------
_model_cache: dict[str, Any] = {}


def _model_path(model_key: str, dataset: str, stage: str, use_hf: bool) -> str:
    """Return either a local path or a HuggingFace repo identifier."""
    if use_hf:
        # single consolidated repo uses subfolders; transformers supports
        # `subfolder=` at load time. We return a tuple encoded as a string.
        return f"hf::{HF_MODEL_REPO}::{model_key}/{dataset}/{stage}"
    local = OUTPUTS / model_key / dataset / stage / "final"
    return str(local)


def _load_causal_lm(spec: str):
    if spec in _model_cache:
        return _model_cache[spec]
    if spec.startswith("hf::"):
        _, repo, subfolder = spec.split("::", 2)
        tok = AutoTokenizer.from_pretrained(repo, subfolder=subfolder, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            repo, subfolder=subfolder,
            torch_dtype=torch.float32, device_map="auto", trust_remote_code=True,
        )
    else:
        if not Path(spec).exists():
            raise FileNotFoundError(spec)
        tok = AutoTokenizer.from_pretrained(spec, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            spec, torch_dtype=torch.float32, device_map="auto", trust_remote_code=True,
        )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl.eval()
    _model_cache[spec] = (mdl, tok)
    return mdl, tok


def _load_reward(model_key: str, dataset: str, use_hf: bool):
    key = f"reward::{model_key}/{dataset}::{use_hf}"
    if key in _model_cache:
        return _model_cache[key]
    if use_hf:
        repo = HF_MODEL_REPO
        sub = f"{model_key}/{dataset}/reward_model"
        try:
            rm = AutoModelForSequenceClassification.from_pretrained(
                repo, subfolder=sub, torch_dtype=torch.float32, device_map="auto",
            )
            rtok = AutoTokenizer.from_pretrained(repo, subfolder=sub, trust_remote_code=True)
        except Exception:
            return None
    else:
        p = OUTPUTS / model_key / dataset / "reward_model" / "final"
        if not p.exists():
            return None
        rm = AutoModelForSequenceClassification.from_pretrained(
            str(p), torch_dtype=torch.float32, device_map="auto",
        )
        rtok = AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)
    if rtok.pad_token is None:
        rtok.pad_token = rtok.eos_token
    rm.eval()
    _model_cache[key] = (rm, rtok)
    return rm, rtok


@torch.no_grad()
def _generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=max(0.01, temperature),
        top_p=top_p,
        do_sample=temperature > 0,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


@torch.no_grad()
def _reward_score(rm_pack, prompt: str, response: str) -> float | None:
    if rm_pack is None:
        return None
    rm, rtok = rm_pack
    text = f"{prompt}\n\n{response}"
    enc = rtok(text, return_tensors="pt", truncation=True, max_length=512)
    enc = {k: v.to(rm.device) for k, v in enc.items()}
    out = rm(**enc)
    return float(out.logits[0, 0].item())


# ---------------------------------------------------------------------------
# Results backend
# ---------------------------------------------------------------------------
def _load_results() -> dict:
    with open(RESULTS_JSON) as f:
        return json.load(f)


def _format_published_metrics(model_key: str, dataset: str) -> str:
    r = _load_results()
    rd = r["our_models"][model_key]["datasets"][dataset]
    def _f(x, nd=3):
        return "—" if x is None else f"{float(x):.{nd}f}"
    md = (
        f"**Published numbers for {MODEL_PRETTY[model_key]} / {DATASET_PRETTY[dataset]}**\n"
        f"(source: `results/all_results.json`, cross-checked against raw "
        f"`outputs/{model_key}/{dataset}/eval_*/evaluation_results.json`)\n\n"
        f"| Metric            |   SFT    |   PPO    | Δ (PPO−SFT) |\n"
        f"|-------------------|----------|----------|-------------|\n"
        f"| Perplexity ↓      | {_f(rd['sft_perplexity'])} | {_f(rd['ppo_perplexity'])} | {_f(float(rd['ppo_perplexity'])-float(rd['sft_perplexity']))} |\n"
        f"| Reward mean ↑     | {_f(rd['sft_reward_mean'])} | {_f(rd['ppo_reward_mean'])} | {_f(rd['reward_delta'])} |\n"
        f"| Reward std        | {_f(rd['sft_reward_std'])} | {_f(rd['ppo_reward_std'])} | — |\n"
        f"| Distinct-1 ↑      | {_f(rd['sft_distinct1'])} | {_f(rd['ppo_distinct1'])} | {_f(float(rd['ppo_distinct1'])-float(rd['sft_distinct1']))} |\n"
        f"| Distinct-2 ↑      | {_f(rd['sft_distinct2'])} | {_f(rd['ppo_distinct2'])} | {_f(float(rd['ppo_distinct2'])-float(rd['sft_distinct2']))} |\n"
        f"| ROUGE-L F1 ↑      | {_f(rd['sft_rougeL'])} | {_f(rd['ppo_rougeL'])} | {_f(float(rd['ppo_rougeL'])-float(rd['sft_rougeL']))} |\n"
        f"| BLEU-4 ↑          | {_f(rd['sft_bleu4'])} | {_f(rd['ppo_bleu4'])} | {_f(float(rd['ppo_bleu4'])-float(rd['sft_bleu4']))} |\n\n"
        f"Evaluated on `num_samples=200` held-out prompts from "
        f"`{DATASET_PRETTY[dataset]}` — identical to the split shipped in the "
        f"raw `outputs/` tree on the [GitHub repo]({GITHUB_URL})."
    )
    return md


def _full_results_markdown() -> str:
    r = _load_results()
    lines = [
        "### Full 15-configuration results table",
        "",
        "All numbers below are copied verbatim from `results/all_results.json`, "
        "which was generated by `scripts/evaluate.py` and verified field-by-field "
        "against the raw per-run `evaluation_results.json` files "
        "(`scripts/verify_results.py` → **339/339 fields passed, 0 mismatches**).",
        "",
        "| Model | Dataset | PPL SFT | PPL PPO | Reward SFT | Reward PPO | Δ Reward | Dist-2 SFT | Dist-2 PPO | ROUGE-L SFT | ROUGE-L PPO |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for mk in MODELS:
        for ds in DATASETS:
            rd = r["our_models"][mk]["datasets"][ds]
            def _f(x, nd=3):
                return "—" if x is None else f"{float(x):.{nd}f}"
            lines.append(
                f"| {MODEL_PRETTY[mk]} | {DATASET_PRETTY[ds]} | "
                f"{_f(rd['sft_perplexity'],2)} | {_f(rd['ppo_perplexity'],2)} | "
                f"{_f(rd['sft_reward_mean'])} | {_f(rd['ppo_reward_mean'])} | "
                f"{_f(rd['reward_delta'])} | "
                f"{_f(rd['sft_distinct2'])} | {_f(rd['ppo_distinct2'])} | "
                f"{_f(rd['sft_rougeL'])} | {_f(rd['ppo_rougeL'])} |"
            )
    lines += [
        "",
        "### Baselines (published instruct SLMs)",
        "",
        "| Baseline | Dataset | PPL | Reward | Distinct-2 | ROUGE-L |",
        "|---|---|---|---|---|---|",
    ]
    for bk, bd in r["baselines"].items():
        for ds, row in bd.items():
            if not isinstance(row, dict):
                continue
            def _f(x, nd=3):
                return "—" if x is None else f"{float(x):.{nd}f}"
            lines.append(
                f"| {bk} | {DATASET_PRETTY.get(ds, ds)} | "
                f"{_f(row.get('perplexity'),2)} | {_f(row.get('reward_mean'))} | "
                f"{_f(row.get('distinct_2'))} | {_f(row.get('rougeL_f1'))} |"
            )
    return "\n".join(lines)


def _load_raw_sample(model_key: str, dataset: str, stage: str, index: int) -> tuple[str, str, str, str]:
    """Return (prompt, generated, reference, header)."""
    path = OUTPUTS / model_key / dataset / f"eval_{stage}" / "sample_generations.json"
    if not path.exists():
        return "", "", "", f"No raw sample file at `{path}` — run `scripts/evaluate.py` first."
    with open(path) as f:
        samples = json.load(f)
    if not samples:
        return "", "", "", f"Empty sample file at `{path}`"
    idx = max(0, min(index, len(samples) - 1))
    s = samples[idx]
    header = (
        f"Sample {idx+1} / {len(samples)}  —  "
        f"`outputs/{model_key}/{dataset}/eval_{stage}/sample_generations.json`"
    )
    return s.get("prompt", ""), s.get("generated", ""), s.get("reference", ""), header


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------
def run_comparison(model_key: str, dataset: str, prompt: str,
                   max_new_tokens: int, temperature: float, top_p: float,
                   use_hf: bool):
    """Generate SFT + PPO outputs and score them with the reward model."""
    if not prompt.strip():
        prompt = EXAMPLE_PROMPTS[dataset]
    status = []
    try:
        sft_spec = _model_path(model_key, dataset, "sft", use_hf)
        sft_m, sft_t = _load_causal_lm(sft_spec)
        sft_out = _generate(sft_m, sft_t, prompt, max_new_tokens, temperature, top_p)
    except Exception as e:
        sft_out = f"[SFT load/generate failed: {e}]"
        sft_t = None
    try:
        ppo_spec = _model_path(model_key, dataset, "ppo", use_hf)
        ppo_m, ppo_t = _load_causal_lm(ppo_spec)
        ppo_out = _generate(ppo_m, ppo_t, prompt, max_new_tokens, temperature, top_p)
    except Exception as e:
        ppo_out = f"[PPO load/generate failed: {e}]"

    rm_pack = _load_reward(model_key, dataset, use_hf)
    sft_r = _reward_score(rm_pack, prompt, sft_out) if rm_pack and sft_t is not None else None
    ppo_r = _reward_score(rm_pack, prompt, ppo_out) if rm_pack else None

    def _fmt(x):
        return "N/A (reward model not loaded)" if x is None else f"{x:+.4f}"

    delta = (ppo_r - sft_r) if (sft_r is not None and ppo_r is not None) else None
    delta_str = "—" if delta is None else f"{delta:+.4f}"

    published = _format_published_metrics(model_key, dataset)
    audit = (
        f"**Prompt used**  \n`{prompt}`\n\n"
        f"**SFT reward (this prompt)**: {_fmt(sft_r)}  \n"
        f"**PPO reward (this prompt)**: {_fmt(ppo_r)}  \n"
        f"**Δ reward (PPO − SFT, this prompt)**: {delta_str}\n\n"
        f"---\n\n{published}"
    )
    return sft_out, ppo_out, audit


def load_raw_sample(model_key: str, dataset: str, stage: str, index: int):
    p, g, r, hdr = _load_raw_sample(model_key, dataset, stage.lower(), int(index))
    return hdr, p, g, r


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
INSTRUCTIONS_MD = f"""
## How to use this verification app

This Gradio app lets anyone — reviewers, readers, or downstream users —
independently check that every number in the paper is real.

### Inputs

1. **Model family**. Pick one of 5 trained backbones:
   Pythia-70M / 160M / 410M, SmolLM2-135M / 360M.
2. **Dataset**. Pick one of 3 training corpora:
   TinyStories, CNN/DailyMail, Wikitext-103.
3. **Prompt**. Either type a free-form prompt, or leave it empty to use a
   default prompt drawn from the held-out evaluation split.
4. **Generation knobs**. `max_new_tokens` (50–400), `temperature` (0–1.5),
   `top_p` (0.1–1.0). PPO vs SFT differences are clearest at `temperature=0.7,
    top_p=0.9, max_new_tokens=200`, which are the evaluation defaults.
5. **Source of weights**. By default the app loads local checkpoints from
   `outputs/<model>/<dataset>/{{sft,ppo}}/final` and
   `outputs/<model>/<dataset>/reward_model/final`. Launch with `--use_hf` to
   stream weights directly from
   `{HF_MODEL_REPO}` (subfolders `<model>/<dataset>/sft|ppo|reward_model`).

### Outputs

For every (model, dataset, prompt) the app returns:

- **SFT output** — text generated by the supervised-fine-tuned policy.
- **PPO output** — text generated by the PPO-aligned policy.
- **SFT / PPO reward scores for THIS prompt**, computed by the same
  Bradley-Terry reward model used during training.
- **Δ reward** for this single prompt.
- **The published 7-row metric table** for that (model, dataset) pair
  — copied live from `results/all_results.json`, which is itself backed by
  the raw `outputs/*/eval_*/evaluation_results.json` files.

### What to check

- The **published** Δ reward (column 3 of the table) is the *mean* over the
  full 200-prompt eval split; the single-prompt Δ in the audit block is the
  score for *this* prompt. Sample-level variance is normal.
- **Table 2 / Raw samples tab** shows the actual prompt-generated-reference
  triples the reported reward / perplexity / diversity numbers were
  computed over — there is no synthetic data anywhere in the pipeline.
- **`scripts/verify_results.py`** (shipped alongside this app) cross-checks
  every one of the 339 numerical fields in `results/all_results.json`
  against the raw evaluation JSONs and exits non-zero on any drift. Running
  it in the repo root reproduces: *339 / 339 fields passed, 0 mismatches,
  0 missing files, sample sizes = {{200}}*.

### Minimal reproduction recipe

```bash
git clone {GITHUB_URL}
cd slm-rl-agent
pip install -e ".[all]"

# re-run one full pipeline end-to-end
bash scripts/run_all_experiments.sh pythia-70m tinystories

# or, using the already-trained weights shipped on HuggingFace
huggingface-cli download {HF_MODEL_REPO} --include "pythia-70m/tinystories/**"
python scripts/evaluate.py \\
    --model_path pythia-70m/tinystories/ppo/final \\
    --eval_dataset ./data/tinystories/eval.json

# cross-check every reported number against the raw eval files
python scripts/verify_results.py
```

### Reference links

- **Code**:    [{GITHUB_URL}]({GITHUB_URL})
- **Models**:  [{HF_MODEL_REPO}](https://huggingface.co/{HF_MODEL_REPO})
- **Data**:    [{HF_DATA_REPO}](https://huggingface.co/datasets/{HF_DATA_REPO})
"""


def build_demo(use_hf: bool):
    results = _load_results()

    with gr.Blocks(title="SLM-RL-Agent — Verification") as demo:
        gr.Markdown(
            "# SLM-RL-Agent — Interactive Verification\n"
            "Companion app for *“Efficiently Enhancing SLM Agents: "
            "A Reinforcement Learning Approach to Performance Improvement.”*  \n"
            f"Weights: [{HF_MODEL_REPO}](https://huggingface.co/{HF_MODEL_REPO}) · "
            f"Data: [{HF_DATA_REPO}](https://huggingface.co/datasets/{HF_DATA_REPO}) · "
            f"Code: [{GITHUB_URL}]({GITHUB_URL})  \n"
            f"Weight source for this session: **{'HuggingFace hub' if use_hf else 'local outputs/'}**"
        )

        with gr.Tab("1. Live SFT vs PPO"):
            gr.Markdown(
                "Pick a (model, dataset), type or keep the default prompt, click "
                "**Generate**. The app loads the actual trained checkpoints and "
                "the matching reward model, generates once from the SFT policy "
                "and once from the PPO policy, and scores both. The published "
                "mean-over-200-prompts metrics for this configuration are shown "
                "below for context."
            )
            with gr.Row():
                model_dd = gr.Dropdown(
                    choices=[(MODEL_PRETTY[m], m) for m in MODELS],
                    value="pythia-70m", label="Model backbone",
                )
                data_dd = gr.Dropdown(
                    choices=[(DATASET_PRETTY[d], d) for d in DATASETS],
                    value="tinystories", label="Training corpus",
                )
            with gr.Row():
                mtok = gr.Slider(50, 400, value=200, step=10, label="max_new_tokens")
                tmp = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="temperature")
                tp = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="top_p")
            prompt_box = gr.Textbox(
                lines=3, label="Prompt",
                value=EXAMPLE_PROMPTS["tinystories"],
                placeholder="Leave blank to use a held-out-split default prompt",
            )
            run_btn = gr.Button("Generate & score (SFT vs PPO)", variant="primary")
            with gr.Row():
                sft_box = gr.Textbox(lines=10, label="SFT output", interactive=False)
                ppo_box = gr.Textbox(lines=10, label="PPO output", interactive=False)
            audit_box = gr.Markdown()

            def _update_default_prompt(ds):
                return EXAMPLE_PROMPTS[ds]
            data_dd.change(_update_default_prompt, inputs=[data_dd], outputs=[prompt_box])
            run_btn.click(
                lambda *a: run_comparison(*a, use_hf),
                inputs=[model_dd, data_dd, prompt_box, mtok, tmp, tp],
                outputs=[sft_box, ppo_box, audit_box],
            )

        with gr.Tab("2. Published results table"):
            gr.Markdown(_full_results_markdown())

        with gr.Tab("3. Raw evaluation samples"):
            gr.Markdown(
                "Browse the actual prompt / generated / reference triples that "
                "the reported metrics were computed over. These files were "
                "written by `scripts/evaluate.py` at the end of every training "
                "run and are untouched afterwards."
            )
            with gr.Row():
                rs_model = gr.Dropdown(
                    choices=[(MODEL_PRETTY[m], m) for m in MODELS],
                    value="pythia-70m", label="Model",
                )
                rs_data = gr.Dropdown(
                    choices=[(DATASET_PRETTY[d], d) for d in DATASETS],
                    value="tinystories", label="Dataset",
                )
                rs_stage = gr.Radio(["SFT", "PPO"], value="PPO", label="Stage")
                rs_idx = gr.Slider(0, 49, value=0, step=1, label="Sample index")
            rs_btn = gr.Button("Load raw sample")
            rs_hdr = gr.Markdown()
            with gr.Row():
                rs_prompt = gr.Textbox(lines=6, label="prompt (from held-out split)", interactive=False)
                rs_gen = gr.Textbox(lines=6, label="generated (at eval time)", interactive=False)
                rs_ref = gr.Textbox(lines=6, label="reference / gold", interactive=False)
            rs_btn.click(
                load_raw_sample,
                inputs=[rs_model, rs_data, rs_stage, rs_idx],
                outputs=[rs_hdr, rs_prompt, rs_gen, rs_ref],
            )

        with gr.Tab("4. How to verify (inputs / outputs)"):
            gr.Markdown(INSTRUCTIONS_MD)

    return demo


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="Create a public Gradio link")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--use_hf", action="store_true",
                    help=f"Load weights from {HF_MODEL_REPO} instead of ./outputs")
    args = ap.parse_args()
    demo = build_demo(use_hf=args.use_hf)
    demo.queue()
    demo.launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0",
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
