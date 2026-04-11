#!/usr/bin/env python3
"""Aggregate all SFT/PPO/baseline evaluation results into a single JSON table.

Reads outputs/{model}/{dataset}/eval_{sft,ppo}/evaluation_results.json for
every trained config, plus outputs/baselines/{name}/{dataset}/evaluation_results.json
for each external SOTA baseline, and writes results/all_results.json.
"""

import json
import os
from pathlib import Path

REPO = Path(__file__).parent.parent
OUT = REPO / "outputs"
RESULTS_DIR = REPO / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODELS = ["pythia-70m", "pythia-160m", "pythia-410m", "smollm2-135m", "smollm2-360m"]
DATASETS = ["tinystories", "cnn_dailymail", "wikitext"]

PARAMS_M = {
    "pythia-70m": 70,
    "pythia-160m": 162,
    "pythia-410m": 410,
    "smollm2-135m": 135,
    "smollm2-360m": 361,
}


def load(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f).get("metrics", {})


def main():
    all_results = {}
    for model in MODELS:
        all_results[model] = {"params_m": PARAMS_M[model], "datasets": {}}
        for ds in DATASETS:
            sft = load(OUT / model / ds / "eval_sft" / "evaluation_results.json")
            ppo = load(OUT / model / ds / "eval_ppo" / "evaluation_results.json")
            entry = {}
            if sft is not None:
                entry.update({
                    "sft_perplexity": round(sft.get("perplexity", 0.0), 2),
                    "sft_reward_mean": round(sft.get("reward_mean", 0.0), 4),
                    "sft_reward_std": round(sft.get("reward_std", 0.0), 4),
                    "sft_distinct1": round(sft.get("distinct_1", 0.0), 4),
                    "sft_distinct2": round(sft.get("distinct_2", 0.0), 4),
                    "sft_rouge1": round(sft.get("rouge1_f1", 0.0), 4),
                    "sft_rouge2": round(sft.get("rouge2_f1", 0.0), 4),
                    "sft_rougeL": round(sft.get("rougeL_f1", 0.0), 4),
                    "sft_bleu4": round(sft.get("bleu_4", 0.0), 4),
                })
            if ppo is not None:
                entry.update({
                    "ppo_perplexity": round(ppo.get("perplexity", 0.0), 2),
                    "ppo_reward_mean": round(ppo.get("reward_mean", 0.0), 4),
                    "ppo_reward_std": round(ppo.get("reward_std", 0.0), 4),
                    "ppo_distinct1": round(ppo.get("distinct_1", 0.0), 4),
                    "ppo_distinct2": round(ppo.get("distinct_2", 0.0), 4),
                    "ppo_rouge1": round(ppo.get("rouge1_f1", 0.0), 4),
                    "ppo_rouge2": round(ppo.get("rouge2_f1", 0.0), 4),
                    "ppo_rougeL": round(ppo.get("rougeL_f1", 0.0), 4),
                    "ppo_bleu4": round(ppo.get("bleu_4", 0.0), 4),
                })
                if sft is not None:
                    entry["reward_delta"] = round(
                        entry["ppo_reward_mean"] - entry["sft_reward_mean"], 4
                    )
            all_results[model]["datasets"][ds] = entry

    # External SOTA baselines (instruct-tuned, scored with our reward models)
    baselines = {}
    base_dir = OUT / "baselines"
    if base_dir.exists():
        for name_dir in sorted(base_dir.iterdir()):
            if not name_dir.is_dir():
                continue
            baselines[name_dir.name] = {}
            for ds_dir in sorted(name_dir.iterdir()):
                if not ds_dir.is_dir():
                    continue
                m = load(ds_dir / "evaluation_results.json")
                if m is None:
                    continue
                baselines[name_dir.name][ds_dir.name] = {
                    "perplexity": round(m.get("perplexity", 0.0), 2),
                    "reward_mean": round(m.get("reward_mean", 0.0), 4),
                    "distinct_1": round(m.get("distinct_1", 0.0), 4),
                    "distinct_2": round(m.get("distinct_2", 0.0), 4),
                    "rouge1_f1": round(m.get("rouge1_f1", 0.0), 4),
                    "rougeL_f1": round(m.get("rougeL_f1", 0.0), 4),
                }

    output = {"our_models": all_results, "baselines": baselines}
    out_path = RESULTS_DIR / "all_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {out_path}")

    # Also print a compact summary table
    print("\n=== Our Models (SFT / PPO) ===")
    print(f"{'Model':<14}{'Dataset':<16}{'SFT_PPL':>9}{'PPO_PPL':>9}{'SFT_R':>8}{'PPO_R':>8}{'ΔR':>8}")
    for model in MODELS:
        for ds in DATASETS:
            e = all_results[model]["datasets"].get(ds, {})
            if not e:
                continue
            print(f"{model:<14}{ds:<16}"
                  f"{e.get('sft_perplexity', 0):>9.2f}"
                  f"{e.get('ppo_perplexity', 0):>9.2f}"
                  f"{e.get('sft_reward_mean', 0):>8.3f}"
                  f"{e.get('ppo_reward_mean', 0):>8.3f}"
                  f"{e.get('reward_delta', 0):>8.3f}")

    if baselines:
        print("\n=== External SOTA Baselines ===")
        for name, by_ds in baselines.items():
            for ds, m in by_ds.items():
                print(f"{name:<32}{ds:<16}PPL={m['perplexity']:7.2f}  R={m['reward_mean']:7.3f}")


if __name__ == "__main__":
    main()
