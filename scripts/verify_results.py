"""Cross-check results/all_results.json against raw outputs/*/eval_*/evaluation_results.json.

Prints a pass/fail per (model, dataset, stage) and exits non-zero on any mismatch.
Used as a sanity check to confirm aggregated numbers in the paper and HF model cards
are identical to the raw evaluation outputs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
RESULTS_JSON = ROOT / "results" / "all_results.json"

MODELS = ["pythia-70m", "pythia-160m", "pythia-410m", "smollm2-135m", "smollm2-360m"]
DATASETS = ["tinystories", "cnn_dailymail", "wikitext"]
BASELINES = ["qwen25-05b-instruct", "smollm2-135m-instruct", "smollm2-360m-instruct"]

# Map aggregated-json field -> (stage, raw metric name)
OUR_FIELDS = [
    ("sft_perplexity",  "sft", "perplexity"),
    ("sft_reward_mean", "sft", "reward_mean"),
    ("sft_reward_std",  "sft", "reward_std"),
    ("sft_distinct1",   "sft", "distinct_1"),
    ("sft_distinct2",   "sft", "distinct_2"),
    ("sft_rouge1",      "sft", "rouge1_f1"),
    ("sft_rouge2",      "sft", "rouge2_f1"),
    ("sft_rougeL",      "sft", "rougeL_f1"),
    ("sft_bleu4",       "sft", "bleu_4"),
    ("ppo_perplexity",  "ppo", "perplexity"),
    ("ppo_reward_mean", "ppo", "reward_mean"),
    ("ppo_reward_std",  "ppo", "reward_std"),
    ("ppo_distinct1",   "ppo", "distinct_1"),
    ("ppo_distinct2",   "ppo", "distinct_2"),
    ("ppo_rouge1",      "ppo", "rouge1_f1"),
    ("ppo_rouge2",      "ppo", "rouge2_f1"),
    ("ppo_rougeL",      "ppo", "rougeL_f1"),
    ("ppo_bleu4",       "ppo", "bleu_4"),
]

BASELINE_FIELDS = [
    ("perplexity",   "perplexity"),
    ("reward_mean",  "reward_mean"),
    ("distinct_1",   "distinct_1"),
    ("distinct_2",   "distinct_2"),
    ("rouge1_f1",    "rouge1_f1"),
    ("rougeL_f1",    "rougeL_f1"),
]

TOL = 1e-2  # tolerance for rounding (aggregator rounds to 2–4 decimals)


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def near(a, b, tol=TOL) -> bool:
    if a is None or b is None:
        return a == b
    return abs(float(a) - float(b)) <= tol


def main() -> int:
    agg = load(RESULTS_JSON)
    ours = agg["our_models"]
    base = agg["baselines"]

    total = 0
    ok = 0
    missing_files = []
    mismatches = []
    sample_sizes: set[int] = set()

    print("=" * 80)
    print("Cross-check: results/all_results.json  vs  outputs/*/eval_*/evaluation_results.json")
    print("=" * 80)

    # --- Our models ---
    for mk in MODELS:
        for ds in DATASETS:
            rd = ours[mk]["datasets"][ds]
            for stage in ("sft", "ppo"):
                raw_path = OUTPUTS / mk / ds / f"eval_{stage}" / "evaluation_results.json"
                if not raw_path.exists():
                    missing_files.append(str(raw_path))
                    print(f"[MISSING] {mk}/{ds}/{stage}")
                    continue
                raw = load(raw_path)
                sample_sizes.add(raw.get("num_samples", -1))
                raw_m = raw["metrics"]
                row_ok = True
                for agg_field, expected_stage, raw_key in OUR_FIELDS:
                    if expected_stage != stage:
                        continue
                    total += 1
                    if near(rd.get(agg_field), raw_m.get(raw_key)):
                        ok += 1
                    else:
                        row_ok = False
                        mismatches.append(
                            f"  {mk}/{ds}/{stage}: {agg_field}  agg={rd.get(agg_field)}  raw={raw_m.get(raw_key)}"
                        )
                marker = "PASS" if row_ok else "FAIL"
                print(f"[{marker}] {mk:14s} {ds:14s} {stage.upper():3s}  "
                      f"(raw PPL={raw_m['perplexity']:.3f}, reward={raw_m['reward_mean']:+.3f})")

            # Also verify the reward_delta is ppo - sft
            sft_rm = load(OUTPUTS / mk / ds / "eval_sft" / "evaluation_results.json")["metrics"]["reward_mean"]
            ppo_rm = load(OUTPUTS / mk / ds / "eval_ppo" / "evaluation_results.json")["metrics"]["reward_mean"]
            expected_delta = ppo_rm - sft_rm
            total += 1
            if near(rd.get("reward_delta"), expected_delta):
                ok += 1
            else:
                mismatches.append(
                    f"  {mk}/{ds}: reward_delta  agg={rd.get('reward_delta')}  expected={expected_delta:.4f}"
                )

    # --- Baselines ---
    for bk in BASELINES:
        for ds in DATASETS:
            raw_path = OUTPUTS / "baselines" / bk / ds / "evaluation_results.json"
            if not raw_path.exists():
                missing_files.append(str(raw_path))
                print(f"[MISSING] baseline {bk}/{ds}")
                continue
            raw = load(raw_path)
            sample_sizes.add(raw.get("num_samples", -1))
            raw_m = raw["metrics"]
            agg_m = base[bk][ds]
            row_ok = True
            for agg_field, raw_key in BASELINE_FIELDS:
                total += 1
                if near(agg_m.get(agg_field), raw_m.get(raw_key)):
                    ok += 1
                else:
                    row_ok = False
                    mismatches.append(
                        f"  baseline {bk}/{ds}: {agg_field}  agg={agg_m.get(agg_field)}  raw={raw_m.get(raw_key)}"
                    )
            marker = "PASS" if row_ok else "FAIL"
            print(f"[{marker}] baseline {bk:25s} {ds:14s}    "
                  f"(raw PPL={raw_m['perplexity']:.3f}, reward={raw_m['reward_mean']:+.3f})")

    print()
    print("-" * 80)
    print(f"Checked fields : {total}")
    print(f"Passed         : {ok}")
    print(f"Mismatches     : {len(mismatches)}")
    print(f"Missing files  : {len(missing_files)}")
    print(f"Eval sample sizes observed in raw files : {sorted(sample_sizes)}")
    print("-" * 80)

    if mismatches:
        print("\nMISMATCHES:")
        for m in mismatches:
            print(m)
        return 1
    if missing_files:
        print("\nMISSING FILES:")
        for m in missing_files:
            print(m)
        return 2
    print("\nOK: every number in results/all_results.json is backed by a raw eval file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
