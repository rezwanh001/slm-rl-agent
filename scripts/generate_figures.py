"""Generate publication-quality figures (PNG + PDF) for the SLM-RL-Agent paper.

All figures are built deterministically from results/all_results.json — the same
file verified by scripts/verify_results.py against the raw eval outputs. Outputs
are written to figures/ which is gitignored.

Figures produced:

    fig1_sft_perplexity_grid.{png,pdf}     — 5x3 heatmap of SFT perplexity
    fig2_ppo_reward_delta_bars.{png,pdf}   — 15-config grouped bar chart with error
    fig3_sft_vs_ppo_scatter.{png,pdf}      — SFT reward vs PPO reward scatter
    fig4_sota_comparison.{png,pdf}         — 360M class vs SOTA instruct baselines
    fig5_capacity_headroom.{png,pdf}       — PPL vs Δ-reward (capacity hypothesis)
    fig6_diversity_preservation.{png,pdf}  — SFT vs PPO Distinct-1 / Distinct-2
    fig7_win_rate_matrix.{png,pdf}         — analytical win-rate heatmap

Usage:
    python scripts/generate_figures.py [--outdir figures]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

ROOT = Path(__file__).resolve().parents[1]
RESULTS_JSON = ROOT / "results" / "all_results.json"

MODELS = [
    ("pythia-70m",   "Pythia-70M",   70),
    ("pythia-160m",  "Pythia-160M",  162),
    ("pythia-410m",  "Pythia-410M",  410),
    ("smollm2-135m", "SmolLM2-135M", 135),
    ("smollm2-360m", "SmolLM2-360M", 361),
]
DATASETS = [
    ("tinystories",   "TinyStories"),
    ("cnn_dailymail", "CNN/DailyMail"),
    ("wikitext",      "Wikitext-103"),
]
FAMILY_COLORS = {
    "Pythia":  "#1f77b4",
    "SmolLM2": "#d62728",
}
FAMILY_OF = {
    "pythia-70m":  "Pythia",
    "pythia-160m": "Pythia",
    "pythia-410m": "Pythia",
    "smollm2-135m":"SmolLM2",
    "smollm2-360m":"SmolLM2",
}

# Publication style — LaTeX-safe, serif, consistent line widths
mpl.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["DejaVu Serif", "Times New Roman", "Times"],
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        140,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.linewidth":    0.8,
    "grid.linewidth":    0.4,
    "lines.linewidth":   1.6,
    "pdf.fonttype":      42,   # TrueType in PDF, embeds cleanly
    "ps.fonttype":       42,
})


def _load() -> dict:
    with open(RESULTS_JSON) as f:
        return json.load(f)


def _save(fig, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.png")
    fig.savefig(outdir / f"{name}.pdf")
    plt.close(fig)
    print(f"  wrote {outdir / f'{name}.png'}  +  .pdf")


def _winrate(delta: float, s_sft: float, s_ppo: float) -> float:
    denom = math.sqrt((s_sft or 0.0) ** 2 + (s_ppo or 0.0) ** 2)
    if denom < 1e-9:
        return 50.0
    z = delta / denom
    return 50.0 * (1.0 + erf(z / math.sqrt(2.0)))


# --------------------------------------------------------------------------
# Figure 1 — SFT perplexity heatmap (5 models x 3 datasets)
# --------------------------------------------------------------------------

def fig1_sft_perplexity(agg: dict, outdir: Path) -> None:
    mat = np.zeros((len(MODELS), len(DATASETS)))
    for i, (mk, _, _) in enumerate(MODELS):
        for j, (dk, _) in enumerate(DATASETS):
            mat[i, j] = agg["our_models"][mk]["datasets"][dk]["sft_perplexity"]

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    im = ax.imshow(mat, cmap="YlGnBu_r", aspect="auto")

    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([d[1] for d in DATASETS])
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([m[1] for m in MODELS])
    ax.set_title("SFT perplexity across 5 models × 3 domains (lower is better)")

    for i in range(len(MODELS)):
        for j in range(len(DATASETS)):
            val = mat[i, j]
            color = "white" if val > 40 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    color=color, fontsize=10)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Perplexity")
    _save(fig, outdir, "fig1_sft_perplexity_grid")


# --------------------------------------------------------------------------
# Figure 2 — PPO Δ reward bar chart (15 configurations)
# --------------------------------------------------------------------------

def fig2_reward_delta(agg: dict, outdir: Path) -> None:
    labels, values, colors = [], [], []
    for mk, mlabel, _ in MODELS:
        for dk, dlabel in DATASETS:
            d = agg["our_models"][mk]["datasets"][dk]
            labels.append(f"{mlabel}\n{dlabel}")
            values.append(d["reward_delta"])
            colors.append(FAMILY_COLORS[FAMILY_OF[mk]])

    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    xs = np.arange(len(labels))
    bars = ax.bar(xs, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(r"$\Delta$ reward (PPO $-$ SFT)")
    ax.set_title(r"PPO reward gain over SFT across 15 configurations")
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Annotate bars
    for bar, v in zip(bars, values):
        y = v + (0.03 if v >= 0 else -0.05)
        ax.text(bar.get_x() + bar.get_width() / 2, y,
                f"{v:+.2f}", ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=7.5)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=FAMILY_COLORS["Pythia"],  label="Pythia"),
        Patch(color=FAMILY_COLORS["SmolLM2"], label="SmolLM2"),
    ], loc="upper left", frameon=True)

    ymax = max(values) + 0.3
    ymin = min(values) - 0.3
    ax.set_ylim(ymin, ymax)
    _save(fig, outdir, "fig2_ppo_reward_delta_bars")


# --------------------------------------------------------------------------
# Figure 3 — Scatter: SFT reward vs PPO reward (diagonal = no change)
# --------------------------------------------------------------------------

def fig3_scatter(agg: dict, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5.2))

    for mk, mlabel, _ in MODELS:
        color = FAMILY_COLORS[FAMILY_OF[mk]]
        for dk, dlabel in DATASETS:
            d = agg["our_models"][mk]["datasets"][dk]
            x = d["sft_reward_mean"]
            y = d["ppo_reward_mean"]
            marker = {"tinystories": "o", "cnn_dailymail": "s", "wikitext": "^"}[dk]
            ax.scatter(x, y, c=color, marker=marker, s=80,
                       edgecolor="black", linewidth=0.5, zorder=3)

    lo, hi = -10, 8
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="no change (SFT=PPO)")
    ax.fill_between([lo, hi], [lo, hi], hi, color="#b2df8a", alpha=0.2,
                    label="PPO improves")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("SFT reward (mean)")
    ax.set_ylabel("PPO reward (mean)")
    ax.set_title("SFT vs PPO reward, all 15 configurations")
    ax.grid(linestyle=":", alpha=0.6)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=FAMILY_COLORS["Pythia"],
               markersize=9, markeredgecolor="black", label="Pythia family"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=FAMILY_COLORS["SmolLM2"],
               markersize=9, markeredgecolor="black", label="SmolLM2 family"),
        Line2D([0], [0], marker="o", color="gray", linestyle="",
               markeredgecolor="black", label="TinyStories"),
        Line2D([0], [0], marker="s", color="gray", linestyle="",
               markeredgecolor="black", label="CNN/DailyMail"),
        Line2D([0], [0], marker="^", color="gray", linestyle="",
               markeredgecolor="black", label="Wikitext-103"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=True)
    _save(fig, outdir, "fig3_sft_vs_ppo_scatter")


# --------------------------------------------------------------------------
# Figure 4 — 360M-class SOTA comparison (perplexity + reward dual panel)
# --------------------------------------------------------------------------

def fig4_sota(agg: dict, outdir: Path) -> None:
    rows = [
        ("SmolLM2-360M-Instruct",    "baseline", "smollm2-360m-instruct"),
        ("Qwen2.5-0.5B-Instruct",    "baseline", "qwen25-05b-instruct"),
        ("SmolLM2-360M (ours, SFT)", "ours-sft", "smollm2-360m"),
        ("SmolLM2-360M (ours, PPO)", "ours-ppo", "smollm2-360m"),
    ]
    ds_keys = ["tinystories", "cnn_dailymail", "wikitext"]

    ppl = np.zeros((len(rows), 3))
    rwd = np.zeros((len(rows), 3))
    for i, (_, kind, key) in enumerate(rows):
        for j, ds in enumerate(ds_keys):
            if kind == "baseline":
                b = agg["baselines"][key][ds]
                ppl[i, j] = b["perplexity"]
                rwd[i, j] = b["reward_mean"]
            else:
                d = agg["our_models"][key]["datasets"][ds]
                field = "sft" if kind == "ours-sft" else "ppo"
                ppl[i, j] = d[f"{field}_perplexity"]
                rwd[i, j] = d[f"{field}_reward_mean"]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    xs = np.arange(len(ds_keys))
    width = 0.2
    colors = ["#999999", "#66a0c8", "#1f77b4", "#d62728"]

    # Left: perplexity (lower is better)
    ax = axes[0]
    for i, (label, _, _) in enumerate(rows):
        ax.bar(xs + i * width - 1.5 * width, ppl[i], width,
               label=label, color=colors[i], edgecolor="black", linewidth=0.4)
    ax.set_xticks(xs); ax.set_xticklabels([DATASETS[k][1] for k in range(3)])
    ax.set_ylabel("Perplexity (lower is better)")
    ax.set_title("Perplexity vs SOTA instruct baselines (360M class)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Right: reward mean (higher is better)
    ax = axes[1]
    for i, (label, _, _) in enumerate(rows):
        ax.bar(xs + i * width - 1.5 * width, rwd[i], width,
               label=label, color=colors[i], edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(xs); ax.set_xticklabels([DATASETS[k][1] for k in range(3)])
    ax.set_ylabel("Reward mean (higher is better)")
    ax.set_title("Reward vs SOTA instruct baselines (360M class)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle("SLM-RL-Agent 360M class vs published SOTA instruct SLMs", fontsize=12)
    _save(fig, outdir, "fig4_sota_comparison")


# --------------------------------------------------------------------------
# Figure 5 — Capacity-headroom hypothesis
# --------------------------------------------------------------------------

def fig5_capacity(agg: dict, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.6))

    for mk, mlabel, _ in MODELS:
        fam = FAMILY_OF[mk]
        color = FAMILY_COLORS[fam]
        for dk, dlabel in DATASETS:
            d = agg["our_models"][mk]["datasets"][dk]
            x = d["sft_perplexity"]
            y = d["reward_delta"]
            s = 120
            marker = {"tinystories": "o", "cnn_dailymail": "s", "wikitext": "^"}[dk]
            ax.scatter(x, y, c=color, marker=marker, s=s,
                       edgecolor="black", linewidth=0.5, zorder=3)
            # Annotate the positive-gain highlights
            if y > 0.2:
                ax.annotate(f"{mlabel}\n{dlabel}", (x, y),
                            textcoords="offset points", xytext=(6, 4),
                            fontsize=7.5)

    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("SFT perplexity (log scale) — proxy for headroom")
    ax.set_ylabel(r"PPO $\Delta$ reward")
    ax.set_title("Capacity-headroom hypothesis:\nPPO gain correlates with a fluent SFT prior, not raw parameters")
    ax.grid(linestyle=":", alpha=0.6)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=FAMILY_COLORS["Pythia"], markersize=9,
               markeredgecolor="black", label="Pythia family"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=FAMILY_COLORS["SmolLM2"], markersize=9,
               markeredgecolor="black", label="SmolLM2 family"),
        Line2D([0], [0], marker="o", color="gray", linestyle="",
               markeredgecolor="black", label="TinyStories"),
        Line2D([0], [0], marker="s", color="gray", linestyle="",
               markeredgecolor="black", label="CNN/DailyMail"),
        Line2D([0], [0], marker="^", color="gray", linestyle="",
               markeredgecolor="black", label="Wikitext-103"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    _save(fig, outdir, "fig5_capacity_headroom")


# --------------------------------------------------------------------------
# Figure 6 — Diversity preservation (Distinct-1 and Distinct-2)
# --------------------------------------------------------------------------

def fig6_diversity(agg: dict, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))

    labels = []
    d1_sft, d1_ppo, d2_sft, d2_ppo, col = [], [], [], [], []
    for mk, mlabel, _ in MODELS:
        for dk, dlabel in DATASETS:
            d = agg["our_models"][mk]["datasets"][dk]
            labels.append(f"{mlabel.split('-')[1]}/{dlabel[:4]}")
            d1_sft.append(d["sft_distinct1"])
            d1_ppo.append(d["ppo_distinct1"])
            d2_sft.append(d["sft_distinct2"])
            d2_ppo.append(d["ppo_distinct2"])
            col.append(FAMILY_COLORS[FAMILY_OF[mk]])

    xs = np.arange(len(labels))
    width = 0.38

    for ax, (sft, ppo, name) in zip(axes, [
        (d1_sft, d1_ppo, "Distinct-1"),
        (d2_sft, d2_ppo, "Distinct-2"),
    ]):
        ax.bar(xs - width/2, sft, width, color="#cccccc",
               edgecolor="black", linewidth=0.4, label="SFT")
        ax.bar(xs + width/2, ppo, width, color=col,
               edgecolor="black", linewidth=0.4, label="PPO")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7.5)
        ax.set_ylabel(name)
        ax.set_title(f"{name} — SFT vs PPO (15 configurations)")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Text diversity is preserved or improved after PPO alignment",
                 fontsize=12)
    _save(fig, outdir, "fig6_diversity_preservation")


# --------------------------------------------------------------------------
# Figure 7 — Analytical win-rate heatmap
# --------------------------------------------------------------------------

def fig7_winrate(agg: dict, outdir: Path) -> None:
    mat = np.zeros((len(MODELS), len(DATASETS)))
    for i, (mk, _, _) in enumerate(MODELS):
        for j, (dk, _) in enumerate(DATASETS):
            d = agg["our_models"][mk]["datasets"][dk]
            mat[i, j] = _winrate(d["reward_delta"],
                                  d["sft_reward_std"],
                                  d["ppo_reward_std"])

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=35, vmax=65, aspect="auto")

    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([d[1] for d in DATASETS])
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([m[1] for m in MODELS])
    ax.set_title("PPO win-rate vs SFT\n"
                 r"(analytical: $\Phi(\Delta / \sqrt{\sigma_{SFT}^{2} + \sigma_{PPO}^{2}})$)")

    for i in range(len(MODELS)):
        for j in range(len(DATASETS)):
            val = mat[i, j]
            txt = f"{val:.1f}%"
            ax.text(j, i, txt, ha="center", va="center",
                    color="black", fontsize=10,
                    weight="bold" if val >= 55 else "normal")

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("PPO win rate (%)")
    _save(fig, outdir, "fig7_win_rate_matrix")


# --------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=str(ROOT / "figures"))
    args = p.parse_args()
    outdir = Path(args.outdir)

    agg = _load()
    print(f"Loaded {RESULTS_JSON}")
    print(f"Writing to {outdir}")

    fig1_sft_perplexity(agg, outdir)
    fig2_reward_delta(agg, outdir)
    fig3_scatter(agg, outdir)
    fig4_sota(agg, outdir)
    fig5_capacity(agg, outdir)
    fig6_diversity(agg, outdir)
    fig7_winrate(agg, outdir)
    print("\nDone.")


if __name__ == "__main__":
    main()
