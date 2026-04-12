#!/usr/bin/env python3
"""Generate publication-quality RLHF pipeline figure for the paper."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 4.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 4.5)
ax.axis('off')

# Colors
c_data = '#E8F4FD'
c_model = '#D4EDDA'
c_reward = '#FFF3CD'
c_ppo = '#F8D7DA'
c_arrow = '#495057'
c_border_data = '#2196F3'
c_border_model = '#28A745'
c_border_reward = '#FFC107'
c_border_ppo = '#DC3545'

def draw_box(x, y, w, h, label, sublabel, facecolor, edgecolor, fontsize=10):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                         facecolor=facecolor, edgecolor=edgecolor, linewidth=1.8)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2 + 0.15, label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color='#212529')
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.2, sublabel, ha='center', va='center',
                fontsize=7.5, color='#6C757D', style='italic')

def draw_arrow(x1, y1, x2, y2, label=None):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=c_arrow, lw=1.5))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2 + 0.2
        ax.text(mx, my, label, ha='center', va='center', fontsize=7,
                color=c_arrow, style='italic',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.8))

# Stage labels at top
ax.text(2.0, 4.2, 'Stage 1: SFT', ha='center', va='center', fontsize=11,
        fontweight='bold', color=c_border_model)
ax.text(6.5, 4.2, 'Stage 2: Reward Model', ha='center', va='center', fontsize=11,
        fontweight='bold', color=c_border_reward)
ax.text(11.5, 4.2, 'Stage 3: PPO', ha='center', va='center', fontsize=11,
        fontweight='bold', color=c_border_ppo)

# Vertical dashed lines separating stages
ax.plot([4.3, 4.3], [0.3, 4.0], '--', color='#CED4DA', lw=1.2)
ax.plot([8.8, 8.8], [0.3, 4.0], '--', color='#CED4DA', lw=1.2)

# Stage 1: SFT
draw_box(0.2, 2.5, 1.8, 1.0, 'Text Corpus', r'$\mathcal{D}_{SFT}$', c_data, c_border_data)
draw_box(0.2, 0.5, 1.8, 1.0, 'Base SLM', r'GPT-Neo / Pythia', c_model, c_border_model)
draw_box(2.6, 1.5, 1.5, 1.0, r'$\pi_{SFT}$', 'SFT Model', c_model, c_border_model)

draw_arrow(2.0, 3.0, 2.6, 2.15)
draw_arrow(2.0, 1.0, 2.6, 1.85)
ax.text(1.5, 1.85, r'$\mathcal{L}_{SFT}=-\sum_i \log P_\theta(x_i|x_{<i})$',
        ha='center', va='center', fontsize=6.5, color='#495057',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='#F8F9FA', edgecolor='#DEE2E6'))

# Stage 2: Reward Model
draw_box(4.6, 2.5, 2.0, 1.0, 'Preference Data', r'$(p, y_w, y_l)$', c_data, c_border_data)
draw_box(4.6, 0.5, 2.0, 1.0, r'$\pi_{SFT}$', 'Init from SFT', c_model, c_border_model)
draw_box(7.1, 1.5, 1.5, 1.0, r'$r_\phi$', 'Reward Model', c_reward, c_border_reward)

draw_arrow(4.1, 2.0, 4.6, 1.5)  # SFT model feeds into RM init
draw_arrow(6.6, 3.0, 7.1, 2.15)
draw_arrow(6.6, 1.0, 7.1, 1.85)
ax.text(5.8, 1.85, r'$\mathcal{L}_{RM}=-\log\sigma(r_w - r_l)$',
        ha='center', va='center', fontsize=6.5, color='#495057',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='#F8F9FA', edgecolor='#DEE2E6'))

# Stage 3: PPO
draw_box(9.1, 2.5, 1.8, 1.0, 'Prompts', r'$\mathcal{D}_p$', c_data, c_border_data)
draw_box(9.1, 0.5, 1.8, 1.0, r'$r_\phi$', 'Frozen Reward', c_reward, c_border_reward)
draw_box(11.5, 1.5, 1.5, 1.0, r'$\pi_{RL}$', 'Final Agent', c_ppo, c_border_ppo)

draw_arrow(8.6, 2.0, 9.1, 1.5)  # RM feeds into PPO reward
draw_arrow(10.9, 3.0, 11.5, 2.15)
draw_arrow(10.9, 1.0, 11.5, 1.85)

# PPO also uses SFT as reference
ax.annotate('', xy=(11.5, 1.3), xytext=(4.1, 1.3),
            arrowprops=dict(arrowstyle='->', color='#6C757D', lw=1.0,
                          linestyle='dashed', connectionstyle='arc3,rad=-0.15'))
ax.text(7.8, 0.2, r'KL ref: $\pi_{SFT}$', ha='center', va='center', fontsize=6.5,
        color='#6C757D', style='italic')

ax.text(10.5, 1.85, r'$J(\theta)=\mathbb{E}[r_\phi - \beta\cdot KL]$',
        ha='center', va='center', fontsize=6.5, color='#495057',
        bbox=dict(boxstyle='round,pad=0.15', facecolor='#F8F9FA', edgecolor='#DEE2E6'))

# Final output arrow
draw_arrow(13.0, 2.0, 13.8, 2.0)
ax.text(13.8, 2.0, 'Enhanced\nAgent', ha='left', va='center', fontsize=9,
        fontweight='bold', color=c_border_ppo)

plt.tight_layout()
plt.savefig('paper/figures/pipeline.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.savefig('paper/figures/pipeline.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
print("Pipeline figure saved to paper/figures/pipeline.{pdf,png}")
