"""
Generate pipeline diagram - Muted Matte Colors
Windows compatible - no special characters
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

os.makedirs('figures', exist_ok=True)

# Muted matte color palette
PALETTE = {
    'primary': '#2d3142',
    'secondary': '#4f5d75',
    'stage1': '#6c8ea0',
    'stage2': '#c1666b',
    'stage3': '#d4a574',
    'stage4': '#8b9bab',
    'stage5': '#90a955',
    'background': '#e8e9ed',
    'accent': '#bfc0c0',
}

fig, ax = plt.subplots(figsize=(12, 11))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')
fig.patch.set_facecolor('white')

def add_box(ax, x, y, width, height, text, color, fontsize=11):
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.12",
        facecolor=color,
        edgecolor=PALETTE['primary'],
        linewidth=2.5,
        alpha=0.9
    )
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center',
            fontsize=fontsize, fontweight='bold',
            color='white')

def add_arrow(ax, x1, y1, x2, y2, label=''):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.5,head_length=0.9',
        linewidth=2.5,
        color=PALETTE['secondary'],
        zorder=1
    )
    ax.add_patch(arrow)
    
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.6, mid_y, label,
                fontsize=9, style='italic', color=PALETTE['primary'],
                bbox=dict(boxstyle='round', facecolor='white', 
                         edgecolor=PALETTE['accent'], alpha=0.95, linewidth=1.5))

# Title
ax.text(5, 13.5, 'Seven-Stage Cascaded Pipeline Architecture',
        ha='center', fontsize=15, fontweight='bold', color=PALETTE['primary'])

# Stage 1
add_box(ax, 3.5, 12, 3, 0.8, 'Stage 1: Video Ingestion', PALETTE['stage1'], 11)
ax.text(8.2, 12.4, '600 frames\n(100ms sampling)', fontsize=8, 
        style='italic', color=PALETTE['secondary'])
add_arrow(ax, 5, 12, 5, 11.2)

# Stage 2
add_box(ax, 3.5, 10.4, 3, 0.8, 'Stage 2: Scene Detection', PALETTE['stage2'], 11)
ax.text(8.2, 10.8, '~44 scenes', fontsize=8, 
        style='italic', color=PALETTE['secondary'])
add_arrow(ax, 5, 10.4, 5, 9.6)

# Stage 3
add_box(ax, 3.5, 8.8, 3, 0.8, 'Stage 3: Candidate Extraction', PALETTE['stage2'], 11)
ax.text(8.2, 9.2, '68 candidates\n(84% reduction)', fontsize=8, 
        style='italic', color=PALETTE['secondary'])
add_arrow(ax, 5, 8.8, 5, 8.0)

# Stage 4: Hierarchical Deduplication
dedup_box = FancyBboxPatch(
    (1.3, 5.4), 7.4, 2.4,
    boxstyle="round,pad=0.18",
    facecolor=PALETTE['background'],
    edgecolor=PALETTE['stage3'],
    linewidth=3.5,
    alpha=0.5
)
ax.add_patch(dedup_box)
ax.text(5, 7.6, 'Stage 4: Hierarchical Deduplication Cascade', 
        ha='center', fontsize=12, fontweight='bold', color=PALETTE['primary'])

# Sub-tiers
add_box(ax, 1.9, 6.65, 1.9, 0.65, 'Tier 1\npHash', PALETTE['stage3'], 10)
ax.text(4.2, 7.0, '> 46\n(-31%)', fontsize=8, fontweight='bold', color=PALETTE['primary'])

add_box(ax, 4.05, 6.65, 1.9, 0.65, 'Tier 2\nSSIM', PALETTE['stage3'], 10)
ax.text(6.3, 7.0, '> 45\n(-3%)', fontsize=8, fontweight='bold', color=PALETTE['primary'])

add_box(ax, 6.2, 6.65, 1.9, 0.65, 'Tier 3\nCLIP', PALETTE['stage3'], 10)
ax.text(8.5, 7.0, '> 14\n(-69%)', fontsize=8, fontweight='bold', color=PALETTE['primary'])

# Complexity notes
ax.text(2.8, 6.05, 'O(n^2)\n~2ms/frame', fontsize=7, style='italic', 
        ha='center', color=PALETTE['secondary'], alpha=0.8)
ax.text(4.95, 6.05, 'O(n^2*w*h)\n~50ms/comp', fontsize=7, style='italic',
        ha='center', color=PALETTE['secondary'], alpha=0.8)
ax.text(7.1, 6.05, 'O(n*d+n^2)\n~100ms/frame', fontsize=7, style='italic',
        ha='center', color=PALETTE['secondary'], alpha=0.8)

add_arrow(ax, 5, 5.4, 5, 4.65)

# Stage 5: Audio
add_box(ax, 0.4, 3.85, 2.1, 0.65, 'Stage 5:\nAudio Events', PALETTE['stage4'], 10)
add_arrow(ax, 2.5, 4.17, 3.4, 4.17)

# Stage 6
add_box(ax, 3.5, 3.85, 3, 0.8, 'Stage 6: Representative Selection', PALETTE['stage4'], 11)
ax.text(8.2, 4.25, '10 frames\n(-26%)', fontsize=8, 
        style='italic', color=PALETTE['secondary'])
ax.text(5, 3.4, 'K-means clustering + Importance scoring', 
        ha='center', fontsize=8, style='italic', color=PALETTE['secondary'], alpha=0.8)
add_arrow(ax, 5, 3.85, 5, 3.05)

# Stage 7
add_box(ax, 3.5, 2.25, 3, 0.8, 'Stage 7: VLM Extraction', PALETTE['stage5'], 11)
ax.text(8.2, 2.65, 'Structured\nJSON output', fontsize=8, 
        style='italic', color=PALETTE['secondary'])
add_arrow(ax, 5, 2.25, 5, 1.4)

# Output
output_box = FancyBboxPatch(
    (2.8, 0.4), 4.4, 0.95,
    boxstyle="round,pad=0.12",
    facecolor='white',
    edgecolor=PALETTE['stage5'],
    linewidth=3,
    linestyle='--',
    alpha=0.95
)
ax.add_patch(output_box)
ax.text(5, 1.0, 'Structured Advertisement Analysis',
        ha='center', fontsize=11, fontweight='bold', color=PALETTE['primary'])
ax.text(5, 0.65, 'Brand, Message, Creative, Audience, Persuasion',
        ha='center', fontsize=9, style='italic', color=PALETTE['secondary'])

# Reduction box
reduction_box = FancyBboxPatch(
    (9.0, 6.5), 0.9, 1.3,
    boxstyle="round,pad=0.08",
    facecolor=PALETTE['stage5'],
    edgecolor=PALETTE['primary'],
    linewidth=2.5
)
ax.add_patch(reduction_box)
ax.text(9.45, 7.5, '84.5%', ha='center', fontsize=13, fontweight='bold', color='white')
ax.text(9.45, 7.1, 'Avg', ha='center', fontsize=9, color='white')
ax.text(9.45, 6.8, 'Reduction', ha='center', fontsize=8, color='white')

# Cost box - using text instead of arrow symbol
cost_box = FancyBboxPatch(
    (9.0, 2.5), 0.9, 1.5,
    boxstyle="round,pad=0.08",
    facecolor=PALETTE['stage1'],
    edgecolor=PALETTE['primary'],
    linewidth=2.5
)
ax.add_patch(cost_box)
ax.text(9.45, 3.7, '$4.50', ha='center', fontsize=10, fontweight='bold', color='white')
ax.text(9.45, 3.4, 'to', ha='center', fontsize=9, color='white')
ax.text(9.45, 3.05, '$0.08', ha='center', fontsize=11, fontweight='bold', color='white')
ax.text(9.45, 2.65, '98.3%', ha='center', fontsize=9, fontweight='bold', color='white')
ax.text(9.45, 2.4, 'Savings', ha='center', fontsize=8, color='white')

try:
    plt.tight_layout()
except:
    pass

plt.savefig('figures/pipeline_diagram.png', bbox_inches='tight', dpi=300, facecolor='white')
print("✓ Saved: figures/pipeline_diagram.png")
plt.close()