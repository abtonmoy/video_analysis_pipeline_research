"""
Generate all statistical figures - Windows Compatible
Muted matte colors, no font issues
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('figures', exist_ok=True)

# Muted matte palette
PALETTE = {
    'primary': '#2d3142',
    'secondary': '#4f5d75',
    'accent1': '#bfc0c0',
    'accent2': '#ef8354',
    'accent3': '#a8dadc',
    'success': '#90a955',
    'warning': '#d4a574',
    'error': '#c1666b',
}

CASCADE_COLORS = ['#6c7b8b', '#8b9bab', '#a8b5c7', '#c4d1e0', '#90a955']
ADAPTIVE_COLORS = ['#6c8ea0', '#d4a574', '#90a955', '#c1666b']
COST_COLORS = ['#c1666b', '#d4a574', '#90a955']

plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5

def safe_layout():
    try:
        plt.tight_layout()
    except:
        pass

# Figure 1: Cascade
def generate_cascade_figure():
    stages = ['Candidates', 'After\npHash', 'After\nSSIM', 'After\nCLIP', 'Final']
    means = [67.65, 46.48, 45.22, 14.13, 10.48]
    stds = [56.31, 41.16, 39.95, 13.78, 9.72]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(stages)), means, yerr=stds, 
                   color=CASCADE_COLORS, edgecolor=PALETTE['primary'], 
                   linewidth=2, capsize=6, alpha=0.85,
                   error_kw={'linewidth': 2, 'ecolor': PALETTE['secondary']})
    
    for i in range(1, len(means)):
        reduction = (means[i-1] - means[i]) / means[i-1] * 100
        y_pos = max(means[i-1], means[i]) + 15
        ax.annotate(f'-{reduction:.1f}%', 
                   xy=(i-0.5, y_pos), xytext=(i, y_pos + 8),
                   ha='center', fontsize=10, fontweight='bold',
                   color=PALETTE['accent2'],
                   arrowprops=dict(arrowstyle='->', lw=2, color=PALETTE['accent2']))
    
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 3,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=10, 
                fontweight='bold', color=PALETTE['primary'])
    
    ax.set_ylabel('Number of Frames', fontsize=13, fontweight='bold', color=PALETTE['primary'])
    ax.set_xlabel('Pipeline Stage', fontsize=13, fontweight='bold', color=PALETTE['primary'])
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=10)
    ax.set_ylim(0, 140)
    ax.grid(axis='y', alpha=0.2, linestyle='--', color=PALETTE['secondary'])
    ax.set_title('Stage-by-Stage Frame Reduction (n=23 videos)', 
                 fontsize=14, fontweight='bold', pad=15, color=PALETTE['primary'])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(PALETTE['secondary'])
    ax.spines['bottom'].set_color(PALETTE['secondary'])
    ax.tick_params(colors=PALETTE['primary'])
    
    ax.text(0.98, 0.97, 'Error bars: std deviation', transform=ax.transAxes,
            fontsize=9, style='italic', ha='right', va='top', color=PALETTE['primary'],
            bbox=dict(boxstyle='round', facecolor='white', 
                     edgecolor=PALETTE['accent1'], alpha=0.9, linewidth=1.5))
    
    safe_layout()
    plt.savefig('figures/reduction_cascade.png', bbox_inches='tight', dpi=300, facecolor='white')
    print("✓ reduction_cascade.png")
    plt.close()

# Figure 2: Adaptive
def generate_adaptive_figure():
    videos = ['Bernie\nSanders\n(60s)', 'Burger\nKing\n(73s)', 'A&F\nHoodie\n(14s)', 'Blackjack\nGame\n(20s)']
    reductions = [84.2, 82.6, 0.0, 96.8]
    candidates = [279, 149, 13, 95]
    final = [44, 26, 13, 3]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    bars1 = ax1.bar(videos, reductions, color=ADAPTIVE_COLORS, 
                    edgecolor=PALETTE['primary'], linewidth=2, alpha=0.85)
    bars1[2].set_edgecolor(PALETTE['success'])
    bars1[2].set_linewidth(3.5)
    bars1[3].set_edgecolor(PALETTE['error'])
    bars1[3].set_linewidth(3.5)
    
    for i, (bar, reduction) in enumerate(zip(bars1, reductions)):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                f'{reduction}%', ha='center', va='bottom', fontsize=12, 
                fontweight='bold', color=PALETTE['primary'])
    
    ax1.axhline(y=84.5, color=PALETTE['secondary'], linestyle='--', 
                linewidth=2.5, alpha=0.7, label='Mean: 84.5%')
    ax1.set_ylabel('Reduction Rate (%)', fontsize=13, fontweight='bold', color=PALETTE['primary'])
    ax1.set_xlabel('Video Advertisement', fontsize=13, fontweight='bold', color=PALETTE['primary'])
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.2, linestyle='--', color=PALETTE['secondary'])
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.95,
              edgecolor=PALETTE['accent1'], fancybox=True)
    ax1.set_title('Content-Aware Adaptive Reduction', 
                  fontsize=14, fontweight='bold', pad=15, color=PALETTE['primary'])
    
    ax1.annotate('', xy=(2, 0), xytext=(3, 0),
                arrowprops=dict(arrowstyle='<->', lw=2.5, color=PALETTE['accent2']))
    ax1.text(2.5, -8, '96.8% Adaptive Range', ha='center', 
            fontsize=10, fontweight='bold', color=PALETTE['accent2'])
    
    for spine in ['top', 'right']:
        ax1.spines[spine].set_visible(False)
    ax1.spines['left'].set_color(PALETTE['secondary'])
    ax1.spines['bottom'].set_color(PALETTE['secondary'])
    ax1.tick_params(colors=PALETTE['primary'])
    
    x_pos = np.arange(len(videos))
    width = 0.35
    
    bars2a = ax2.bar(x_pos - width/2, candidates, width, 
                     label='Candidates', color=PALETTE['accent2'], 
                     edgecolor=PALETTE['primary'], linewidth=2, alpha=0.85)
    bars2b = ax2.bar(x_pos + width/2, final, width, 
                     label='Final Selected', color=PALETTE['success'],
                     edgecolor=PALETTE['primary'], linewidth=2, alpha=0.85)
    
    for bars in [bars2a, bars2b]:
        for bar in bars:
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                    f'{int(bar.get_height())}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold', color=PALETTE['primary'])
    
    ax2.set_ylabel('Number of Frames', fontsize=13, fontweight='bold', color=PALETTE['primary'])
    ax2.set_xlabel('Video Advertisement', fontsize=13, fontweight='bold', color=PALETTE['primary'])
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(videos, fontsize=10)
    ax2.set_ylim(0, 310)
    ax2.legend(loc='upper right', fontsize=11, framealpha=0.95,
              edgecolor=PALETTE['accent1'], fancybox=True)
    ax2.grid(axis='y', alpha=0.2, linestyle='--', color=PALETTE['secondary'])
    ax2.set_title('Frame Counts: Before vs After', 
                  fontsize=14, fontweight='bold', pad=15, color=PALETTE['primary'])
    
    for spine in ['top', 'right']:
        ax2.spines[spine].set_visible(False)
    ax2.spines['left'].set_color(PALETTE['secondary'])
    ax2.spines['bottom'].set_color(PALETTE['secondary'])
    ax2.tick_params(colors=PALETTE['primary'])
    
    safe_layout()
    plt.savefig('figures/adaptive_comparison.png', bbox_inches='tight', dpi=300, facecolor='white')
    print("✓ adaptive_comparison.png")
    plt.close()

# Figure 3: Cost - FIXED for Windows
def generate_cost_figure():
    approaches = ['Dense\nSampling\n(100ms)', 'Moderate\nSampling\n(250ms)', 'Our\nPipeline']
    costs = [4.50, 1.80, 0.08]
    
    fig = plt.figure(figsize=(14, 6))
    ax1 = plt.subplot(1, 2, 1)
    
    bars1 = ax1.bar(approaches, costs, color=COST_COLORS, 
                    edgecolor=PALETTE['primary'], linewidth=2, alpha=0.85)
    bars1[2].set_edgecolor(PALETTE['success'])
    bars1[2].set_linewidth(3.5)
    
    for i, (bar, cost) in enumerate(zip(bars1, costs)):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.15,
                f'${cost:.2f}', ha='center', va='bottom', fontsize=13, 
                fontweight='bold', color=PALETTE['primary'])
    
    ax1.annotate('', xy=(2, 4.50), xytext=(2, 0.08),
                arrowprops=dict(arrowstyle='<->', lw=2.5, color=PALETTE['accent2']))
    ax1.text(2.35, 2.3, '98.3%\nSavings', ha='left', 
            fontsize=11, fontweight='bold', color=PALETTE['primary'],
            bbox=dict(boxstyle='round,pad=0.5', facecolor=PALETTE['accent3'], 
                     edgecolor=PALETTE['primary'], linewidth=2, alpha=0.9))
    
    ax1.set_ylabel('Cost per Video ($)', fontsize=13, fontweight='bold', color=PALETTE['primary'])
    ax1.set_xlabel('Sampling Approach', fontsize=13, fontweight='bold', color=PALETTE['primary'])
    ax1.set_ylim(0, 5.0)
    ax1.grid(axis='y', alpha=0.2, linestyle='--', color=PALETTE['secondary'])
    ax1.set_title('API Cost Comparison (Gemini 2.0 Flash)', 
                  fontsize=14, fontweight='bold', pad=15, color=PALETTE['primary'])
    
    for spine in ['top', 'right']:
        ax1.spines[spine].set_visible(False)
    ax1.spines['left'].set_color(PALETTE['secondary'])
    ax1.spines['bottom'].set_color(PALETTE['secondary'])
    ax1.tick_params(colors=PALETTE['primary'])
    
    # Second subplot
    ax2 = plt.subplot(1, 2, 2)
    
    videos_scale = np.array([10, 100, 1000])
    dense_costs = np.array([45, 450, 4500])
    pipeline_costs = np.array([0.8, 8, 80])
    
    ax2.plot(videos_scale, dense_costs, 'o-', linewidth=3, markersize=10,
            color=PALETTE['error'], label='Dense Sampling', alpha=0.85)
    ax2.plot(videos_scale, pipeline_costs, 's-', linewidth=3, markersize=10,
            color=PALETTE['success'], label='Our Pipeline', alpha=0.85)
    
    ax2.fill_between(videos_scale, dense_costs, pipeline_costs, 
                     alpha=0.15, color=PALETTE['success'])
    
    ax2.annotate('Save $4,420\nat 1000 videos', 
                xy=(1000, 2290), xytext=(400, 3500),
                fontsize=11, fontweight='bold', color=PALETTE['primary'],
                bbox=dict(boxstyle='round,pad=0.5', facecolor=PALETTE['accent3'],
                         edgecolor=PALETTE['primary'], linewidth=2, alpha=0.9),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=PALETTE['accent2']))
    
    ax2.set_xlabel('Number of Videos', fontsize=13, fontweight='bold', color=PALETTE['primary'])
    ax2.set_ylabel('Total Cost ($)', fontsize=13, fontweight='bold', color=PALETTE['primary'])
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.2, linestyle='--', which='both', color=PALETTE['secondary'])
    ax2.legend(loc='upper left', fontsize=11, framealpha=0.95,
              edgecolor=PALETTE['accent1'], fancybox=True)
    ax2.set_title('Cost Scaling Analysis', 
                  fontsize=14, fontweight='bold', pad=15, color=PALETTE['primary'])
    
    for spine in ['top', 'right']:
        ax2.spines[spine].set_visible(False)
    ax2.spines['left'].set_color(PALETTE['secondary'])
    ax2.spines['bottom'].set_color(PALETTE['secondary'])
    ax2.tick_params(colors=PALETTE['primary'])
    
    safe_layout()
    plt.savefig('figures/cost_comparison.png', bbox_inches='tight', dpi=300, facecolor='white')
    print("✓ cost_comparison.png")
    plt.close()

# Figure 4: Time
def generate_time_figure():
    stages = ['Video Loading', 'Scene Detection', 'Candidate Extract', 
              'pHash Dedup', 'SSIM Dedup', 'CLIP Dedup', 
              'Audio Extract', 'Frame Selection', 'VLM Extraction']
    times = [0.15, 5.89, 18.47, 2.31, 4.62, 5.18, 1.83, 1.96, 7.89]
    percentages = [0.4, 14.3, 44.7, 5.6, 11.2, 12.5, 4.4, 4.7, 19.1]
    
    colors = ['#8b9bab', '#a8b5c7', '#c4d1e0', '#d4a574', '#90a955',
              '#8b9bab', '#a8b5c7', '#c4d1e0', '#d4a574']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    bars = ax1.barh(stages, times, color=colors, 
                    edgecolor=PALETTE['primary'], linewidth=2, alpha=0.85)
    bars[2].set_edgecolor(PALETTE['accent2'])
    bars[2].set_linewidth(3)
    bars[8].set_edgecolor(PALETTE['warning'])
    bars[8].set_linewidth(3)
    
    for i, (bar, time, pct) in enumerate(zip(bars, times, percentages)):
        ax1.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2.,
                f'{time:.2f}s ({pct:.1f}%)', va='center', fontsize=10, 
                fontweight='bold', color=PALETTE['primary'])
    
    ax1.set_xlabel('Processing Time (seconds)', fontsize=13, fontweight='bold', color=PALETTE['primary'])
    ax1.set_ylabel('Pipeline Stage', fontsize=13, fontweight='bold', color=PALETTE['primary'])
    ax1.set_xlim(0, 26)
    ax1.grid(axis='x', alpha=0.2, linestyle='--', color=PALETTE['secondary'])
    ax1.set_title('Processing Time Breakdown (n=23 videos)', 
                  fontsize=14, fontweight='bold', pad=15, color=PALETTE['primary'])
    
    total_time = sum(times)
    ax1.text(0.98, 0.02, f'Total: {total_time:.1f}s', transform=ax1.transAxes,
            fontsize=12, fontweight='bold', ha='right', va='bottom',
            color=PALETTE['primary'],
            bbox=dict(boxstyle='round,pad=0.5', facecolor=PALETTE['accent3'],
                     edgecolor=PALETTE['primary'], linewidth=2, alpha=0.9))
    
    for spine in ['top', 'right']:
        ax1.spines[spine].set_visible(False)
    ax1.spines['left'].set_color(PALETTE['secondary'])
    ax1.spines['bottom'].set_color(PALETTE['secondary'])
    ax1.tick_params(colors=PALETTE['primary'])
    
    grouped_stages = []
    grouped_percentages = []
    other_total = 0
    
    for stage, pct in zip(stages, percentages):
        if pct >= 10:
            grouped_stages.append(stage)
            grouped_percentages.append(pct)
        else:
            other_total += pct
    
    if other_total > 0:
        grouped_stages.append('Other Stages\n(< 10%)')
        grouped_percentages.append(other_total)
    
    pie_colors = ['#c1666b', '#6c8ea0', '#d4a574', '#8b9bab', '#90a955']
    
    wedges, texts, autotexts = ax2.pie(grouped_percentages, labels=grouped_stages,
                                        autopct='%1.1f%%', startangle=90,
                                        colors=pie_colors, 
                                        wedgeprops={'edgecolor': PALETTE['primary'], 'linewidth': 2},
                                        textprops={'fontsize': 10, 'fontweight': 'bold',
                                                  'color': PALETTE['primary']})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    ax2.set_title('Time Distribution by Stage', 
                  fontsize=14, fontweight='bold', pad=15, color=PALETTE['primary'])
    
    safe_layout()
    plt.savefig('figures/time_breakdown.png', bbox_inches='tight', dpi=300, facecolor='white')
    print("✓ time_breakdown.png")
    plt.close()

# Main
if __name__ == "__main__":
    print("Generating figures with muted matte colors...")
    print("Professional academic color palette")
    print("-" * 50)
    
    try:
        generate_cascade_figure()
    except Exception as e:
        print(f"✗ cascade error: {e}")
    
    try:
        generate_adaptive_figure()
    except Exception as e:
        print(f"✗ adaptive error: {e}")
    
    try:
        generate_cost_figure()
    except Exception as e:
        print(f"✗ cost error: {e}")
    
    try:
        generate_time_figure()
    except Exception as e:
        print(f"✗ time error: {e}")
    
    print("-" * 50)
    print("✓ All figures saved in figures/")