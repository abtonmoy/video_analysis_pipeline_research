import matplotlib.pyplot as plt
import os

def main():
    os.makedirs('new_experiments/figures', exist_ok=True)
    
    stages = ['Native\n(30 FPS)', 'Candidates\n(change det.)', 'After\npHash', 'After\nLPIPS', 'After\nCLIP', 'Final\nSelected']
    counts = [981, 259, 237, 230, 82, 24]
    colors = ['#94a3b8', '#60a5fa', '#3b82f6', '#2563eb', '#1d4ed8', '#1e40af']

    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.barh(stages[::-1], counts[::-1], color=colors[::-1])

    # Add count labels on bars
    for bar, count in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                str(count), va='center', fontsize=10, fontweight='bold')

    ax.set_xlabel('Frame Count')
    ax.set_title('Progressive Frame Reduction (Sword of Chaos Gaming Ad)')
    plt.tight_layout()
    plt.savefig('new_experiments/figures/cascade_waterfall.pdf', bbox_inches='tight')

if __name__ == "__main__":
    main()
