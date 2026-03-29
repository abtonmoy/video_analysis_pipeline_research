import os
import json
import csv
import matplotlib.pyplot as plt
import seaborn as sns

def generate_exp1_plot():
    results_path = "new_experiments/results/budget_ablation/ablation_results.json"
    if not os.path.exists(results_path):
        return
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    strategies = ["A", "B", "C", "D", "E"]
    labels = ["Full AdaFrame", "ISD Only", "Linear Duration", "Fixed-25", "Energy Only"]
    mean_frames = [data["strategies"][s]["mean_frames"] for s in strategies]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=labels, y=mean_frames, palette="viridis")
    plt.title("Mean Extracted Frames by Budget Strategy")
    plt.ylabel("Mean Frames Selected")
    plt.xlabel("Strategy")
    plt.savefig("new_experiments/results/budget_ablation/exp1_strategies.png")
    plt.close()

def generate_exp5_plot():
    results_path = "new_experiments/results/cost_breakdown/cost_analysis.csv"
    if not os.path.exists(results_path):
        return
    
    durations = []
    net_savings = []
    
    with open(results_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            durations.append(float(row['duration_s']))
            net_savings.append(float(row['net_saving_usd']))
            
    plt.figure(figsize=(10, 6))
    plt.scatter(durations, net_savings, alpha=0.5, color='blue')
    plt.axhline(0, color='red', linestyle='--', label='Breakeven')
    plt.title("Net Savings (USD) vs Video Duration")
    plt.xlabel("Video Duration (Seconds)")
    plt.ylabel("Net Savings (USD)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("new_experiments/results/cost_breakdown/exp5_savings.png")
    plt.close()

if __name__ == "__main__":
    os.makedirs("new_experiments/results/budget_ablation", exist_ok=True)
    os.makedirs("new_experiments/results/cost_breakdown", exist_ok=True)
    generate_exp1_plot()
    generate_exp5_plot()
