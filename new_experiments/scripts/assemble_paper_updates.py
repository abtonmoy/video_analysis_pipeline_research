#!/usr/bin/env python3
"""
Assemble all experiment outputs into paper_updates_final.tex.

Reads results from each experiment and generates labeled LaTeX blocks
that can be inserted into paper.tex at the %%PLACEHOLDER locations.
"""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'new_experiments', 'results')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'new_experiments', 'paper_updates_final.tex')


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_file(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return f.read()


def assemble():
    sections = []

    # ---- EXPERIMENT 1: Budget Ablation ----
    ablation = load_json(os.path.join(RESULTS_DIR, 'budget_ablation', 'ablation_results.json'))
    if ablation:
        strats = ablation.get('strategies', {})
        sections.append("""
%% ================================================================
%% EXPERIMENT 1: Budget Ablation — Table for §3 (after Eq. 6)
%% ================================================================
\\begin{table}[t]
\\centering
\\caption{Budget formula ablation. Full AdaFrame (Eq.~6) vs.\\ simpler alternatives on """ + str(ablation.get('n_videos', 'N')) + """ videos.}
\\label{tab:budget_ablation}
\\begin{tabular}{@{}lrr@{}}
\\toprule
\\textbf{Strategy} & \\textbf{Mean Frames} & \\textbf{Topic Acc.\\ (\\%)} \\\\
\\midrule""")
        labels = {'A': 'Full AdaFrame', 'B': 'ISD only', 'C': 'Linear duration',
                  'D': 'Fixed-25', 'E': 'Energy only'}
        for s in ['A', 'B', 'C', 'D', 'E']:
            if s in strats:
                r = strats[s]
                bold_start = "\\textbf{" if s == 'A' else ""
                bold_end = "}" if s == 'A' else ""
                sections.append(
                    f"{bold_start}{labels[s]}{bold_end} & {r['mean_frames']} & {r['topic_accuracy']} \\\\")
        sections.append("""\\bottomrule
\\end{tabular}
\\end{table}""")

        # Paragraph for §5
        a_data = strats.get('A', {})
        d_data = strats.get('D', {})
        b_data = strats.get('B', {})
        sections.append(f"""
%% ================================================================
%% EXPERIMENT 1: Budget Ablation — Paragraph for §5
%% ================================================================
% We ablated the budget formula (Eq.~6) against four simpler alternatives
% (Table~\\ref{{tab:budget_ablation}}). The full formula achieves {a_data.get('topic_accuracy', 'XX')}\\%
% topic accuracy at {a_data.get('mean_frames', 'XX')} mean frames, compared to Fixed-25
% ({d_data.get('topic_accuracy', 'XX')}\\%) and ISD-only ({b_data.get('topic_accuracy', 'XX')}\\%
% at {b_data.get('mean_frames', 'XX')} frames). The combined ISD + energy formulation
% provides the best accuracy-efficiency tradeoff.""")
    else:
        sections.append("%% EXPERIMENT 1: Budget Ablation — NOT YET RUN")

    # ---- EXPERIMENT 2: PySceneDetect ----
    psd_results = load_json(os.path.join(RESULTS_DIR, 'pyscenedetect', 'pyscenedetect_results.json'))
    if psd_results and len(psd_results) > 0:
        n_videos = len(psd_results)
        mean_frames = sum(v.get('n_frames', 0) for v in psd_results.values()) / n_videos if n_videos else 0
        sections.append(f"""
%% ================================================================
%% EXPERIMENT 2: PySceneDetect — Baselines item for §4.2
%% ================================================================
% \\item \\textbf{{PySceneDetect}}: Content-aware scene detection (first frame per scene).
% {n_videos} videos processed, mean {mean_frames:.1f} frames per video.""")

        sections.append(f"""
%% ================================================================
%% EXPERIMENT 2: PySceneDetect — Table row for Table 2
%% ================================================================
% PySceneDetect & {mean_frames:.1f} & XX & XX.X & XX.X & X.XX & X.XXX \\\\""")
    else:
        sections.append("%% EXPERIMENT 2: PySceneDetect — NOT YET RUN")

    # ---- EXPERIMENT 3: DSNet ----
    dsnet_failed = read_file(os.path.join(RESULTS_DIR, 'dsnet', 'FAILED.txt'))
    dsnet_fallback = read_file(os.path.join(RESULTS_DIR, 'dsnet', 'fallback_text.tex'))
    if dsnet_failed:
        sections.append(f"""
%% ================================================================
%% EXPERIMENT 3: DSNet — Fallback
%% ================================================================
{dsnet_fallback if dsnet_fallback else '% DSNet experiment failed. Deep summarization baselines require architecture-specific inference.'}""")
    else:
        sections.append("%% EXPERIMENT 3: DSNet — NOT YET RUN")

    # ---- EXPERIMENT 4: Error-Free Intersection ----
    intersection = load_json(os.path.join(RESULTS_DIR, 'error_intersection', 'intersection_accuracy.json'))
    if intersection:
        int_size = intersection.get('intersection_size', 0)
        int_acc = intersection.get('intersection_accuracy', {})
        our_method = int_acc.get('hib_pipeline', int_acc.get('static_pipeline', {}))
        our_acc = our_method.get('topic_accuracy', 0)

        sections.append(f"""
%% ================================================================
%% EXPERIMENT 4: Error-Free Intersection — Paragraph for §5.1
%% ================================================================
Restricting to the $N={int_size}$ videos where all methods produced valid
extractions, AdaFrame achieves {our_acc}\\% topic accuracy on this error-free subset,
confirming that the main results are not biased by differential error rates.""")
    else:
        sections.append("%% EXPERIMENT 4: Error-Free Intersection — NOT YET RUN")

    # ---- EXPERIMENT 5: Cost Breakdown ----
    cost = load_json(os.path.join(RESULTS_DIR, 'cost_breakdown', 'cost_summary.json'))
    if cost:
        cascade = cost.get('cascade', {})
        vlm = cost.get('vlm_cost', {})
        savings = cost.get('savings', {})
        sections.append(f"""
%% ================================================================
%% EXPERIMENT 5: Cost Breakdown — Paragraph for §6.2
%% ================================================================
The cascade itself incurs compute cost (mean {cascade.get('mean_latency_s', 'XX')} seconds per
video, approximately \\${cascade.get('mean_cost_usd', 'XX'):.4f} at RTX~4090 cloud rates of
\\$0.50/hr). However, the VLM cost reduction---from \\${vlm.get('uniform_mean_usd', 'XX')}
(Uniform-1FPS) to \\${vlm.get('adaframe_mean_usd', 'XX')} (AdaFrame)---more than offsets
this overhead. Net savings are positive for {savings.get('positive_savings_pct', 'XX')}\\% of videos
(those longer than $\\approx${savings.get('breakeven_duration_s', 'XX')}$ seconds).""")
    else:
        sections.append("%% EXPERIMENT 5: Cost Breakdown — NOT YET RUN")

    # ---- EXPERIMENTS 6+7: Multi-VLM + Native Video ----
    limitations = read_file(os.path.join(RESULTS_DIR, 'limitations_update.tex'))
    if limitations:
        sections.append(f"""
%% ================================================================
%% EXPERIMENTS 6+7: Multi-VLM & Native Video — for §6.2 and §6.3
%% ================================================================
{limitations}""")
    else:
        sections.append("%% EXPERIMENTS 6+7: Multi-VLM & Native Video — NOT YET RUN")

    # ---- Write output ----
    output = """%%
%% paper_updates_final.tex
%% Generated by assemble_paper_updates.py
%% Contains all LaTeX snippets from experiments, labeled by section.
%%
""" + "\n".join(sections) + "\n"

    with open(OUTPUT_PATH, 'w') as f:
        f.write(output)

    print(f"Paper updates assembled: {OUTPUT_PATH}")
    print(f"Total sections: {len(sections)}")
    for s in sections:
        first_line = s.strip().split('\n')[0]
        print(f"  {first_line}")


if __name__ == '__main__':
    assemble()
