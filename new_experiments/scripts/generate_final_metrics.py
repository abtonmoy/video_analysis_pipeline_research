#!/usr/bin/env python3
import json
import os
import sys

TOPIC_SUPER_CATEGORIES = {
    "food_beverage":    [1, 2, 3, 4, 5, 6, 7, 8],
    "auto":             [9],
    "tech_service":     [10, 11, 14, 15],
    "finance_edu":      [12, 13],
    "other_service":    [16],
    "personal_care":    [17, 18, 19],
    "home_family":      [20, 22, 23, 24],
    "entertainment":    [21, 26, 27, 29],
    "travel_shopping":  [25, 28],
    "social_cause":     [30, 31, 32, 33, 34, 35, 36, 37, 38],
}
_T_TO_S = {}
for cat, ids in TOPIC_SUPER_CATEGORIES.items():
    for tid in ids: _T_TO_S[tid] = cat

def load_json(path):
    if not os.path.exists(path): return None
    with open(path, 'r') as f: return json.load(f)

def normalize_vid_id(vid_name):
    return os.path.splitext(os.path.basename(vid_name))[0]

def mean(xs):
    return sum(xs) / len(xs) if xs else 0

def main():
    gt_dir = 'data/annotations_videos/video/cleaned_result/'
    gt_topics = load_json(gt_dir + 'video_Topics_clean.json')
    gt_eff = load_json(gt_dir + 'video_Effective_clean.json')
    
    benchmark = load_json('test_results/benchmark_results.json')
    psd = load_json('new_experiments/results/pyscenedetect/pyscenedetect_results.json')
    dsnet = load_json('new_experiments/results/dsnet/dsnet_results.json')
    native = load_json('new_experiments/results/native_video_50/native_video_results.json')

    methods = {}

    def add_result(m, vid_id, pred_topic, pred_eff, frames, density=0.0):
        if m not in methods: methods[m] = {'frames': [], 'topic_correct': 0, 'super_correct': 0, 'eff_scores': [], 'density': [], 'total': 0}
        methods[m]['total'] += 1
        methods[m]['frames'].append(frames)
        methods[m]['density'].append(density)
        
        # Topic
        if vid_id in gt_topics:
            gt_t = int(gt_topics[vid_id])
            try:
                p_t = int(pred_topic)
                if p_t == gt_t: methods[m]['topic_correct'] += 1
                if _T_TO_S.get(p_t) == _T_TO_S.get(gt_t): methods[m]['super_correct'] += 1
            except: pass
            
        # Effectiveness
        if vid_id in gt_eff:
            try:
                methods[m]['eff_scores'].append(float(pred_eff))
            except: pass

    # 1. Benchmark
    if benchmark:
        for vid, data in benchmark.get('per_video', {}).items():
            if vid == 'metadata': continue
            vid_id = normalize_vid_id(vid)
            for m, mdata in data.get('baselines', {}).items():
                ext = mdata.get('full_extraction', {})
                add_result(m, vid_id, 
                           ext.get('topic', {}).get('topic_id'),
                           ext.get('engagement_metrics', {}).get('effectiveness_score'),
                           mdata.get('metrics', {}).get('num_frames', 0),
                           mdata.get('selection', {}).get('info_density', 0))

    # 2. PySceneDetect
    if psd:
        for r in psd.get('results', []):
            if r.get('status') == 'success':
                ext = r.get('extraction', {})
                add_result('PySceneDetect', normalize_vid_id(r['video_name']),
                           ext.get('topic', {}).get('topic_id'),
                           ext.get('engagement_metrics', {}).get('effectiveness_score'),
                           r.get('n_frames', 0))

    # 3. DSNet
    if dsnet:
        for r in dsnet.get('results', []):
            if r.get('status') == 'success':
                ext = r.get('extraction', {})
                add_result('DSNet', normalize_vid_id(r['video_name']),
                           ext.get('topic', {}).get('topic_id'),
                           ext.get('engagement_metrics', {}).get('effectiveness_score'),
                           r.get('n_frames', 0))

    # 4. Native
    if native:
        for r in native.get('results', []):
            if r.get('status') == 'success':
                ext = r.get('extraction', {})
                add_result('Gemini Native (Ours)', normalize_vid_id(r['video_name']),
                           ext.get('topic', {}).get('topic_id'),
                           ext.get('engagement_metrics', {}).get('effectiveness_score'),
                           0.0)

    print(f"{'Method':<25} | {'Frames':<7} | {'Topic%':<7} | {'Super%':<7} | {'Eff':<5} | {'Dens':<5}")
    print("-" * 75)
    
    for m in sorted(methods.keys()):
        s = methods[m]
        avg_f = mean(s['frames'])
        topic_acc = (s['topic_correct'] / s['total'] * 100) if s['total'] else 0
        super_acc = (s['super_correct'] / s['total'] * 100) if s['total'] else 0
        avg_eff = mean(s['eff_scores'])
        avg_dens = mean(s['density'])
        
        print(f"{m:<25} | {avg_f:>7.1f} | {topic_acc:>7.1f} | {super_acc:>7.1f} | {avg_eff:>5.2f} | {avg_dens:>5.3f}")

if __name__ == '__main__':
    main()
