import json
import os

def main():
    bench_file = "test_results/benchmark_results.json"
    data_dir = "/home/wabashcs/abt/use_data"
    
    with open(bench_file, 'r') as f:
        data = json.load(f)
        
    results = data.get('results', [])
    
    fast_cut_video = None
    slow_paced_video = None
    
    for i, r in enumerate(results[:1]):
        print(f"Keys: {r.keys()}")
        print(f"Uniform exists: {'uniform_1fps' in r}")
        print(f"HIB exists: {'hib_pipeline' in r}")
        print(f"Scenes natively stored: {'scenes' in r}")
        vid_name = os.path.basename(r.get('video_name', ''))
        vid_path = os.path.join(data_dir, vid_name)
        if not os.path.exists(vid_path):
            continue
            
        uniform_frames = r.get('uniform_1fps', {}).get('selected_frames', [])
        hib_frames = r.get('hib_pipeline', {}).get('selected_frames', [])
        
        # In processing_results.json, it's 'pipeline_stats' and 'extraction', but in test_results it has metric comparisons.
        num_scenes = len(r.get('scenes', []))
        
        if not uniform_frames or not hib_frames:
            continue
            
        if num_scenes >= 15 and fast_cut_video is None:
            fast_cut_video = r
            
        if 2 <= num_scenes <= 6 and slow_paced_video is None:
            slow_paced_video = r
            
        if fast_cut_video and slow_paced_video:
            break

    print(f"Fast Cut Found: {fast_cut_video is not None}")
    if fast_cut_video:
        print(f"Fast Cut Name: {fast_cut_video['video_name']}")
    print(f"Slow Paced Found: {slow_paced_video is not None}")
    if slow_paced_video:
        print(f"Slow Cut Name: {slow_paced_video['video_name']}")

if __name__ == "__main__":
    main()
