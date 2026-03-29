import json
import os

def check_schemas():
    main_path = 'main_results/processing_results.json'
    test_path = 'new_experiments/results/test/pyscenedetect/pyscenedetect_results.json'
    
    with open(main_path, 'r') as f:
        main_data = json.load(f)
        
    if not os.path.exists(test_path):
        print(f"Test path {test_path} does not exist yet.")
        return
        
    with open(test_path, 'r') as f:
        test_data = json.load(f)
        
    print("=== Metadata Keys ===")
    print("Main:", list(main_data.get('metadata', {}).keys()))
    print("Test:", list(test_data.get('metadata', {}).keys()))
    
    print("\n=== Result Object Keys ===")
    main_res = main_data.get('results', [])[0] if main_data.get('results') else {}
    test_res = test_data.get('results', [])[0] if test_data.get('results') else {}
    
    print("Main:", list(main_res.keys()))
    print("Test:", list(test_res.keys()))
    
    print("\n=== Video ID Format ===")
    print("Main example:", main_res.get('video_name', 'N/A'))
    print("Test example:", test_res.get('video_name', 'N/A'))
    
if __name__ == '__main__':
    check_schemas()
