# AGENT TASK: Fill All Paper Placeholders (Final Revision)

---

## ⚠️ CRITICAL CONSTRAINTS — READ BEFORE DOING ANYTHING

1. **DO NOT modify anything in `src/`**. Copy files you need to `new_experiments/src_copy/` and modify the copies.
2. **Use `uv` for everything** (dependency management, running scripts).
3. **Test every experiment on 10 videos first** from `/home/wabashcs/abt/use_data` before running on the full set.
4. **Store ALL new results in `new_experiments/`** — never overwrite `main_results/`, `test_results/`, or any existing data.
5. **Same extraction format**: Output JSON must match the schema in `main_results/processing_results.json` exactly. Reuse the prompt and schema from `src/extraction/prompts.py` and `src/extraction/schema.py`.
6. **Method name is AdaFrame** in all generated text.

---

## Directory Structure

```
new_experiments/
├── src_copy/                    # Copied+modified source files (NEVER edit src/ directly)
│   ├── selection/
│   │   └── representative.py    # Modified budget formula for ablation
│   └── extraction/              # Copied extraction code (prompts, schema, client)
├── scripts/                     # Experiment scripts
│   ├── exp1_budget_ablation.py
│   ├── exp2_pyscenedetect.py
│   ├── exp3_dsnet.py
│   ├── exp4_error_intersection.py
│   ├── exp5_cost_breakdown.py
│   ├── exp6_claude.py
│   ├── exp7_native_video.py
│   └── shared_50_video_ids.json # Shared video list for exp6+exp7
├── results/                     # All output JSONs and CSVs
│   ├── budget_ablation/
│   ├── pyscenedetect/
│   ├── dsnet/
│   ├── error_intersection/
│   ├── cost_breakdown/
│   ├── claude_50/
│   └── native_video_50/
├── figures/                     # Generated plots
└── paper_updates_final.tex      # All LaTeX snippets, labeled by experiment
```

---

## Data Locations

| Data | Path | Notes |
|------|------|-------|
| Main pipeline results | `main_results/processing_results.json` | 1,704 videos, full AdaFrame pipeline. **READ ONLY.** |
| Benchmark results | `test_results/benchmark_results.json` | 484 videos × 9 methods. **READ ONLY.** |
| Ground truth | `data/` directory | Pitt Ads annotations. **READ ONLY.** |
| Pipeline source | `src/` | **DO NOT MODIFY.** Copy to `new_experiments/src_copy/`. |
| Test videos (10) | `/home/wabashcs/abt/use_data` | Use these for testing BEFORE full runs. |
| New results | `new_experiments/results/` | All new outputs go here. |

---

## Setup

```bash
# Create experiment directory structure
mkdir -p new_experiments/{src_copy/selection,scripts,results/{budget_ablation,pyscenedetect,dsnet,error_intersection,cost_breakdown,claude_50,native_video_50},figures}

# Copy source files needed for modification (DO NOT EDIT ORIGINALS)
cp src/selection/representative.py new_experiments/src_copy/selection/representative.py
cp -r src/extraction new_experiments/src_copy/extraction

# Initialize uv project
cd new_experiments
uv init --name adaframe-experiments
uv add pandas numpy scipy matplotlib seaborn

# Experiment-specific deps
uv add scenedetect[opencv]   # Experiment 2
uv add anthropic              # Experiment 6
uv add google-generativeai    # Experiment 7
```

---

## Extraction Format

Every VLM extraction experiment MUST produce JSON matching the format in `main_results/processing_results.json`. To ensure this:

1. Import the prompt builder and schema from the COPIED source:
```python
import sys
sys.path.insert(0, 'new_experiments/src_copy')
from extraction.prompts import build_single_pass_prompt, prepare_frames_for_prompt
from extraction.schema import get_schema
from extraction.llm_client import get_llm_client, AdExtractor
```

2. Or instantiate `AdExtractor` the same way the main pipeline does:
```python
extractor = AdExtractor(
    provider="gemini",
    model="gemini-2.0-flash",  # This is Gemini-3-Flash
    single_pass=True,
    schema_mode="fixed",
    temporal_context=True,
    include_timestamps=True,
    include_time_deltas=True,
    include_position_labels=True,
    include_narrative_instructions=True,
)

result = extractor.extract(
    frames=selected_frames,        # List[Tuple[float, np.ndarray]]
    video_duration=duration,
    audio_context=audio_ctx,       # Optional[Dict], can be None
)
```

3. Verify output keys match a sample from `main_results/processing_results.json` before running at scale.

---

## EXPERIMENT 1: Budget Formula Ablation

### Placeholders filled
- `%%PLACEHOLDER` after Eq. 6 in §3 (table)
- `%%PLACEHOLDER` after ablation subsection in §5 (paragraph)

### Setup
```bash
cp src/selection/representative.py new_experiments/src_copy/selection/representative.py
# Edit ONLY the copy at new_experiments/src_copy/selection/representative.py
```

### What to implement

In `new_experiments/src_copy/selection/representative.py`, create a wrapper around `_compute_frame_budget()` that accepts a `strategy` parameter. Test 5 strategies — only the budget computation changes, everything else (cascade, TA-NMS, VLM) stays identical.

**Strategy A: Full AdaFrame (current code — control)**
```python
budget = min(base + floor(1.5 * isd), max(base, floor(base + duration * 0.25 * e_k * e_p)))
```

**Strategy B: ISD only**
```python
budget = max(5, isd)
```

**Strategy C: Linear duration**
```python
budget = max(5, floor(duration * 0.25))
```

**Strategy D: Fixed-25**
```python
budget = 25
```

**Strategy E: Energy only (no ISD cap)**
```python
budget = max(5, floor(base + duration * 0.25 * e_k * e_p))
```

### Approach (minimize VLM calls)

The cascade already produces a ranked candidate list. The budget only determines how many survive the final cut. For each video:
1. Load the full pipeline result from `main_results/processing_results.json` — get the post-cascade candidates with importance scores, e_k, e_p, isd values
2. For each strategy, compute what the budget WOULD be
3. Take the top-N candidates by importance score
4. If the selected frame SET differs from Strategy A, run VLM extraction on the new set
5. If the selected set is identical to A, reuse A's extraction result

### Also: ISD multiplier sensitivity
Vary the 1.5× multiplier: {0.5, 1.0, 1.5, 2.0, 3.0}. Report mean frames and topic accuracy.

### Test first
```bash
uv run scripts/exp1_budget_ablation.py --test --videos /home/wabashcs/abt/use_data --output results/budget_ablation/test/
```

### Full run
```bash
uv run scripts/exp1_budget_ablation.py --input main_results/processing_results.json --output results/budget_ablation/
```

### Output
- `new_experiments/results/budget_ablation/ablation_results.json`
- `new_experiments/results/budget_ablation/multiplier_sensitivity.csv`
- `new_experiments/figures/budget_ablation.pdf`

---

## EXPERIMENT 2: PySceneDetect Baseline

### Placeholders filled
- `%%PLACEHOLDER` in §4.2 baselines list
- `%%PLACEHOLDER` row in Table 2

### Setup
```bash
uv add "scenedetect[opencv]"
```

### What to implement

For each of the 484 benchmark videos:
```python
from scenedetect import detect, ContentDetector

scene_list = detect(video_path, ContentDetector())
# Select first frame of each detected scene + the very first frame
```

Then run VLM extraction on selected frames using AdExtractor (from copied source) with the SAME prompt/schema. Save in same JSON format as `main_results/processing_results.json`.

### Test first
```bash
uv run scripts/exp2_pyscenedetect.py --test --videos /home/wabashcs/abt/use_data --output results/pyscenedetect/test/
```

### Full run
```bash
uv run scripts/exp2_pyscenedetect.py --benchmark test_results/benchmark_results.json --output results/pyscenedetect/
```

### Output
- `new_experiments/results/pyscenedetect/pyscenedetect_results.json`

---

## EXPERIMENT 3: DSNet Baseline (Optional)

### Placeholders filled
- `%%PLACEHOLDER` in §4.2 baselines list
- `%%PLACEHOLDER` row in Table 2

### Setup
```bash
git clone https://github.com/li-plus/DSNet.git new_experiments/dsnet_repo
cd new_experiments/dsnet_repo && uv add -r requirements.txt
```

### What to implement
For each benchmark video: extract features → DSNet inference → select top-22 frames → VLM extraction (same prompt/schema).

### If it fails
Save `new_experiments/results/dsnet/FAILED.txt` with the error, and produce this disclaimer:
```latex
Deep summarization baselines (DSNet, TaskSumm) require architecture-specific
inference and were not included in our benchmark.
```

### Test first
```bash
uv run scripts/exp3_dsnet.py --test --videos /home/wabashcs/abt/use_data --output results/dsnet/test/
```

---

## EXPERIMENT 4: Error-Free Intersection Accuracy

### Placeholder filled
- `%%PLACEHOLDER` after main results in §5.1

### NO VLM calls needed — pure analysis of existing data

```python
import json

with open('test_results/benchmark_results.json') as f:
    benchmark = json.load(f)

methods = ['uniform_1fps', 'random', 'histogram', 'orb', 'optical_flow',
           'clip_only', 'kmeans', 'hib_pipeline']

# Find videos where ALL methods produced non-error bare_extraction
intersection = []
for vid, data in benchmark.items():
    if all('error' not in data.get(m, {}).get('bare_extraction', {'error': True})
           for m in methods):
        intersection.append(vid)

# Recompute topic accuracy per method on intersection only
# Use bare_extraction.topic.topic_id vs ground truth
```

### Test first
```bash
uv run scripts/exp4_error_intersection.py --test --input test_results/benchmark_results.json --n 50
```

### Full run
```bash
uv run scripts/exp4_error_intersection.py --input test_results/benchmark_results.json --gt data/ --output results/error_intersection/
```

### Output
- `new_experiments/results/error_intersection/intersection_accuracy.csv`

---

## EXPERIMENT 5: Cascade Cost Breakdown

### Placeholder filled
- `%%PLACEHOLDER` in §6.2

### NO VLM calls — arithmetic on existing data

From `main_results/processing_results.json`, for each video:
```python
GPU_HOURLY_RATE = 0.50  # RTX 4090 cloud rate, conservative

# Inspect the actual JSON keys first!
cascade_latency_s = result['processing']['latency_s']  # adapt to actual key name
cascade_cost = cascade_latency_s / 3600 * GPU_HOURLY_RATE
vlm_cost = ...  # from actual data or compute from frame count × per-frame price
total_adaframe = cascade_cost + vlm_cost

# Uniform baseline: just VLM cost, no cascade overhead
uniform_vlm_cost = ...  # compute from uniform frame count
net_savings = uniform_vlm_cost - total_adaframe
```

**IMPORTANT**: Inspect `main_results/processing_results.json` first to find the correct field names for latency, frame counts, and costs.

### Test first
```bash
uv run scripts/exp5_cost_breakdown.py --test --input main_results/processing_results.json --n 10
```

### Full run
```bash
uv run scripts/exp5_cost_breakdown.py --input main_results/processing_results.json --output results/cost_breakdown/
```

### Output
- `new_experiments/results/cost_breakdown/cost_analysis.csv`
- `new_experiments/figures/cost_breakdown.pdf`

---

## EXPERIMENT 6: Second VLM — Claude (50 videos)

### Placeholder filled
- `%%PLACEHOLDER` in §6.2

### ⚠️ Use the SAME 50 videos as Experiment 7

Generate the shared list ONCE:
```python
import random, json
random.seed(42)
with open('test_results/benchmark_results.json') as f:
    all_videos = list(json.load(f).keys())
selected_50 = random.sample(all_videos, 50)
with open('new_experiments/scripts/shared_50_video_ids.json', 'w') as f:
    json.dump(selected_50, f)
```

### What to implement

For each of the 50 videos:
1. Load AdaFrame's selected frames from `main_results/processing_results.json`
2. Re-encode as base64
3. Send to Claude using AnthropicClient from COPIED `src_copy/extraction/llm_client.py`
4. Use the SAME prompt built with `build_single_pass_prompt` from copied source
5. Save in same JSON format

```python
sys.path.insert(0, 'new_experiments/src_copy')
from extraction.llm_client import get_llm_client

client = get_llm_client(
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    max_retries=3,
)
```

### Test first
```bash
uv run scripts/exp6_claude.py --test --videos /home/wabashcs/abt/use_data --n 3 --output results/claude_50/test/
```

### Full run
```bash
uv run scripts/exp6_claude.py --video-ids scripts/shared_50_video_ids.json --input main_results/processing_results.json --output results/claude_50/
```

### Output
- `new_experiments/results/claude_50/claude_results.json`

---

## EXPERIMENT 7: Gemini Native Video Mode (50 videos)

### Placeholder filled
- `%%PLACEHOLDER` in §6.2

### ⚠️ Use the SAME 50 videos as Experiment 6

Load from `new_experiments/scripts/shared_50_video_ids.json`.

### What to implement

```python
import google.generativeai as genai
import time

model = genai.GenerativeModel('gemini-3-flash')

video_file = genai.upload_file(path=video_path)
while video_file.state.name == "PROCESSING":
    time.sleep(2)
    video_file = genai.get_file(video_file.name)

# Use same prompt/schema from copied source (but no frames — video is the input)
response = model.generate_content(
    [video_file, prompt],
    generation_config={"max_output_tokens": 2000, "temperature": 0.0}
)
```

Save in same JSON format. Record cost and latency.

### Test first
```bash
uv run scripts/exp7_native_video.py --test --videos /home/wabashcs/abt/use_data --n 3 --output results/native_video_50/test/
```

### Full run
```bash
uv run scripts/exp7_native_video.py --video-ids scripts/shared_50_video_ids.json --output results/native_video_50/
```

### Output
- `new_experiments/results/native_video_50/native_video_results.json`

---

## EXPERIMENT 8: Limitations Update (depends on 6+7)

### Placeholder filled
- `%%PLACEHOLDER` in §6.3

### What to produce

Read results from Experiments 6 and 7. Write the appropriate paragraph:

**Both succeeded:**
```latex
We validated AdaFrame with Claude Haiku 4.5 on 50 videos, achieving XX.X\%
topic accuracy (95\% CI: [XX.X\%, XX.X\%]) compared to XX.X\% for Gemini-3-Flash,
supporting provider-agnostic applicability.
```

**Neither succeeded:**
```latex
All experiments use Gemini-3-Flash. While AdaFrame's cascade is provider-agnostic
by design, multi-VLM evaluation is a priority for future work.
```

---

## Execution Order

```
PHASE 1 — No API calls (existing data only):
  1. EXPERIMENT 4: Error-free intersection    [30 min]
  2. EXPERIMENT 5: Cost breakdown             [30 min]

PHASE 2 — Some VLM calls:
  3. EXPERIMENT 1: Budget ablation            [2-4 hrs]

PHASE 3 — API-heavy:
  4. Generate shared_50_video_ids.json
  5. EXPERIMENT 2: PySceneDetect              [half day]
  6. EXPERIMENT 6: Claude (50 videos)         [2 hrs]
  7. EXPERIMENT 7: Native video (50 videos)   [2 hrs]
  8. EXPERIMENT 3: DSNet (try, may fail)      [2 hrs]
  9. EXPERIMENT 8: Write limitations          [10 min]
```

## Testing Workflow (EVERY experiment)

```bash
# Step 1: Test on 10 videos from /home/wabashcs/abt/use_data
uv run scripts/expN.py --test --videos /home/wabashcs/abt/use_data --output results/EXPNAME/test/

# Step 2: Inspect test output
cat results/EXPNAME/test/*.json | python -m json.tool | head -50

# Step 3: Verify JSON format matches main_results
python -c "
import json
with open('main_results/processing_results.json') as f:
    ref = json.load(f)
ref_keys = set(list(ref.values())[0].keys())
with open('new_experiments/results/EXPNAME/test/results.json') as f:
    new = json.load(f)
new_keys = set(list(new.values())[0].keys())
print('Missing keys:', ref_keys - new_keys)
print('Extra keys:', new_keys - ref_keys)
"

# Step 4: Only after test passes, run full
uv run scripts/expN.py --full --output results/EXPNAME/
```

## Final Output

Single file: `new_experiments/paper_updates_final.tex` with labeled blocks:
```latex
%% EXPERIMENT 1: Budget Ablation — Table for §3
...
%% EXPERIMENT 1: Budget Ablation — Paragraph for §5
...
%% EXPERIMENT 2: PySceneDetect — Baselines item for §4.2
...
%% EXPERIMENT 2: PySceneDetect — Table row for Table 2
...
```

All figures saved to `new_experiments/figures/`.

---

## ⚠️ API Error Handling & Retry Queue

Every experiment that calls a VLM API (Experiments 1, 2, 3, 6, 7) MUST implement the following error handling. **Do not silently skip failed videos.**

### On any API error (429, 500, 503, timeout, rate limit, content policy, etc.):

1. **Save everything needed to retry later** to `new_experiments/results/EXPNAME/retry_queue/VIDEO_ID.json`:
```json
{
  "video_id": "abc123.mp4",
  "experiment": "exp2_pyscenedetect",
  "provider": "gemini",
  "model": "gemini-3-flash",
  "error_code": 429,
  "error_message": "Resource exhausted: quota exceeded",
  "timestamp": "2026-03-24T14:32:01Z",
  "selected_frames_b64": ["base64_frame_1...", "base64_frame_2..."],
  "selected_timestamps": [0.5, 2.1, 4.8, ...],
  "frame_count": 22,
  "video_duration": 30.5,
  "prompt": "You are analyzing a 30.5-second video advertisement...",
  "audio_context": {"transcription": [...], "mood": "energetic", ...}
}
```

2. **Log the failure** to `new_experiments/results/EXPNAME/failed_videos.jsonl` (append mode):
```json
{"video_id": "abc123.mp4", "error_code": 429, "error_message": "quota exceeded", "timestamp": "2026-03-24T14:32:01Z", "attempt": 1}
```

3. **Continue processing** the remaining videos — do NOT stop the batch.

4. **At the end of a run**, print a summary:
```
=== Run Complete ===
Succeeded: 412 / 484
Failed:     72 / 484 (saved to retry_queue/)
Retry with: uv run scripts/expN.py --retry results/EXPNAME/retry_queue/
```

### Retry mechanism

Every experiment script MUST accept a `--retry` flag:
```bash
uv run scripts/exp2_pyscenedetect.py --retry results/pyscenedetect/retry_queue/
```

This loads each `VIDEO_ID.json` from the retry queue, re-sends the saved frames+prompt to the API, and on success moves the file from `retry_queue/` to `retry_queue/done/`. On repeated failure, the file stays in `retry_queue/` with the attempt counter incremented.

### Implementation pattern

```python
import json, os, time
from pathlib import Path
from datetime import datetime

def call_vlm_with_retry_queue(
    video_id: str,
    frames_b64: list,
    timestamps: list,
    prompt: str,
    video_duration: float,
    audio_context: dict,
    provider: str,
    model: str,
    extractor,  # AdExtractor or BaseLLMClient
    retry_queue_dir: str,
    failed_log_path: str,
    max_immediate_retries: int = 2,
    retry_delay: float = 5.0,
):
    """
    Call VLM API with immediate retries. On persistent failure,
    save to retry queue instead of crashing.
    """
    for attempt in range(1, max_immediate_retries + 1):
        try:
            result = extractor.extract(frames, video_duration, audio_context)
            return result  # Success
        except Exception as e:
            error_code = getattr(e, 'status_code', None) or type(e).__name__
            error_msg = str(e)
            
            if attempt < max_immediate_retries:
                wait = retry_delay * (2 ** (attempt - 1))
                print(f"  [RETRY] {video_id}: {error_code} — waiting {wait}s (attempt {attempt}/{max_immediate_retries})")
                time.sleep(wait)
                continue
            
            # All immediate retries exhausted — save to queue
            print(f"  [QUEUED] {video_id}: {error_code} — saved to retry queue")
            
            os.makedirs(retry_queue_dir, exist_ok=True)
            
            queue_entry = {
                "video_id": video_id,
                "experiment": os.path.basename(retry_queue_dir.rstrip('/').rsplit('/', 1)[0]),
                "provider": provider,
                "model": model,
                "error_code": str(error_code),
                "error_message": error_msg[:500],
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "selected_frames_b64": frames_b64,
                "selected_timestamps": timestamps,
                "frame_count": len(frames_b64),
                "video_duration": video_duration,
                "prompt": prompt,
                "audio_context": audio_context,
            }
            
            queue_path = os.path.join(retry_queue_dir, f"{video_id}.json")
            with open(queue_path, 'w') as f:
                json.dump(queue_entry, f)
            
            # Append to failed log
            with open(failed_log_path, 'a') as f:
                f.write(json.dumps({
                    "video_id": video_id,
                    "error_code": str(error_code),
                    "error_message": error_msg[:200],
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "attempt": attempt,
                }) + "\n")
            
            return None  # Signal failure to caller
```

### Directory layout per experiment with retry support
```
new_experiments/results/pyscenedetect/
├── pyscenedetect_results.json    # Successful extractions
├── failed_videos.jsonl            # Log of all failures
└── retry_queue/                   # Saved payloads for retry
    ├── abc123.mp4.json
    ├── def456.mp4.json
    └── done/                      # Successfully retried
        └── ghi789.mp4.json
```

---

## Rules

- **Every number from real computation. No fabricated data.**
- Report N for every metric.
- 95% CI for small-sample experiments (N=50).
- If an experiment fails or undermines a claim, report honestly.
- If an experiment fails (deps, API errors), produce fallback text.
- **Never silently skip failed API calls.** Save to retry queue.