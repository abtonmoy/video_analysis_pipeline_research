#!/usr/bin/env python3
"""
Shared VLM retry queue utilities for all API-calling experiments.
Implements the error handling pattern from agents.md.
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime, timezone


import logging

# Configure logging to show our diagnostic errors in the terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Keep it clean for terminal output
)
logger = logging.getLogger(__name__)


def call_vlm_with_retry_queue(
    video_id: str,
    extract_fn,  # callable: () -> dict (result)
    provider: str,
    model: str,
    retry_queue_dir: str,
    failed_log_path: str,
    max_immediate_retries: int = 2,
    retry_delay: float = 5.0,
    # Optional metadata to save for retry
    extra_metadata: dict = None,
):
    """
    Call VLM API with immediate retries. On persistent failure,
    save to retry queue instead of crashing.

    Args:
        video_id: Video identifier
        extract_fn: Callable that performs extraction, returns dict on success
        provider: VLM provider name
        model: Model name
        retry_queue_dir: Directory for retry queue files
        failed_log_path: Path for JSONL failure log
        max_immediate_retries: Number of immediate retry attempts
        retry_delay: Base delay for exponential backoff
        extra_metadata: Additional data to save for retry (frames, prompt, etc.)

    Returns:
        dict result on success, None on failure (saved to queue)
    """
    for attempt in range(1, max_immediate_retries + 1):
        try:
            result = extract_fn()
            return result  # Success
        except Exception as e:
            error_code = getattr(e, 'status_code', None) or type(e).__name__
            error_msg = str(e)

            if attempt < max_immediate_retries:
                wait = retry_delay * (2 ** (attempt - 1))
                print(f"  [RETRY] {video_id}: {error_code} — waiting {wait}s "
                      f"(attempt {attempt}/{max_immediate_retries})")
                time.sleep(wait)
                continue

            # All immediate retries exhausted — save to queue
            print(f"  [QUEUED] {video_id}: {error_code} — saved to retry queue")

            os.makedirs(retry_queue_dir, exist_ok=True)
            os.makedirs(os.path.dirname(failed_log_path), exist_ok=True)

            queue_entry = {
                "video_id": video_id,
                "provider": provider,
                "model": model,
                "error_code": str(error_code),
                "error_message": error_msg[:500],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "attempt": attempt,
            }
            if extra_metadata:
                queue_entry.update(extra_metadata)

            queue_path = os.path.join(retry_queue_dir, f"{video_id}.json")
            with open(queue_path, 'w') as f:
                json.dump(queue_entry, f, indent=2)

            # Append to failed log
            with open(failed_log_path, 'a') as f:
                f.write(json.dumps({
                    "video_id": video_id,
                    "error_code": str(error_code),
                    "error_message": error_msg[:200],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "attempt": attempt,
                }) + "\n")

            return None  # Signal failure to caller

    return None


def process_retry_queue(retry_queue_dir, extract_fn_factory, results_dict):
    """
    Process saved retry queue entries.

    Args:
        retry_queue_dir: Dir containing VIDEO_ID.json files
        extract_fn_factory: callable(queue_entry) -> callable that does extraction
        results_dict: Dict to add successful results to
    """
    queue_dir = Path(retry_queue_dir)
    done_dir = queue_dir / "done"
    done_dir.mkdir(exist_ok=True)

    queue_files = list(queue_dir.glob("*.json"))
    print(f"Retry queue: {len(queue_files)} entries")

    succeeded = 0
    failed = 0

    for qf in queue_files:
        with open(qf) as f:
            entry = json.load(f)

        video_id = entry["video_id"]
        print(f"  Retrying {video_id}...")

        try:
            extract_fn = extract_fn_factory(entry)
            result = extract_fn()
            results_dict[video_id] = result
            # Move to done
            qf.rename(done_dir / qf.name)
            succeeded += 1
            print(f"    ✓ Success")
        except Exception as e:
            entry["attempt"] = entry.get("attempt", 1) + 1
            entry["error_message"] = str(e)[:500]
            entry["timestamp"] = datetime.now(timezone.utc).isoformat()
            with open(qf, 'w') as f:
                json.dump(entry, f, indent=2)
            failed += 1
            print(f"    ✗ Failed again: {e}")

    print(f"Retry complete: {succeeded} succeeded, {failed} still failing")


def print_run_summary(total, succeeded, failed, experiment_name, retry_queue_dir):
    """Print standard run completion summary."""
    print(f"\n=== {experiment_name} Run Complete ===")
    print(f"Succeeded: {succeeded} / {total}")
    print(f"Failed:    {failed} / {total} (saved to retry_queue/)")
    if failed > 0:
        script_name = experiment_name.lower().replace(' ', '_')
        print(f"Retry with: uv run scripts/{script_name}.py --retry {retry_queue_dir}")
