#!/usr/bin/env python3
"""
Shared VLM retry queue utilities for all API-calling experiments.
Implements the error handling pattern from new_experiments.
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime, timezone
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def call_vlm_with_retry_queue(
    video_id: str,
    extract_fn,
    provider: str,
    model: str,
    retry_queue_dir: str,
    failed_log_path: str,
    max_immediate_retries: int = 2,
    retry_delay: float = 5.0,
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
        extra_metadata: Additional data to save for retry

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
                logger.warning(f"[RETRY] {video_id}: {error_code} — waiting {wait}s "
                              f"(attempt {attempt}/{max_immediate_retries})")
                time.sleep(wait)
                continue

            # All immediate retries exhausted — save to queue
            logger.error(f"[QUEUED] {video_id}: {error_code} — saved to retry queue")

            os.makedirs(retry_queue_dir, exist_ok=True)
            os.makedirs(os.path.dirname(failed_log_path), exist_ok=True)

            queue_entry = {
                "video_id": video_id,
                "provider": provider,
                "model": model,
                "error_code": str(error_code),
                "error_message": error_msg[:500],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": extra_metadata or {},
            }

            # Write to retry queue
            queue_file = Path(retry_queue_dir) / f"{video_id}.json"
            with open(queue_file, 'w') as f:
                json.dump(queue_entry, f, indent=2)

            # Append to failed log
            with open(failed_log_path, 'a') as f:
                f.write(json.dumps(queue_entry) + '\n')

            return None  # Signal failure but don't crash


def process_retry_queue(
    retry_queue_dir: str,
    process_fn,
    max_attempts: int = 3
):
    """
    Process all items in the retry queue.

    Args:
        retry_queue_dir: Directory containing retry queue files
        process_fn: Function to process each queued item
        max_attempts: Maximum number of processing attempts per item

    Returns:
        tuple: (success_count, failed_count, remaining_count)
    """
    retry_dir = Path(retry_queue_dir)
    if not retry_dir.exists():
        logger.info("No retry queue directory found")
        return 0, 0, 0

    queue_files = list(retry_dir.glob("*.json"))
    if not queue_files:
        logger.info("Retry queue is empty")
        return 0, 0, 0

    logger.info(f"Processing {len(queue_files)} items in retry queue")

    success_count = 0
    failed_count = 0

    for queue_file in queue_files:
        try:
            with open(queue_file) as f:
                entry = json.load(f)

            video_id = entry["video_id"]
            attempt_count = entry.get("attempt_count", 0) + 1

            if attempt_count > max_attempts:
                logger.warning(f"[MAX RETRIES] {video_id} — giving up")
                failed_count += 1
                continue

            # Update attempt count
            entry["attempt_count"] = attempt_count

            # Process
            result = process_fn(entry)

            if result is not None:
                # Success — remove from queue
                queue_file.unlink()
                success_count += 1
                logger.info(f"[SUCCESS] {video_id} — removed from queue")
            else:
                # Still failed — update queue file
                entry["last_attempt"] = datetime.now(timezone.utc).isoformat()
                with open(queue_file, 'w') as f:
                    json.dump(entry, f, indent=2)
                failed_count += 1

        except Exception as e:
            logger.error(f"Error processing retry queue file {queue_file}: {e}")
            failed_count += 1

    remaining = len(list(retry_dir.glob("*.json")))
    logger.info(f"Retry queue processing complete: {success_count} succeeded, "
                f"{failed_count} failed, {remaining} remaining")

    return success_count, failed_count, remaining
