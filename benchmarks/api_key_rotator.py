"""
API Key Rotator — round-robin key rotation with retry + backoff.

Sits between the ExtractionWrapper and the LLM provider to distribute
requests across multiple API keys, avoiding 429 (rate limit) and 499 errors.

Usage:
    1. Set numbered environment variables in .env:
         GOOGLE_API_KEY=your_primary_key
         GOOGLE_API_KEY1=your_second_key
         GOOGLE_API_KEY2=your_third_key

       Works with any provider (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)

    2. Import and use in extraction_wrapper.py or runner.py:
         from benchmarks.api_key_rotator import KeyRotator
         rotator = KeyRotator("gemini")
         key = rotator.next_key()
"""

import itertools
import logging
import os
import random
import time
import threading
from typing import Any, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ── Environment variable names for each provider ──────────────────────────
# Maps provider name → (multi-key env var, single-key fallback env vars)
PROVIDER_KEY_ENV = {
    "gemini": ("GEMINI_API_KEYS", ["GEMINI_API_KEY", "GOOGLE_API_KEY"]),
    "openai": ("OPENAI_API_KEYS", ["OPENAI_API_KEY"]),
    "anthropic": ("ANTHROPIC_API_KEYS", ["ANTHROPIC_API_KEY"]),
}


class KeyRotator:
    """
    Thread-safe round-robin API key rotator.

    Loads keys from environment variables. Cycles through them on each call
    to next_key(). Tracks per-key rate-limit cooldowns.
    """

    def __init__(self, provider: str):
        self.provider = provider.lower()
        self.keys = self._load_keys()
        self._cycle = itertools.cycle(range(len(self.keys)))
        self._lock = threading.Lock()

        # Per-key cooldown tracking: {index: earliest_available_time}
        self._cooldowns: Dict[int, float] = {}

        logger.info(
            f"KeyRotator[{self.provider}]: loaded {len(self.keys)} API key(s)"
        )

    def _load_keys(self) -> List[str]:
        """
        Load API keys from numbered environment variables.

        Looks for the base key and then numbered variants:
            GOOGLE_API_KEY   (or GEMINI_API_KEY, OPENAI_API_KEY, etc.)
            GOOGLE_API_KEY1
            GOOGLE_API_KEY2
            ...
        Stops scanning at the first missing number.
        """
        _, base_vars = PROVIDER_KEY_ENV.get(
            self.provider, (None, [f"{self.provider.upper()}_API_KEY"])
        )

        keys: List[str] = []

        for base_var in base_vars:
            # Load the base key (e.g. GOOGLE_API_KEY)
            base_key = os.environ.get(base_var, "").strip()
            if base_key:
                keys.append(base_key)

            # Load numbered keys (e.g. GOOGLE_API_KEY1, GOOGLE_API_KEY2, ...)
            for i in range(1, 20):
                numbered = os.environ.get(f"{base_var}{i}", "").strip()
                if numbered:
                    keys.append(numbered)
                else:
                    break

            if keys:
                break  # Found keys for this base var, stop checking fallbacks

        # Deduplicate while preserving order
        seen = set()
        unique_keys = []
        for k in keys:
            if k not in seen:
                seen.add(k)
                unique_keys.append(k)

        if not unique_keys:
            raise ValueError(
                f"No API keys found for provider '{self.provider}'. "
                f"Set {base_vars[0]} and optionally {base_vars[0]}1, "
                f"{base_vars[0]}2, ... in your .env file."
            )

        return unique_keys

    @property
    def key_count(self) -> int:
        return len(self.keys)

    def next_key(self) -> str:
        """
        Get the next available API key (round-robin).
        Skips keys that are in cooldown if alternatives are available.
        """
        with self._lock:
            now = time.time()
            # Try up to N keys to find one not in cooldown
            for _ in range(len(self.keys)):
                idx = next(self._cycle)
                cooldown_until = self._cooldowns.get(idx, 0)
                if now >= cooldown_until:
                    return self.keys[idx]

            # All keys are in cooldown — return the one with shortest wait
            earliest_idx = min(self._cooldowns, key=self._cooldowns.get)
            wait = self._cooldowns[earliest_idx] - now
            if wait > 0:
                logger.warning(
                    f"All {len(self.keys)} keys rate-limited. "
                    f"Waiting {wait:.1f}s for cooldown..."
                )
                time.sleep(wait)
            return self.keys[earliest_idx]

    def mark_rate_limited(self, key: str, cooldown_seconds: float = 60.0):
        """Mark a key as rate-limited for a cooldown period."""
        with self._lock:
            try:
                idx = self.keys.index(key)
                self._cooldowns[idx] = time.time() + cooldown_seconds
                logger.warning(
                    f"Key {idx + 1}/{len(self.keys)} rate-limited, "
                    f"cooldown {cooldown_seconds:.0f}s"
                )
            except ValueError:
                pass

    def clear_cooldown(self, key: str):
        """Clear cooldown for a key after a successful request."""
        with self._lock:
            try:
                idx = self.keys.index(key)
                self._cooldowns.pop(idx, None)
            except ValueError:
                pass


def retry_with_rotation(
    rotator: KeyRotator,
    call_fn: Callable[[str], T],
    max_retries: int = 6,
    base_delay: float = 2.0,
    max_delay: float = 120.0,
    rate_limit_codes: tuple = (429, 499, 503),
) -> T:
    """
    Execute an API call with key rotation and exponential backoff.

    Args:
        rotator: KeyRotator instance
        call_fn: Function that takes an API key string and makes the request.
                 Should raise an exception with a .status_code attribute on failure.
        max_retries: Maximum retry attempts across all keys
        base_delay: Initial backoff delay in seconds
        max_delay: Maximum backoff delay
        rate_limit_codes: HTTP status codes that trigger key rotation

    Returns:
        The result of call_fn on success

    Raises:
        The last exception if all retries are exhausted
    """
    last_exception = None

    for attempt in range(max_retries):
        key = rotator.next_key()
        try:
            result = call_fn(key)
            rotator.clear_cooldown(key)
            return result

        except Exception as e:
            last_exception = e
            status = getattr(e, "status_code", None) or _extract_status(e)

            if status in rate_limit_codes:
                # Parse Retry-After header if available
                retry_after = _extract_retry_after(e)
                cooldown = retry_after or min(base_delay * (2 ** attempt), max_delay)

                rotator.mark_rate_limited(key, cooldown_seconds=cooldown)
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries}: "
                    f"status {status}, rotating key (backoff {cooldown:.1f}s)"
                )

                # If we have multiple keys, try the next one immediately
                if rotator.key_count > 1:
                    continue

                # Single key — must wait
                jitter = random.uniform(0, cooldown * 0.1)
                time.sleep(cooldown + jitter)
            else:
                # Non-rate-limit error — backoff and retry with same logic
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.error(
                    f"Attempt {attempt + 1}/{max_retries}: "
                    f"error {e}, retrying in {delay:.1f}s"
                )
                time.sleep(delay)

    raise last_exception


def _extract_status(e: Exception) -> Optional[int]:
    """Try to extract HTTP status code from various exception types."""
    # google.api_core.exceptions
    if hasattr(e, "code"):
        code = e.code
        return code if isinstance(code, int) else None

    # httpx / requests
    if hasattr(e, "response") and hasattr(e.response, "status_code"):
        return e.response.status_code

    # String parsing as last resort
    msg = str(e)
    for code in (429, 499, 503, 500, 502):
        if str(code) in msg:
            return code

    return None


def _extract_retry_after(e: Exception) -> Optional[float]:
    """Try to extract Retry-After value from exception/response headers."""
    try:
        if hasattr(e, "response") and hasattr(e.response, "headers"):
            val = e.response.headers.get("Retry-After") or e.response.headers.get("retry-after")
            if val:
                return float(val)
    except (ValueError, AttributeError):
        pass
    return None