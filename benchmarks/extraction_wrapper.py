"""
Dual-mode extraction wrapper for Option C benchmarking.

Wraps the existing AdExtractor to provide two extraction modes:
  - BARE:  No temporal context, no audio, fixed schema (1 LLM call)
           → fair frame-selection comparison
  - FULL:  Complete Stage 7 treatment (2 LLM calls for adaptive schema)
           → system-level comparison

Both modes use the exact same LLM provider/model so the only variable
is the quality of the selected frames + context richness.

Includes API key rotation to avoid 429/499 rate-limit errors when
running benchmarks at scale. Set GEMINI_API_KEYS=key1,key2,key3 in .env.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.extraction.llm_client import AdExtractor, create_extractor
from benchmarks.api_key_rotator import KeyRotator, retry_with_rotation

logger = logging.getLogger(__name__)


class ExtractionWrapper:
    """
    Provides bare and full extraction using the pipeline's own AdExtractor.
    Rotates API keys across calls to stay under rate limits.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Full pipeline config dict (same format as default.yaml).
                    The extraction section is used for provider/model settings.
        """
        ext = config.get("extraction", {})
        # Allow benchmark config to override provider/model
        bench_ext = config.get("benchmark", {}).get("extraction", {})
        self.provider = bench_ext.get("provider", ext.get("provider", "gemini"))
        self.model = bench_ext.get("model", ext.get("model", "gemini-2.0-flash-exp"))
        self.max_tokens = bench_ext.get("max_tokens", ext.get("max_tokens", 4000))

        # Retry settings from config
        retry_cfg = bench_ext.get("retry", {})
        self.max_retries = retry_cfg.get("max_retries", 6)
        self.base_delay = retry_cfg.get("base_delay", 2.0)
        self.max_delay = retry_cfg.get("max_delay", 120.0)

        # Initialize key rotator
        try:
            self.rotator = KeyRotator(self.provider)
        except ValueError:
            self.rotator = None
            logger.warning(
                "No multiple API keys found — running without rotation. "
                "Set GEMINI_API_KEYS=key1,key2,... in .env for rotation."
            )

        # ------------------------------------------------------------------
        # BARE extractor: no temporal context, no audio, fixed schema
        # ------------------------------------------------------------------
        self.bare = AdExtractor(
            provider=self.provider,
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.0,
            schema_mode="fixed",
            temporal_context=False,
            include_timestamps=False,
            include_time_deltas=False,
            include_position_labels=False,
            include_narrative_instructions=False,
        )

        # ------------------------------------------------------------------
        # FULL extractor: complete Stage 7 (temporal + audio + adaptive)
        # ------------------------------------------------------------------
        self.full = create_extractor(config)

        logger.info(
            f"ExtractionWrapper initialized: provider={self.provider}, "
            f"model={self.model}, keys={self.rotator.key_count if self.rotator else 1}"
        )

    def _set_api_key(self, key: str):
        """
        Set the active API key in the environment.

        Most LLM clients (google-genai, openai, etc.) read the key from
        env vars on each request, so swapping the env var is sufficient.
        """
        if self.provider == "gemini":
            os.environ["GEMINI_API_KEY"] = key
            os.environ["GOOGLE_API_KEY"] = key
        elif self.provider == "openai":
            os.environ["OPENAI_API_KEY"] = key
        elif self.provider == "anthropic":
            os.environ["ANTHROPIC_API_KEY"] = key
        else:
            os.environ[f"{self.provider.upper()}_API_KEY"] = key

    def extract_bare(
        self,
        frames: List[Tuple[float, np.ndarray]],
        video_duration: float,
    ) -> Dict[str, Any]:
        """
        Fair comparison extraction — same minimal prompt for all methods.
        Uses key rotation + retry on rate-limit errors.
        """
        if not frames:
            return {"error": "No frames provided"}

        if self.rotator and self.rotator.key_count > 1:
            def _call(key: str) -> Dict[str, Any]:
                self._set_api_key(key)
                return self.bare.extract(frames, video_duration, audio_context=None)

            try:
                return retry_with_rotation(
                    self.rotator, _call,
                    max_retries=self.max_retries,
                    base_delay=self.base_delay,
                    max_delay=self.max_delay,
                )
            except Exception as e:
                logger.error(f"Bare extraction failed after retries: {e}")
                return {"error": str(e)}
        else:
            # Single key — simple retry with backoff
            return self._retry_simple(
                lambda: self.bare.extract(frames, video_duration, audio_context=None),
                label="bare",
            )

    def extract_full(
        self,
        frames: List[Tuple[float, np.ndarray]],
        video_duration: float,
        audio_context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        System comparison extraction — full Stage 7 treatment.
        Uses key rotation + retry on rate-limit errors.
        """
        if not frames:
            return {"error": "No frames provided"}

        if self.rotator and self.rotator.key_count > 1:
            def _call(key: str) -> Dict[str, Any]:
                self._set_api_key(key)
                return self.full.extract(frames, video_duration, audio_context=audio_context)

            try:
                return retry_with_rotation(
                    self.rotator, _call,
                    max_retries=self.max_retries,
                    base_delay=self.base_delay,
                    max_delay=self.max_delay,
                )
            except Exception as e:
                logger.error(f"Full extraction failed after retries: {e}")
                return {"error": str(e)}
        else:
            return self._retry_simple(
                lambda: self.full.extract(frames, video_duration, audio_context=audio_context),
                label="full",
            )

    def _retry_simple(
        self,
        fn,
        label: str = "",
        max_retries: int = 3,
        base_delay: float = 5.0,
    ) -> Dict[str, Any]:
        """Simple exponential backoff retry for single-key setups."""
        for attempt in range(max_retries):
            try:
                return fn()
            except Exception as e:
                status_msg = str(e)
                if any(code in status_msg for code in ("429", "499", "503")):
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"{label} extraction rate-limited (attempt {attempt + 1}), "
                        f"retrying in {delay:.0f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"{label} extraction failed: {e}")
                    return {"error": str(e)}

        logger.error(f"{label} extraction failed after {max_retries} retries")
        return {"error": f"Rate limited after {max_retries} retries"}