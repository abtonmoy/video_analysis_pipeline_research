"""
Dual-mode extraction wrapper for Option C benchmarking.

Wraps the existing AdExtractor to provide two extraction modes:
  - BARE:  No temporal context, no audio, fixed schema (1 LLM call)
           → fair frame-selection comparison
  - FULL:  Complete Stage 7 treatment (2 LLM calls for adaptive schema)
           → system-level comparison

Both modes use the exact same LLM provider/model so the only variable
is the quality of the selected frames + context richness.

API key rotation is handled at the worker level in run_benchmarks.py.
Each worker sets its dedicated GOOGLE_API_KEY before this class is created.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.extraction.llm_client import AdExtractor, create_extractor

logger = logging.getLogger(__name__)


class ExtractionWrapper:
    """
    Provides bare and full extraction using the pipeline's own AdExtractor.
    Includes retry with exponential backoff for rate-limit errors.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Full pipeline config dict (same format as default.yaml).
                    The extraction section is used for provider/model settings.
        """
        ext = config.get("extraction", {})
        bench_ext = config.get("benchmark", {}).get("extraction", {})
        provider = bench_ext.get("provider", ext.get("provider", "gemini"))
        model = bench_ext.get("model", ext.get("model", "gemini-2.0-flash-exp"))
        max_tokens = bench_ext.get("max_tokens", ext.get("max_tokens", 4000))

        # Retry settings
        retry_cfg = bench_ext.get("retry", {})
        self.max_retries = retry_cfg.get("max_retries", 5)
        self.base_delay = retry_cfg.get("base_delay", 10.0)
        self.max_delay = retry_cfg.get("max_delay", 120.0)

        # ------------------------------------------------------------------
        # BARE extractor: no temporal context, no audio, fixed schema
        # ------------------------------------------------------------------
        self.bare = AdExtractor(
            provider=provider,
            model=model,
            max_tokens=max_tokens,
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
            f"ExtractionWrapper initialized: provider={provider}, model={model}"
        )

    def extract_bare(
        self,
        frames: List[Tuple[float, np.ndarray]],
        video_duration: float,
    ) -> Dict[str, Any]:
        """
        Fair comparison extraction — same minimal prompt for all methods.
        Uses fixed schema (1 LLM call) with retry on rate limits.
        """
        if not frames:
            return {"error": "No frames provided"}

        return self._retry(
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
        Uses adaptive schema (2 LLM calls) with retry on rate limits.
        """
        if not frames:
            return {"error": "No frames provided"}

        return self._retry(
            lambda: self.full.extract(
                frames, video_duration, audio_context=audio_context
            ),
            label="full",
        )

    def _retry(
        self,
        fn,
        label: str = "",
    ) -> Dict[str, Any]:
        """
        Retry with exponential backoff on rate-limit errors.

        The base_delay starts at 10s (not 1-2s like the inner llm_client retry)
        because by the time we get here, the inner retry has already failed
        3 times with short delays — we need longer waits to let the quota reset.
        """
        for attempt in range(self.max_retries):
            try:
                return fn()
            except Exception as e:
                status_msg = str(e)
                is_rate_limit = any(
                    code in status_msg for code in ("429", "499", "503", "ResourceExhausted")
                )

                if is_rate_limit and attempt < self.max_retries - 1:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    logger.warning(
                        f"{label} extraction rate-limited "
                        f"(attempt {attempt + 1}/{self.max_retries}), "
                        f"waiting {delay:.0f}s before retry..."
                    )
                    time.sleep(delay)
                elif is_rate_limit:
                    logger.error(
                        f"{label} extraction failed after {self.max_retries} retries"
                    )
                    return {"error": f"Rate limited after {self.max_retries} retries"}
                else:
                    logger.error(f"{label} extraction failed: {e}")
                    return {"error": str(e)}

        return {"error": "Unexpected retry exhaustion"}