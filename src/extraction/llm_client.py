# src\extraction\llm_client.py
"""
LLM clients and extraction orchestration.

Features:
- Multi-provider support (Anthropic, OpenAI, Gemini, Mock)
- Retry with exponential backoff on transient errors
- Robust JSON parsing (regex extraction, markdown strip, trailing comma fix)
- Single-pass extraction (ad type + content in one LLM call, ~50% cost reduction)
- Confidence scoring on extraction results
- Gemini native video input mode (direct video upload)
"""

import re
import json
import time
import random
import logging
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np

from .prompts import (
    FrameForPrompt,
    prepare_frames_for_prompt,
    build_temporal_prompt,
    build_type_detection_prompt,
    build_single_pass_prompt,
    build_segmented_prompt,
)
from .schema import get_schema, get_valid_ad_types

logger = logging.getLogger(__name__)


# ============================================================================
# Gemini safety settings — BLOCK_NONE to prevent false positives on ad content
# (alcohol, gambling, beauty products routinely trigger overzealous filters)
# ---------------------------------------------------------------------------
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    try:
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        GEMINI_SAFETY_PERMISSIVE = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    except ImportError:
        GEMINI_SAFETY_PERMISSIVE = None


# ============================================================================
# Retry Utility
# ============================================================================

def _get_retryable_exceptions() -> tuple:
    """
    Build a tuple of exception types that should trigger automatic retry.
    SDK imports are guarded so missing provider packages don't break anything.
    """
    retryable = [ConnectionError, TimeoutError, OSError]
    try:
        from google.api_core.exceptions import (
            ResourceExhausted,
            ServiceUnavailable,
            TooManyRequests,
            InternalServerError as GoogleInternalError,
        )
        retryable.extend([ResourceExhausted, ServiceUnavailable,
                          TooManyRequests, GoogleInternalError])
    except ImportError:
        pass
    try:
        from anthropic import RateLimitError as AnthropicRateLimit
        from anthropic import InternalServerError as AnthropicInternalError
        retryable.extend([AnthropicRateLimit, AnthropicInternalError])
    except ImportError:
        pass
    try:
        from openai import RateLimitError as OpenAIRateLimit
        retryable.extend([OpenAIRateLimit])
    except ImportError:
        pass
    return tuple(retryable)


def _retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: tuple = None,
):
    """
    Execute a function with exponential backoff retry.

    Args:
        func: Callable to execute
        max_retries: Number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        retryable_exceptions: Tuple of exception types to retry on.
            If None, auto-detects from installed provider SDKs.

    Returns:
        Function result

    Raises:
        Last exception if all retries exhausted
    """
    if retryable_exceptions is None:
        retryable_exceptions = _get_retryable_exceptions()

    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except retryable_exceptions as e:
            last_exception = e
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                jitter = random.uniform(0, delay * 0.25)
                actual_delay = delay + jitter
                status_code = getattr(e, "status_code", None) or getattr(e, "code", None)
                logger.warning(
                    f"Retry {attempt + 1}/{max_retries} after {type(e).__name__} "
                    f"(status={status_code}): {e}. "
                    f"Waiting {actual_delay:.1f}s..."
                )
                time.sleep(actual_delay)
            else:
                logger.error(f"All {max_retries} retries exhausted: {e}")
        except Exception as e:
            # Fallback: inspect error string for retryable patterns not caught by type
            error_str = str(e).lower()
            status_code = getattr(e, "status_code", None) or getattr(e, "code", None)

            is_retryable = (
                status_code in (429, 500, 502, 503, 504)
                or "rate limit" in error_str
                or "too many requests" in error_str
                or "resource_exhausted" in error_str
                or "unavailable" in error_str
                or "server error" in error_str
                or "overloaded" in error_str
                or "timeout" in error_str
            )

            if is_retryable and attempt < max_retries:
                last_exception = e
                delay = min(base_delay * (2 ** attempt), max_delay)
                jitter = random.uniform(0, delay * 0.25)
                actual_delay = delay + jitter
                logger.warning(
                    f"Retry {attempt + 1}/{max_retries} after {type(e).__name__} "
                    f"(status={status_code}): {e}. "
                    f"Waiting {actual_delay:.1f}s..."
                )
                time.sleep(actual_delay)
            else:
                raise

    raise last_exception


# ============================================================================
# JSON Parsing
# ============================================================================

def _repair_truncated_json(text: str) -> str:
    """
    Attempt to repair truncated JSON by closing open structures.

    When an LLM hits max_output_tokens, the response ends mid-JSON.
    This function tracks open braces/brackets with a character-level stack
    and appends the necessary closing characters.

    Args:
        text: Potentially truncated JSON string (should already be
              extracted from markdown / surrounding text by the caller)

    Returns:
        Repaired JSON string with all structures closed.
        The result may have missing fields — that's intentional.
        The existing compute_confidence() function will score it low.
    """
    # Track parser state through the string
    in_string = False
    escape_next = False
    stack = []

    for char in text:
        if escape_next:
            escape_next = False
            continue
        if char == '\\' and in_string:
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        # Outside of string literals: track structure nesting
        if char in ('{', '['):
            stack.append(char)
        elif char == '}' and stack and stack[-1] == '{':
            stack.pop()
        elif char == ']' and stack and stack[-1] == '[':
            stack.pop()

    # Nothing to repair — JSON is already balanced
    if not stack and not in_string:
        return text

    repaired = text.rstrip().rstrip(',')

    # Close an open string literal (truncated mid-value)
    if in_string:
        repaired += '"'

    # Remove trailing incomplete key-value pairs that would cause parse errors
    # Examples: '..."some_key": ' or '..."some_key": "partial'
    repaired = re.sub(r',\s*"[^"]*"\s*:\s*$', '', repaired)
    repaired = re.sub(r',\s*$', '', repaired)

    # Close all open brackets/braces in reverse (innermost first)
    for opener in reversed(stack):
        if opener == '{':
            repaired += '}'
        elif opener == '[':
            repaired += ']'

    return repaired


def _extract_gemini_response(response) -> str:
    """
    Safely extract text from a Gemini GenerateContentResponse.

    Handles three failure modes:
    1. Empty candidates list (safety block before generation started)
    2. finish_reason == SAFETY (blocked during or after generation)
    3. finish_reason == MAX_TOKENS (truncated — logged but text still returned
       so the downstream JSON repair can attempt recovery)

    Returns:
        Response text string. On safety blocks, returns a valid JSON error
        string like '{"error": "SAFETY_BLOCKED", ...}'. This flows through
        _parse_json_response() normally, and AdExtractor.extract() sees the
        "error" key in the parsed dict.
    """
    # Case 1: No candidates at all
    if not response.candidates:
        logger.warning("Gemini returned no candidates (likely safety-blocked)")
        return json.dumps({
            "error": "SAFETY_BLOCKED",
            "detail": "No candidates returned by Gemini",
        })

    candidate = response.candidates[0]

    # Normalize finish_reason to a string (SDK uses enum types)
    finish_reason = getattr(candidate, "finish_reason", None)
    if hasattr(finish_reason, "name"):
        finish_reason = finish_reason.name
    # Some SDK versions use integer enum values
    if isinstance(finish_reason, int):
        _FINISH_REASON_MAP = {
            0: "UNSPECIFIED", 1: "STOP", 2: "MAX_TOKENS",
            3: "SAFETY", 4: "RECITATION", 5: "OTHER",
        }
        finish_reason = _FINISH_REASON_MAP.get(finish_reason, str(finish_reason))

    # Case 2: Safety blocked
    if finish_reason == "SAFETY":
        safety_ratings = getattr(candidate, "safety_ratings", [])
        detail_parts = []
        for r in safety_ratings:
            cat = getattr(r.category, "name", str(r.category)) if hasattr(r, "category") else "?"
            prob = getattr(r.probability, "name", str(r.probability)) if hasattr(r, "probability") else "?"
            detail_parts.append(f"{cat}={prob}")
        detail = ", ".join(detail_parts) if detail_parts else "unknown"
        logger.warning(f"Gemini response safety-blocked: {detail}")
        return json.dumps({
            "error": "SAFETY_BLOCKED",
            "finish_reason": "SAFETY",
            "detail": detail,
        })

    # Case 3: Max tokens (truncated) — log but still return text for JSON repair
    if finish_reason == "MAX_TOKENS":
        logger.warning(
            "Gemini response truncated (finish_reason=MAX_TOKENS). "
            "JSON repair will be attempted by _parse_json_response()."
        )

    # Normal text extraction
    try:
        return response.text
    except (ValueError, AttributeError):
        # Some SDK versions raise ValueError when .text is accessed on empty parts
        try:
            if candidate.content and candidate.content.parts:
                return candidate.content.parts[0].text
        except (AttributeError, IndexError):
            pass
        logger.error(f"Could not extract text from Gemini response (finish_reason={finish_reason})")
        return json.dumps({
            "error": "EMPTY_RESPONSE",
            "finish_reason": str(finish_reason),
        })


def _parse_json_response(response: str) -> Dict[str, Any]:
    """
    Robustly parse JSON from LLM response text.

    Handles:
    - Raw JSON
    - Markdown code blocks (```json ... ```)
    - Nested code blocks
    - Trailing commas
    - Explanatory text before/after JSON

    Args:
        response: Raw LLM response text

    Returns:
        Parsed dictionary

    Raises:
        json.JSONDecodeError if no valid JSON found
    """
    text = response.strip()

    # Attempt 1: Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: Strip markdown code blocks
    code_block_pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    match = re.search(code_block_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Attempt 3: Find JSON object with regex (outermost braces)
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        json_str = brace_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Attempt 4: Fix trailing commas
            fixed = re.sub(r",\s*([}\]])", r"\1", json_str)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

            # Attempt 5: Stack-based truncation repair
            # Handles the case where max_output_tokens cut the response mid-JSON
            try:
                repaired = _repair_truncated_json(fixed)
                result = json.loads(repaired)
                logger.warning(
                    "JSON truncation repaired via autoclose. "
                    "Output may be incomplete — consider increasing max_output_tokens."
                )
                return result
            except (json.JSONDecodeError, Exception):
                pass

    # Attempt 3b: The response may be truncated so severely that there is
    # no closing brace at all (the greedy regex \{.*\} above requires both
    # braces). Try from the first opening brace to end-of-string.
    first_brace = text.find('{')
    if first_brace >= 0:
        partial = text[first_brace:]
        try:
            repaired = _repair_truncated_json(partial)
            fixed = re.sub(r",\s*([}\]])", r"\1", repaired)
            result = json.loads(fixed)
            logger.warning(
                "JSON truncation repaired from partial response (no closing brace found)."
            )
            return result
        except (json.JSONDecodeError, Exception):
            pass

    raise json.JSONDecodeError(
        f"Could not extract valid JSON from response (length={len(text)})",
        text[:200],
        0,
    )


# ============================================================================
# LLM Clients
# ============================================================================

class BaseLLMClient(ABC):
    """Abstract base class for LLM clients with retry support."""

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @abstractmethod
    def _call_api(
        self,
        frames: List[FrameForPrompt],
        prompt: str,
    ) -> str:
        """Raw API call without retries. Subclasses implement this."""
        pass

    def extract(
        self,
        frames: List[FrameForPrompt],
        prompt: str,
    ) -> str:
        """Send frames and prompt to LLM with automatic retry on transient errors."""
        return _retry_with_backoff(
            func=lambda: self._call_api(frames, prompt),
            max_retries=self.max_retries,
            base_delay=self.retry_delay,
        )


class AnthropicClient(BaseLLMClient):
    """Claude API client with retry."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4000,
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        super().__init__(max_retries=max_retries, retry_delay=retry_delay)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def _call_api(self, frames: List[FrameForPrompt], prompt: str) -> str:
        client = self._get_client()

        content = []
        for frame in frames:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": frame.base64_image,
                },
            })

        content.append({"type": "text", "text": prompt})

        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": content}],
        )

        return response.content[0].text


class OpenAIClient(BaseLLMClient):
    """OpenAI GPT-4V client with retry."""

    def __init__(
        self,
        model: str = "gpt-4o",
        max_tokens: int = 4000,
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        super().__init__(max_retries=max_retries, retry_delay=retry_delay)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def _call_api(self, frames: List[FrameForPrompt], prompt: str) -> str:
        client = self._get_client()

        content = []
        for frame in frames:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame.base64_image}"
                },
            })

        content.append({"type": "text", "text": prompt})

        response = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": content}],
        )

        return response.choices[0].message.content


class GeminiClient(BaseLLMClient):
    """Google Gemini API client with retry and optional JSON mode."""

    def __init__(
        self,
        model: str = "gemini-3.0-flash-exp",
        max_tokens: int = 4000,
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        json_mode: bool = True,
    ):
        super().__init__(max_retries=max_retries, retry_delay=retry_delay)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.json_mode = json_mode
        self._client = None

    def _get_client(self):
        if self._client is None:
            import google.generativeai as genai
            import os

            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            genai.configure(api_key=api_key, client_options={'api_endpoint': 'generativelanguage.googleapis.com'})
            self._client = genai.GenerativeModel(self.model)
        return self._client

    def _call_api(self, frames: List[FrameForPrompt], prompt: str) -> str:
        import base64
        from PIL import Image
        from io import BytesIO

        client = self._get_client()

        content = []
        for frame in frames:
            image_data = base64.b64decode(frame.base64_image)
            pil_image = Image.open(BytesIO(image_data))
            content.append(pil_image)

        content.append(prompt)

        generation_config = {
            "max_output_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        # Use JSON mode if available — guarantees valid JSON output
        if self.json_mode:
            generation_config["response_mime_type"] = "application/json"

        # Add safety settings to prevent false-positive blocks on ad content
        gen_kwargs = {"generation_config": generation_config}
        if GEMINI_SAFETY_PERMISSIVE is not None:
            gen_kwargs["safety_settings"] = GEMINI_SAFETY_PERMISSIVE

        response = client.generate_content(content, **gen_kwargs)

        return _extract_gemini_response(response)


class GeminiVideoClient(BaseLLMClient):
    """
    Gemini client that uploads video directly instead of sending frames.

    Uses Gemini's native video understanding — eliminates frame extraction overhead
    and lets the model choose its own optimal sampling rate.
    """

    def __init__(
        self,
        model: str = "gemini-3.0-flash-exp",
        max_tokens: int = 2000,
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        super().__init__(max_retries=max_retries, retry_delay=retry_delay)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            import google.generativeai as genai
            import os

            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            genai.configure(api_key=api_key, client_options={'api_endpoint': 'generativelanguage.googleapis.com'})
            self._client = genai.GenerativeModel(self.model)
        return self._client

    def extract_from_video(self, video_path: str, prompt: str) -> str:
        """
        Upload video directly to Gemini and extract information.

        Args:
            video_path: Path to video file
            prompt: Extraction prompt

        Returns:
            LLM response text
        """
        import google.generativeai as genai

        def _do_call():
            client = self._get_client()

            # Upload video file
            logger.info(f"Uploading video to Gemini: {video_path}")
            video_file = genai.upload_file(video_path)

            # Wait for processing
            while video_file.state.name == "PROCESSING":
                logger.debug("Waiting for video processing...")
                time.sleep(2)
                video_file = genai.get_file(video_file.name)

            if video_file.state.name == "FAILED":
                raise RuntimeError(f"Video processing failed: {video_file.state.name}")

            # Generate with video + prompt
            response = client.generate_content(
                [video_file, prompt],
                generation_config={
                    "max_output_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "response_mime_type": "application/json",
                },
            )

            # Clean up uploaded file
            try:
                genai.delete_file(video_file.name)
            except Exception:
                pass

            return response.text

        return _retry_with_backoff(
            func=_do_call,
            max_retries=self.max_retries,
            base_delay=self.retry_delay,
        )

    def _call_api(self, frames: List[FrameForPrompt], prompt: str) -> str:
        """Fallback: use frame-based extraction if called via standard interface."""
        # Delegate to regular Gemini client behavior
        import base64
        from PIL import Image
        from io import BytesIO

        client = self._get_client()
        content = []

        for frame in frames:
            image_data = base64.b64decode(frame.base64_image)
            pil_image = Image.open(BytesIO(image_data))
            content.append(pil_image)

        content.append(prompt)

        response = client.generate_content(
            content,
            generation_config={
                "max_output_tokens": self.max_tokens,
                "temperature": self.temperature,
                "response_mime_type": "application/json",
            },
        )

        return response.text


class MockLLMClient(BaseLLMClient):
    """Mock client for testing without API calls."""

    def __init__(self):
        super().__init__(max_retries=0, retry_delay=0)

    def _call_api(self, frames: List[FrameForPrompt], prompt: str) -> str:
        return json.dumps({
            "ad_type": "brand_awareness",
            "brand": {
                "brand_name_text": "Test Brand",
                "logo_visible": True,
                "brand_text_contrast": "high",
            },
            "product": {
                "product_name": "Test Product",
                "industry": "technology",
            },
            "promotion": {
                "promo_present": True,
                "promo_text": "50% off",
                "promo_deadline": "limited time",
                "price_value": "$9.99",
            },
            "call_to_action": {
                "cta_present": True,
                "cta_type": "Sign up button",
            },
            "visual_elements": {"text_density": "medium"},
            "content_rating": {"is_nsfw": False},
            "_mock": True,
            "_num_frames": len(frames),
        })


def get_llm_client(
    provider: str,
    model: str,
    max_tokens: int = 4000,
    temperature: float = 0.0,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> BaseLLMClient:
    """
    Factory function to get LLM client.

    Args:
        provider: "anthropic", "openai", "gemini", "gemini_video", or "mock"
        model: Model name
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        max_retries: Number of retry attempts on transient errors
        retry_delay: Base delay for exponential backoff (seconds)

    Returns:
        LLM client instance
    """
    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )

    if provider == "anthropic":
        return AnthropicClient(**kwargs)
    elif provider == "openai":
        return OpenAIClient(**kwargs)
    elif provider == "gemini":
        return GeminiClient(**kwargs)
    elif provider == "gemini_video":
        return GeminiVideoClient(**kwargs)
    elif provider == "mock":
        return MockLLMClient()
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Choose from: anthropic, openai, gemini, gemini_video, mock"
        )


# ============================================================================
# Confidence Scoring
# ============================================================================

def compute_confidence(
    result: Dict[str, Any],
    audio_context: Optional[Dict] = None,
    num_frames: int = 0,
) -> float:
    """
    Compute confidence score for extraction quality.

    Factors:
    - Schema field completeness (how many fields are non-null)
    - Audio context availability (transcription boosts confidence)
    - Frame count (more frames = more information)

    Args:
        result: Extraction result dictionary
        audio_context: Audio context used during extraction
        num_frames: Number of frames sent to LLM

    Returns:
        Confidence score between 0.0 and 1.0
    """
    if "error" in result:
        return 0.0

    score = 0.0

    # Factor 1: Schema field completeness (0-0.5)
    total_fields = 0
    non_null_fields = 0

    def _count_fields(obj, prefix=""):
        nonlocal total_fields, non_null_fields
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key.startswith("_"):
                    continue
                if isinstance(value, dict):
                    _count_fields(value, f"{prefix}{key}.")
                else:
                    total_fields += 1
                    if value is not None and value != "" and value != []:
                        non_null_fields += 1

    _count_fields(result)
    if total_fields > 0:
        score += 0.5 * (non_null_fields / total_fields)

    # Factor 2: Audio context (0-0.25)
    if audio_context:
        audio_score = 0.0
        if audio_context.get("has_speech"):
            audio_score += 0.1
        if audio_context.get("transcription"):
            audio_score += 0.1
        if audio_context.get("key_phrases"):
            audio_score += 0.05
        score += min(audio_score, 0.25)

    # Factor 3: Frame count (0-0.25)
    if num_frames >= 5:
        score += 0.25
    elif num_frames >= 3:
        score += 0.15
    elif num_frames >= 1:
        score += 0.05

    return min(score, 1.0)


# ============================================================================
# Extractor
# ============================================================================

class AdExtractor:
    """
    Main extractor class with:
    - Adaptive schema support
    - Audio context integration
    - Single-pass extraction (ad type + content in one call)
    - Confidence scoring
    - Robust JSON parsing
    - Segment-level prompting
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4000,
        temperature: float = 0.0,
        schema_mode: str = "adaptive",
        temporal_context: bool = True,
        include_timestamps: bool = True,
        include_time_deltas: bool = True,
        include_position_labels: bool = True,
        include_narrative_instructions: bool = True,
        single_pass: bool = True,
        segment_prompting: bool = False,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.client = get_llm_client(
            provider, model, max_tokens, temperature,
            max_retries=max_retries, retry_delay=retry_delay,
        )
        self.schema_mode = schema_mode
        self.temporal_context = temporal_context
        self.include_timestamps = include_timestamps
        self.include_time_deltas = include_time_deltas
        self.include_position_labels = include_position_labels
        self.include_narrative_instructions = include_narrative_instructions
        self.single_pass = single_pass
        self.segment_prompting = segment_prompting

    def detect_ad_type(self, frames: List[FrameForPrompt]) -> str:
        """
        Detect ad type from frames (used when single_pass is False).

        Args:
            frames: Prepared frames

        Returns:
            Ad type string
        """
        prompt = build_type_detection_prompt()

        try:
            response = self.client.extract(frames, prompt)
            ad_type = response.strip().lower().replace(" ", "_")

            valid_types = get_valid_ad_types()
            if ad_type in valid_types:
                return ad_type

            for valid in valid_types:
                if valid in ad_type or ad_type in valid:
                    return valid

            logger.warning(f"Unknown ad type: {ad_type}, defaulting to brand_awareness")
            return "brand_awareness"

        except Exception as e:
            logger.error(f"Ad type detection failed: {e}")
            return "brand_awareness"

    def extract(
        self,
        frames: List[Tuple[float, np.ndarray]],
        video_duration: float,
        audio_context: Optional[Dict] = None,
        scene_boundaries: Optional[List[Tuple[float, float]]] = None,
    ) -> Dict[str, Any]:
        """
        Extract structured information from ad frames.

        Single-pass mode: type detection + extraction in one LLM call (default).
        Segment mode: groups frames by scene for better context.

        Args:
            frames: List of (timestamp, frame) tuples
            video_duration: Total video duration
            audio_context: Optional audio transcription and features
            scene_boundaries: Optional scene boundaries for segment prompting

        Returns:
            Extracted information dictionary with _confidence score
        """
        if not frames:
            return {"error": "No frames provided"}

        # Prepare frames
        prepared_frames = prepare_frames_for_prompt(
            frames,
            video_duration,
            include_position_labels=self.include_position_labels,
        )

        # Determine ad type and schema
        ad_type = None

        if self.single_pass:
            # Single-pass: include ad type classification in the main prompt
            schema = get_schema(mode="fixed")
        elif self.schema_mode == "adaptive":
            ad_type = self.detect_ad_type(prepared_frames)
            logger.info(f"Detected ad type: {ad_type}")
            schema = get_schema(mode=self.schema_mode, ad_type=ad_type)
        else:
            schema = get_schema(mode=self.schema_mode, ad_type=ad_type)

        # Build prompt
        if self.single_pass:
            prompt = build_single_pass_prompt(
                prepared_frames,
                video_duration,
                schema,
                include_timestamps=self.include_timestamps,
                include_time_deltas=self.include_time_deltas,
                include_position_labels=self.include_position_labels,
                include_narrative_instructions=self.include_narrative_instructions,
                audio_context=audio_context,
            )
        elif self.segment_prompting and scene_boundaries:
            prompt = build_segmented_prompt(
                prepared_frames,
                video_duration,
                schema,
                scene_boundaries,
                audio_context=audio_context,
            )
        else:
            prompt = build_temporal_prompt(
                prepared_frames,
                video_duration,
                schema,
                include_timestamps=self.include_timestamps,
                include_time_deltas=self.include_time_deltas,
                include_position_labels=self.include_position_labels,
                include_narrative_instructions=self.include_narrative_instructions,
                audio_context=audio_context,
            )

        # Extract with robust JSON parsing
        try:
            response = self.client.extract(prepared_frames, prompt)
            result = _parse_json_response(response)
            
            # Handle case where LLM returns a list (e.g. `[{...}]` instead of `{...}`)
            if isinstance(result, list):
                if len(result) == 1 and isinstance(result[0], dict):
                    result = result[0]
                else:
                    result = {"items": result}

            # Extract ad_type from single-pass response
            if self.single_pass and "ad_type" in result:
                ad_type = result["ad_type"]

            # Compute confidence
            confidence = compute_confidence(
                result,
                audio_context=audio_context,
                num_frames=len(frames),
            )

            # Add metadata
            result["_metadata"] = {
                "ad_type": ad_type,
                "schema_mode": self.schema_mode,
                "single_pass": self.single_pass,
                "num_frames": len(frames),
                "video_duration": video_duration,
                "has_audio_context": audio_context is not None,
                "confidence": confidence,
            }

            return result

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error after all recovery attempts: {e}")
            return {
                "error": "JSON parse error",
                "raw_response": response[:500] if response else "",
                "parse_stages_attempted": (
                    "direct, markdown_strip, brace_extract, "
                    "trailing_comma_fix, truncation_repair"
                ),
                "_metadata": {"confidence": 0.0},
            }
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            return {
                "error": str(e),
                "_metadata": {"confidence": 0.0},
            }


def create_extractor(config: Dict) -> AdExtractor:
    """
    Create AdExtractor from config dict.

    Args:
        config: Configuration dictionary

    Returns:
        Configured AdExtractor instance
    """
    extraction_config = config.get("extraction", {})
    temporal_config = extraction_config.get("temporal_context", {})
    schema_config = extraction_config.get("schema", {})

    return AdExtractor(
        provider=extraction_config.get("provider", "anthropic"),
        model=extraction_config.get("model", "claude-sonnet-4-20250514"),
        max_tokens=extraction_config.get("max_tokens", 2000),
        temperature=extraction_config.get("temperature", 0.0),
        schema_mode=schema_config.get("mode", "adaptive"),
        temporal_context=temporal_config.get("enabled", True),
        include_timestamps=temporal_config.get("include_timestamps", True),
        include_time_deltas=temporal_config.get("include_time_deltas", True),
        include_position_labels=temporal_config.get("include_position_labels", True),
        include_narrative_instructions=temporal_config.get("include_narrative_instructions", True),
        single_pass=extraction_config.get("single_pass", True),
        segment_prompting=extraction_config.get("segment_prompting", False),
        max_retries=extraction_config.get("max_retries", 3),
        retry_delay=extraction_config.get("retry_delay", 1.0),
    )