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
# Retry Utility
# ============================================================================

def _retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple = None,
):
    """
    Execute a function with exponential backoff retry.

    Args:
        func: Callable to execute
        max_retries: Number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        retryable_exceptions: Tuple of exception types to retry on

    Returns:
        Function result

    Raises:
        Last exception if all retries exhausted
    """
    if retryable_exceptions is None:
        retryable_exceptions = (
            ConnectionError,
            TimeoutError,
            OSError,
        )

    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except retryable_exceptions as e:
            last_exception = e
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(
                    f"Retry {attempt + 1}/{max_retries} after {type(e).__name__}: {e}. "
                    f"Waiting {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                logger.error(f"All {max_retries} retries exhausted: {e}")
        except Exception as e:
            # Check for HTTP status code errors (rate limit, server error)
            error_str = str(e).lower()
            status_code = getattr(e, "status_code", None) or getattr(e, "code", None)

            is_retryable = (
                status_code in (429, 500, 502, 503, 504)
                or "rate limit" in error_str
                or "too many requests" in error_str
                or "server error" in error_str
                or "overloaded" in error_str
                or "timeout" in error_str
            )

            if is_retryable and attempt < max_retries:
                last_exception = e
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(
                    f"Retry {attempt + 1}/{max_retries} after {type(e).__name__}: {e}. "
                    f"Waiting {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                raise

    raise last_exception


# ============================================================================
# JSON Parsing
# ============================================================================

def parse_json_response(text: str) -> Dict[str, Any]:
    """
    Extract and parse JSON from LLM response text.
    Handles markdown blocks, prefix/suffix text, and truncated JSON.
    - Markdown code blocks (```json ... ```)
    - Nested code blocks
    - Trailing commas
    - Explanatory text before/after JSON

    Args:
        text: Raw LLM response text

    Returns:
        Parsed dictionary

    Raises:
        json.JSONDecodeError if no valid JSON found
    """
    text = text.strip()

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

    # Attempt 3: Find JSON object or array with regex (outermost braces/brackets)
    brace_match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
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

    # Attempt 5: Robust autoclose for truncated JSON
    try:
        # Simple stack-based autoclose
        stack = []
        in_string = False
        escaped = False
        
        # Only try if it looks like JSON (starts with { or [)
        clean_text = text.strip()
        if clean_text.startswith('{') or clean_text.startswith('['):
            for i, char in enumerate(clean_text):
                if char == '"' and not escaped:
                    in_string = not in_string
                if in_string:
                    if char == '\\' and not escaped:
                        escaped = True
                    else:
                        escaped = False
                    continue
                
                if char == '{':
                    stack.append('}')
                elif char == '[':
                    stack.append(']')
                elif char == '}':
                    if stack and stack[-1] == '}':
                        stack.pop()
                elif char == ']':
                    if stack and stack[-1] == ']':
                        stack.pop()
            
            fixed = clean_text
            if in_string:
                fixed += '"'
            # Remove trailing commas before closing
            fixed = re.sub(r",\s*$", "", fixed)
            while stack:
                fixed += stack.pop()
                
            return json.loads(fixed)
    except Exception:
        pass

    # Final attempt: log and fail
    logger.error(f"JSON Parsing Failed. Raw Response (first 1000 chars): {text[:1000]}")
    if len(text) > 1000:
        logger.error(f"Raw Response (last 1000 chars): {text[-1000:]}")

    raise json.JSONDecodeError(
        f"Could not extract valid JSON from response (length={len(text)})",
        text,
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
        model: str = "gemini-3-flash-preview",
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
            from google import genai
            import os

            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            # The new SDK uses a unified Client
            self._client = genai.Client(api_key=api_key)
        return self._client

    def _call_api(self, frames: List[FrameForPrompt], prompt: str) -> str:
        from google.genai import types
        import base64
        from PIL import Image
        from io import BytesIO

        client = self._get_client()

        # Build contents list (images + text)
        contents = []
        for frame in frames:
            image_data = base64.b64decode(frame.base64_image)
            pil_image = Image.open(BytesIO(image_data))
            contents.append(pil_image)

        contents.append(prompt)

        # Build config
        config = types.GenerateContentConfig(
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
            response_mime_type="application/json" if self.json_mode else None,
            safety_settings=[
                types.SafetySetting(
                    category=cat,
                    threshold="BLOCK_NONE",
                )
                for cat in [
                    "HARM_CATEGORY_HARASSMENT",
                    "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "HARM_CATEGORY_CIVIC_INTEGRITY",
                ]
            ]
        )

        try:
            response = client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config
            )

            # Check for safety blocks or empty response
            if not response.candidates:
                return json.dumps({
                    "error": "SAFETY_BLOCKED",
                    "message": "The response was blocked by safety filters (no candidates)."
                })

            if any(candidate.finish_reason == "SAFETY" for candidate in response.candidates):
                 return json.dumps({
                    "error": "SAFETY_BLOCKED",
                    "message": "The response was blocked by safety filters (SAFETY reason)."
                })

            # Diagnostics for truncation
            candidate = response.candidates[0]
            if candidate.finish_reason != "STOP":
                logger.warning(f"Gemini response finished with reason: {candidate.finish_reason}")
            
            if hasattr(response, 'usage_metadata'):
                logger.debug(f"Tokens: prompt={response.usage_metadata.prompt_token_count}, candidates={response.usage_metadata.candidates_token_count}")

            return response.text
        except Exception as e:
            # Handle potential safety blocks and API errors
            err_msg = str(e)
            if "safety" in err_msg.lower() or "blocked" in err_msg.lower():
                return json.dumps({
                    "error": "SAFETY_BLOCKED",
                    "message": f"The response was blocked by safety filters. Error: {err_msg}"
                })
            logger.error(f"Gemini API call failed: {e}")
            raise


class GeminiVideoClient(BaseLLMClient):
    """
    Gemini client that uploads video directly instead of sending frames.

    Uses Gemini's native video understanding — eliminates frame extraction overhead
    and lets the model choose its own optimal sampling rate.
    """

    def __init__(
        self,
        model: str = "gemini-3-flash-preview",
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
            from google import genai
            import os
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def extract_from_video(self, video_path: str, prompt: str) -> str:
        from google.genai import types
        import time

        def _do_call():
            client = self._get_client()

            # Upload video file
            logger.info(f"Uploading video to Gemini: {video_path}")
            video_file = client.files.upload(file=video_path)

            # Wait for processing
            while video_file.state.name == "PROCESSING":
                logger.debug("Waiting for video processing...")
                time.sleep(2)
                video_file = client.files.get(name=video_file.name)

            if video_file.state.name == "FAILED":
                raise RuntimeError(f"Video processing failed: {video_file.state.name}")

            # Build config with safety and JSON mode
            config = types.GenerateContentConfig(
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
                response_mime_type="application/json",
                safety_settings=[
                    types.SafetySetting(category=cat, threshold="BLOCK_NONE")
                    for cat in [
                        "HARM_CATEGORY_HARASSMENT",
                        "HARM_CATEGORY_HATE_SPEECH",
                        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "HARM_CATEGORY_CIVIC_INTEGRITY",
                    ]
                ]
            )

            # Generate with video + prompt
            response = client.models.generate_content(
                model=self.model,
                contents=[video_file, prompt],
                config=config
            )
            
            # Check for safety blocks
            try:
                if not response.candidates:
                    return json.dumps({
                        "error": "SAFETY_BLOCKED", 
                        "message": "The response was blocked by safety filters (no candidates)."
                    })
                    
                if any(cand.finish_reason == "SAFETY" for cand in response.candidates):
                    return json.dumps({"error": "SAFETY_BLOCKED", "message": "The response was blocked by safety filters (SAFETY reason)."})
                text = response.text
            except Exception as e:
                text = json.dumps({"error": "SAFETY_BLOCKED", "message": f"Safety analysis failed: {str(e)}"})

            # Clean up uploaded file
            try:
                client.files.delete(name=video_file.name)
            except Exception:
                pass

            return text

        return _retry_with_backoff(
            func=_do_call,
            max_retries=self.max_retries,
            base_delay=self.retry_delay,
        )

    def _call_api(self, frames: List[FrameForPrompt], prompt: str) -> str:
        """Fallback: use frame-based extraction if called via standard interface."""
        from google.genai import types
        import base64
        from PIL import Image
        from io import BytesIO

        client = self._get_client()
        contents = []

        for frame in frames:
            image_data = base64.b64decode(frame.base64_image)
            pil_image = Image.open(BytesIO(image_data))
            contents.append(pil_image)

        contents.append(prompt)

        config = types.GenerateContentConfig(
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
            response_mime_type="application/json",
        )

        response = client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config
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
    max_tokens: int = 2000,
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
        max_tokens: int = 2000,
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
            result = parse_json_response(response)
            
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
            logger.error(f"JSON parse error: {e}")
            return {
                "error": "JSON parse error",
                "raw_response": response[:500],
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