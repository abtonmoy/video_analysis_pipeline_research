# Agent Task: Harden VLM Error Resilience in `src/extraction/llm_client.py`

> **Context for the agent:** You are modifying a video advertisement analysis pipeline.
> The file `src/extraction/llm_client.py` is the LLM extraction layer — it sends video
> frames (as base64 images) to LLMs (Anthropic Claude, OpenAI GPT-4V, Google Gemini)
> and parses structured JSON responses. This file is imported by `src/pipeline.py`,
> `src/parallel_pipeline.py`, and `benchmarks/extraction_wrapper.py`. **You must not
> break any of those callers.** No public function signatures change. No imports change
> for consumers. Only internal behavior improves.

---

## Background: Why These Changes Are Needed

When processing hundreds of videos in batch, three failure modes crash the pipeline:

1. **HTTP 429 / 503 from APIs** — Rate limits and server overload. The current retry
   logic catches `ConnectionError`/`TimeoutError`/`OSError` by type, and *separately*
   tries to pattern-match other exceptions by inspecting string representations. This
   misses provider-specific exception classes like `google.api_core.exceptions.ResourceExhausted`.

2. **Truncated JSON** — When the LLM hits `max_output_tokens`, the response ends
   mid-JSON (e.g., `{"brand": {"name": "Nik`). The current 4-stage parser cannot
   recover from this and raises `JSONDecodeError`, which the caller converts to
   `{"error": "JSON parse error"}` — losing all extracted data.

3. **Gemini safety filter silent failures** — Gemini returns empty `candidates` or
   `finish_reason == "SAFETY"` on ad content (alcohol, gambling, beauty ads). The
   current code calls `response.text` which raises an opaque `ValueError`.

All three fixes have been validated in a separate experimental codebase
(`new_experiments/src_copy/`) and are now being ported to the production pipeline.

---

## Architecture You Must Understand

### File Layout

```
src/extraction/
├── llm_client.py    ← YOU MODIFY THIS (and update.md)
├── prompts.py       ← DO NOT MODIFY
├── schema.py        ← DO NOT MODIFY
├── readme.md        ← DO NOT MODIFY
└── update.md        ← APPEND changelog entry
```

### How `llm_client.py` Is Structured (top to bottom)

```
Imports: re, json, time, logging, typing, abc, numpy
         + relative imports from .prompts and .schema

_retry_with_backoff()          ← MODIFY (Change 1)
_parse_json_response()         ← MODIFY (Change 2)

class BaseLLMClient(ABC)       ← DO NOT MODIFY
class AnthropicClient          ← DO NOT MODIFY (except default max_tokens)
class OpenAIClient             ← DO NOT MODIFY (except default max_tokens)
class GeminiClient             ← MODIFY (Changes 3, 4)
class GeminiVideoClient        ← DO NOT MODIFY (video upload client, not used in frame pipeline)
class MockLLMClient            ← DO NOT MODIFY

get_llm_client()               ← MODIFY default max_tokens (Change 4)
compute_confidence()           ← DO NOT MODIFY
class AdExtractor              ← MODIFY error path only (Change 5)
create_extractor()             ← DO NOT MODIFY
```

### Two-Layer Retry Architecture (CRITICAL — Read Carefully)

The pipeline uses **two layers of retry** that must compose correctly:

```
┌────────────────────────────────────────────────────┐
│  OUTER: benchmarks/extraction_wrapper.py._retry()  │
│  - max_retries=5, base_delay=10s, max_delay=120s   │
│  - Catches ANY exception, checks for "429"/"503"   │
│  - Returns {"error": "..."} dict on exhaustion      │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │  INNER: llm_client._retry_with_backoff()     │   │
│  │  - max_retries=3, base_delay=1s, max_delay=30s│  │
│  │  - Catches typed exceptions + string match    │   │
│  │  - RAISES on exhaustion (propagates to outer) │   │
│  └──────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────┘
```

**Critical contract:** The inner layer (`_retry_with_backoff`) must still **RAISE** its
last exception when retries are exhausted. The outer layer in `extraction_wrapper.py`
catches that exception. If you accidentally swallow the exception, the outer layer
never gets a chance to retry with longer delays. **Do not change the raise behavior.**

### How Callers Use This File

```python
# src/pipeline.py and src/parallel_pipeline.py:
from src.extraction.llm_client import AdExtractor, create_extractor
extractor = create_extractor(config)
result = extractor.extract(frames, video_duration, audio_context)
# result is always a dict: {"brand": {...}, ...} or {"error": "...", "_metadata": {...}}

# benchmarks/extraction_wrapper.py:
from src.extraction.llm_client import AdExtractor, create_extractor
# wraps AdExtractor.extract() in its own _retry() with longer delays

# Direct client usage:
from src.extraction.llm_client import get_llm_client
client = get_llm_client("gemini", model="gemini-2.5-flash", max_tokens=4000)
response_text = client.extract(frames, prompt)  # returns raw string
```

---

## Change 1: Harden `_retry_with_backoff()` for 429/503

### Current Code (what exists now)

```python
def _retry_with_backoff(
    func,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple = None,
):
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
```

### What to Change

**Step 1a)** Add `import random` to the file-level imports at the top of the file (it is not currently imported). Place it next to the existing `import time`.

**Step 1b)** Add a helper function **ABOVE** `_retry_with_backoff` that builds the retryable exceptions tuple dynamically. Every SDK import must be wrapped in try/except because not every user has every provider SDK installed:

```python
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
```

**Step 1c)** Replace the entire `_retry_with_backoff` function body with the version below.

Key differences from the original:

- Uses `_get_retryable_exceptions()` as the default instead of hardcoded tuple
- Adds jitter (`random.uniform(0, delay * 0.25)`) to prevent thundering herd when multiple parallel workers retry simultaneously
- `max_delay` default raised from `30.0` → `60.0`
- The string-match fallback (`except Exception`) now also checks for `"resource_exhausted"` and `"unavailable"` patterns
- Logs the HTTP status code when available
- **Still raises `last_exception` on exhaustion** (CRITICAL for outer retry layer)

```python
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
```

### What NOT to Change in This Section

- Do **not** change the function name or parameter names
- Do **not** swallow exceptions on the final attempt
- Do **not** modify `BaseLLMClient.extract()` which calls this function

---

## Change 2: Add Stack-Based JSON Truncation Repair to `_parse_json_response()`

### Current Code (what exists now)

```python
def _parse_json_response(response: str) -> Dict[str, Any]:
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

    raise json.JSONDecodeError(
        f"Could not extract valid JSON from response (length={len(text)})",
        text[:200],
        0,
    )
```

### What to Change

**Step 2a)** Add a new standalone function `_repair_truncated_json()` **ABOVE** `_parse_json_response` (but below `_retry_with_backoff`):

```python
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
```

**Step 2b)** Replace everything from `# Attempt 3` to the final `raise` with the expanded version that includes Attempt 5 (truncation repair) and Attempt 3b (partial response with no closing brace):

```python
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
```

### Why This Is Safe

Truncation repair produces a valid JSON object with **missing fields**, not wrong fields.
The existing `compute_confidence()` function in the same file scores results based on
field completeness (it counts non-null fields / total fields). A repaired object with
half the fields missing will score ~0.25 instead of the normal ~0.7. The `AdExtractor`
already returns this confidence in `result["_metadata"]["confidence"]`, so callers
can filter on quality.

---

## Change 3: Gemini Safety Filter Handling & `finish_reason` Diagnostics

### Current Code (`GeminiClient` only — `GeminiVideoClient` is out of scope)

`GeminiClient._call_api()` ends with:

```python
response = client.generate_content(
    content,
    generation_config={...},
)
return response.text    # ← crashes with ValueError if safety-blocked
```

### What to Change

**Step 3a)** Add a permissive safety settings constant **near the top of the file**, right after the `logger = logging.getLogger(__name__)` line:

```python
# ---------------------------------------------------------------------------
# Gemini safety settings — BLOCK_NONE to prevent false positives on ad content
# (alcohol, gambling, beauty products routinely trigger overzealous filters)
# ---------------------------------------------------------------------------
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
```

**Step 3b)** Add a new helper function `_extract_gemini_response()`. Place it **after** `_repair_truncated_json()` and **before** `class BaseLLMClient`:

```python
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
```

**Step 3c)** Modify **`GeminiClient._call_api()`**. Find the end of this method where it calls `generate_content` and returns `response.text`. Replace:

```python
        response = client.generate_content(
            content,
            generation_config=generation_config,
        )

        return response.text
```

With:

```python
        # Add safety settings to prevent false-positive blocks on ad content
        gen_kwargs = {"generation_config": generation_config}
        if GEMINI_SAFETY_PERMISSIVE is not None:
            gen_kwargs["safety_settings"] = GEMINI_SAFETY_PERMISSIVE

        response = client.generate_content(content, **gen_kwargs)

        return _extract_gemini_response(response)
```

### How Safety-Blocked Responses Flow Through the System (No Other Changes Needed)

```
GeminiClient._call_api()
  → _extract_gemini_response(response)
  → returns '{"error": "SAFETY_BLOCKED", ...}'    ← valid JSON string
  → BaseLLMClient.extract() returns this string
  → AdExtractor.extract() calls _parse_json_response(response)
  → parses successfully into {"error": "SAFETY_BLOCKED", ...}
  → AdExtractor sees this is a dict, NOT a JSONDecodeError
  → compute_confidence({"error": ...}) returns 0.0
  → result["_metadata"]["confidence"] = 0.0
  → caller receives a structured error dict, NOT a crash
```

---

## Change 4: Increase Default `max_output_tokens`

The default across all constructors and the factory function is currently `2000`.
The config files (`config/default.yaml`, `config/benchmark.yaml`) already specify
`max_tokens: 4000`. Align the code defaults so direct programmatic usage (tests,
scripts) also gets the larger budget.

### What to Change (5 locations, same change each time)

Change `max_tokens: int = 2000` → `max_tokens: int = 4000` in:

1. **`AnthropicClient.__init__`** parameter default
2. **`OpenAIClient.__init__`** parameter default
3. **`GeminiClient.__init__`** parameter default
4. **`get_llm_client()`** function parameter default
5. **`AdExtractor.__init__`** parameter default

**Do NOT modify `GeminiVideoClient`** — it is the video upload client and is out of scope.

**Why this is safe:** Any config-driven usage (the normal case) already overrides this
default with the value from yaml. This only affects direct usage like
`get_llm_client("gemini", model="gemini-2.5-flash")` with no explicit max_tokens.

---

## Change 5: Improve `AdExtractor.extract()` Error Diagnostics

### Current Code

Inside `AdExtractor.extract()`, find:

```python
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return {
                "error": "JSON parse error",
                "raw_response": response[:500],
                "_metadata": {"confidence": 0.0},
            }
```

### Replace With

```python
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
```

---

## Change 6: Append Changelog to `src/extraction/update.md`

Add this block at the **end** of the existing file (after the existing content):

```markdown

## VLM Error Resilience (ported from new_experiments/src_copy)

### Retry: 429/503 Transient Error Hardening
`_retry_with_backoff()` now catches provider-specific rate-limit exceptions
(`google.api_core.exceptions.ResourceExhausted`, `anthropic.RateLimitError`,
`openai.RateLimitError`) by type instead of relying solely on string matching.
Added jitter to backoff delays to prevent thundering herd in parallel workers.
Default `max_delay` increased from 30s to 60s. The inner retry still raises on
exhaustion so the outer retry layer in `extraction_wrapper.py` can take over.

### JSON: Stack-Based Truncation Repair
`_parse_json_response()` gains a 5th recovery stage: `_repair_truncated_json()`.
When `max_output_tokens` truncates the response mid-JSON, the parser tracks open
braces/brackets/strings with a character-level stack and appends closing characters
to produce a parseable (partial) object. Also handles the edge case where truncation
is so severe that no closing brace exists in the response at all. Partial extractions
score low via the existing `compute_confidence()` mechanism.

### Gemini: Safety Filter Diagnostics
New `_extract_gemini_response()` helper replaces raw `response.text` access in
`GeminiClient`. Detects `finish_reason == SAFETY` and empty candidates, returning
structured `{"error": "SAFETY_BLOCKED"}` JSON instead of crashing with `ValueError`.
Logs safety rating categories and probabilities.

### Gemini: Permissive Safety Settings
`GeminiClient.generate_content()` calls now pass `BLOCK_NONE` for all four harm
categories via `GEMINI_SAFETY_PERMISSIVE`. This prevents false-positive blocks on
ad content (alcohol, gambling, beauty, pharmaceutical categories). Guarded behind
try/except ImportError for SDK compatibility. `GeminiVideoClient` is unchanged
(out of scope — frame-based pipeline only).

### Defaults: Token Headroom
Default `max_output_tokens` raised from 2000 to 4000 across client constructors
(`AnthropicClient`, `OpenAIClient`, `GeminiClient`), the `get_llm_client()` factory,
and `AdExtractor`. Aligns code defaults with `config/default.yaml` which already
specified 4000.
```

---

## Verification Checklist

After implementing **all 6 changes**, run these checks:

### Automated (run these commands)

```bash
# 1. Import test — must not crash even without API keys
python -c "from src.extraction.llm_client import AdExtractor, get_llm_client, create_extractor, _parse_json_response, _repair_truncated_json; print('All imports OK')"

# 2. Truncation repair test
python -c "
from src.extraction.llm_client import _parse_json_response
# Truncated mid-string
r = _parse_json_response('{\"brand\": {\"name\": \"Nik')
assert 'brand' in r, f'Expected brand key, got {r}'
print(f'Test 1 passed: {r}')

# Truncated mid-object
r = _parse_json_response('{\"a\": 1, \"b\": {\"c\": 2')
assert r['a'] == 1, f'Expected a=1, got {r}'
print(f'Test 2 passed: {r}')

# Trailing comma
r = _parse_json_response('{\"a\": 1, \"b\": 2, }')
assert r == {'a': 1, 'b': 2}, f'Expected dict, got {r}'
print(f'Test 3 passed: {r}')

# Normal JSON still works
r = _parse_json_response('{\"ok\": true}')
assert r == {'ok': True}
print(f'Test 4 passed: {r}')

print('All JSON parsing tests passed!')
"

# 3. Mock client still works
python -c "
from src.extraction.llm_client import get_llm_client
client = get_llm_client('mock', model='test')
# MockLLMClient doesn't use frames, but extract() expects FrameForPrompt objects
# Just verify the client was created without errors
print(f'Mock client created: {type(client).__name__}')
"

# 4. Safety settings loaded (only if google SDK installed)
python -c "
from src.extraction.llm_client import GEMINI_SAFETY_PERMISSIVE
if GEMINI_SAFETY_PERMISSIVE is not None:
    print(f'Safety settings loaded: {len(GEMINI_SAFETY_PERMISSIVE)} categories')
else:
    print('Safety settings skipped (google SDK not installed) — this is OK')
"

# 5. Retry helper built
python -c "
from src.extraction.llm_client import _get_retryable_exceptions
exc = _get_retryable_exceptions()
print(f'Retryable exceptions: {len(exc)} types')
for e in exc:
    print(f'  - {e.__module__}.{e.__name__}')
"
```

### Manual Review

- [ ] `_retry_with_backoff` still has `raise last_exception` as its final line
- [ ] `import random` is in the file-level imports
- [ ] `_get_retryable_exceptions` is defined **above** `_retry_with_backoff`
- [ ] `_repair_truncated_json` is defined **above** `_parse_json_response`
- [ ] `_extract_gemini_response` is defined **above** `class BaseLLMClient`
- [ ] `GEMINI_SAFETY_PERMISSIVE` is defined near the top, after logger
- [ ] `GeminiClient._call_api()` passes safety settings and uses `_extract_gemini_response()`
- [ ] `GeminiVideoClient` is **untouched** (video upload client, out of scope)
- [ ] `max_tokens` default is `4000` in all 5 locations (Anthropic, OpenAI, Gemini, get_llm_client, AdExtractor)
- [ ] `src/extraction/update.md` has the new changelog block appended
- [ ] No changes to: `prompts.py`, `schema.py`, `pipeline.py`, `parallel_pipeline.py`, `extraction_wrapper.py`

---

## Files Modified

| File | What Changes |
|------|-------------|
| `src/extraction/llm_client.py` | Changes 1–5: retry hardening, JSON repair, Gemini safety, token defaults, error diagnostics |
| `src/extraction/update.md` | Change 6: append changelog entry |

## Files NOT Modified (must still work after your changes)

| File | Why It's Unchanged |
|------|-------------------|
| `src/extraction/prompts.py` | No interface changes |
| `src/extraction/schema.py` | No interface changes |
| `src/pipeline.py` | Uses `create_extractor()` → `AdExtractor` — interface unchanged |
| `src/parallel_pipeline.py` | Same as above |
| `benchmarks/extraction_wrapper.py` | Outer retry layer composes with inner — no changes needed |
| `config/default.yaml` | Already specifies `max_tokens: 4000` |
| `config/benchmark.yaml` | Already specifies `max_tokens: 4000` |

---

# New Component: Unified Benchmarks (Added 2026-03-25)

## Overview

Created `unified_benchmarks/` directory — a standardized benchmarking suite that compares 7 baseline frame selection methods against AdaFrame and outputs results in the same format as `main_results` and `new_experiments`.

## Directory Structure

```
unified_benchmarks/
├── README.md                    # Comprehensive documentation
├── config.yaml                  # Configuration
├── ub_src/                      # Renamed to avoid conflict with parent src/
│   ├── __init__.py
│   ├── retry_utils.py          # Error handling with retry queue (from new_experiments)
│   ├── methods.py              # 7 baseline methods (uniform, random, histogram, ORB, optical_flow, clip_only, kmeans)
│   ├── metrics.py              # Metric computation (frames, cost, accuracy, efficiency)
│   ├── extraction_wrapper.py   # VLM wrapper (bare + full extraction)
│   ├── runner.py               # Main orchestrator
│   ├── aggregation.py          # Summary statistics
│   └── visualization.py        # Figure generation
├── scripts/
│   ├── run_benchmark.py        # CLI entry point
│   └── test_structure.py       # Structure verification
├── results/
│   ├── per_video/              # Per-video JSON results
│   ├── figures/                # Generated figures
│   └── retry_queue/            # Failed video queue
```

## Data Structure

### Per-Video Results (matches main_results format)

```json
{
  "metadata": {
    "timestamp": "2026-03-25T...",
    "n_videos": 500,
    "n_methods": 7
  },
  "results": [
    {
      "status": "success",
      "video_path": "...",
      "video_name": "...",
      "processed_at": "...",
      "method": "uniform_1fps",
      "metadata": {
        "duration": 20.02,
        "fps": 29.97,
        "total_frames": 600
      },
      "pipeline_stats": {
        "final_frame_count": 32,
        "total_frames_sampled": 600,
        "reduction_rate": 0.947,
        "compression_ratio": 18.75,
        "selection_latency_s": 0.02,
        "info_density": 0.31,
        "cost_usd": 0.48
      },
      "extraction": {...},
      "comparison": {
        "topic_match": true,
        "brand_match": true,
        "cta_match": false
      }
    }
  ]
}
```

### Summary (matches new_experiments format)

```json
{
  "n_videos": 500,
  "n_success": 484,
  "n_failed": 16,
  "n_methods": 7,
  "methods": {
    "uniform_1fps": {
      "n_videos": 484,
      "frames": {
        "mean": 43.9,
        "median": 42.0,
        "std": 12.3,
        "min": 5,
        "max": 105
      },
      "cost_usd": {
        "mean": 0.504,
        "median": 0.483,
        "total": 243.94
      },
      "accuracy": {
        "topic_accuracy": 84.3,
        "super_category_accuracy": 89.9,
        "brand_detection_rate": 92.1,
        "cta_detection_rate": 81.8
      },
      "efficiency": {
        "mean_compression_ratio": 24.5,
        "mean_info_density": 0.267,
        "mean_reduction_rate": 0.947
      }
    }
  }
}
```

## Baseline Methods

| Method | Description | GPU Required |
|--------|-------------|--------------|
| `uniform_1fps` | Sample 1 frame per second | No |
| `random` | Random frame selection (k from pipeline) | No |
| `histogram` | HSV histogram change detection (threshold=0.95) | No |
| `orb` | ORB feature matching (40 good matches) | No |
| `optical_flow` | Farneback motion peaks (85th percentile) | No |
| `clip_only` | CLIP sequential dedup (cosine=0.92) | Yes |
| `kmeans` | K-means on CLIP embeddings (1 cluster per 3s) | Yes |

## Key Features

1. **Same error handling as new_experiments**: Retry queue with immediate retries, exponential backoff, JSONL failure log
2. **Dual extraction modes**: Bare (1 LLM call) and Full (2 LLM calls with adaptive schema)
3. **Comprehensive metrics**: Frame counts, cost, accuracy, efficiency, info density
4. **Automatic figure generation**: Method comparison, accuracy vs cost, distributions
5. **CSV export**: For easy analysis in pandas/R

## Usage

```bash
# Run all methods on all videos
python unified_benchmarks/scripts/run_benchmark.py \
    --video_dir /path/to/videos \
    --pipeline_results /path/to/filtered_results.json \
    --output_dir ./results

# Run specific methods
python unified_benchmarks/scripts/run_benchmark.py \
    --video_dir /path/to/videos \
    --pipeline_results /path/to/results.json \
    --methods uniform_1fps random histogram \
    --output_dir ./results

# Frame selection only (no VLM calls, free)
python unified_benchmarks/scripts/run_benchmark.py \
    --video_dir /path/to/videos \
    --pipeline_results /path/to/results.json \
    --selection_only \
    --output_dir ./results

# Generate figures from existing results
python unified_benchmarks/scripts/run_benchmark.py \
    --generate_figures \
    --output_dir ./results
```

## Output Files (Per-Method Folders)

```
results/
├── uniform_1fps/
│   ├── results.json          # Per-video results for this method
│   ├── summary.json          # Aggregated stats for this method
│   └── results.csv
├── random/
│   └── ...
├── histogram/
├── orb/
├── optical_flow/
├── clip_only/
├── kmeans/
├── combined/                 # All methods together
│   ├── all_results.json
│   ├── all_summary.json
│   └── all_results.csv
└── figures/
    └── *.png
```

Each method folder contains:
- `results.json` — Per-video results (main_results format)
- `summary.json` — Aggregated statistics (new_experiments format)
- `results.csv` — CSV for easy analysis

The `combined/` folder has all methods together for cross-method comparison.

## Integration with Existing Code

The unified benchmarks imports from existing codebase:
- `src.deduplication.clip_embed.CLIPDeduplicator` — For CLIP embeddings
- `src.extraction.llm_client.AdExtractor` — For VLM extraction
- `src.extraction.llm_client.create_extractor` — For extractor factory

No modifications to existing files required.

## Comparison with Other Benchmark Directories

| Directory | Purpose | Output Format | Status |
|-----------|---------|---------------|--------|
| `benchmark/` | Early/simple testing | Custom | Legacy |
| `benchmarks/` | 7-method comparison | Nested dict | Active |
| `experiments/` | Development scripts | Mixed | Development |
| `new_experiments/` | 8 paper validations | Aggregated JSON | Active |
| `unified_benchmarks/` | Standardized benchmarking | main_results + new_experiments | **New** |

## Files Created

| File | Purpose |
|------|---------|
| `unified_benchmarks/README.md` | Documentation |
| `unified_benchmarks/config.yaml` | Configuration |
| `unified_benchmarks/ub_src/retry_utils.py` | Error handling |
| `unified_benchmarks/ub_src/methods.py` | 7 baseline methods |
| `unified_benchmarks/ub_src/metrics.py` | Metric computation |
| `unified_benchmarks/ub_src/extraction_wrapper.py` | VLM wrapper |
| `unified_benchmarks/ub_src/runner.py` | Main orchestrator |
| `unified_benchmarks/ub_src/aggregation.py` | Summary statistics |
| `unified_benchmarks/ub_src/visualization.py` | Figure generation |
| `unified_benchmarks/scripts/run_benchmark.py` | CLI entry point |
| `unified_benchmarks/scripts/test_structure.py` | Verification |
