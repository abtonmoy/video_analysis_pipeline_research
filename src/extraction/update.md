# Extraction Module — Changelog

## LLM Client Improvements

### Reliability: Retry with Exponential Backoff
All LLM API calls now use `_retry_with_backoff()` — retries on transient errors (ConnectionError, TimeoutError, rate limits, 5xx) with exponential backoff. Configurable via `max_retries` and `retry_delay` parameters.

### Reliability: Robust JSON Parsing
`_parse_json_response()` handles malformed LLM responses: strips markdown code blocks, extracts JSON from surrounding text, fixes trailing commas. Falls back through 4 parsing strategies before raising.

### Cost: Single-Pass Extraction
Ad type detection and content extraction merged into one LLM call (enabled by default via `single_pass=True`). Eliminates the separate `detect_ad_type()` call, reducing API costs by ~50%.

### Architecture: Gemini Video Client
New `GeminiVideoClient` uploads video files directly to Gemini's native video understanding API, bypassing frame extraction entirely. Available via `provider="gemini_video"`.

### Architecture: Confidence Scoring
`compute_confidence()` scores extraction quality (0.0–1.0) based on schema completeness, audio context availability, and frame count. Score is included in `_metadata.confidence`.

## Prompt Improvements

### Single-Pass Prompt
`build_single_pass_prompt()` adds `ad_type` to the schema, letting the LLM classify and extract in one pass.

### Segment-Level Prompting
`build_segmented_prompt()` groups frames by scene boundaries, giving the LLM better narrative context for longer videos.

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
