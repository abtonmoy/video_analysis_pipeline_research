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
