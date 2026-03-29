# Extraction Module

LLM-based content extraction from video advertisement frames.

## Modules

### `llm_client.py`
Multi-provider LLM client with production-grade features.

**Providers:**
| Provider | Class | Description |
|----------|-------|-------------|
| `anthropic` | `AnthropicClient` | Claude API (default) |
| `openai` | `OpenAIClient` | GPT-4V |
| `gemini` | `GeminiClient` | Gemini with JSON mode |
| `gemini_video` | `GeminiVideoClient` | Gemini native video upload |
| `mock` | `MockLLMClient` | Testing without API calls |

**Features:**
- **Retry with backoff** — automatic retry on transient errors, rate limits, 5xx
- **Robust JSON parsing** — strips markdown, extracts from surrounding text, fixes trailing commas
- **Single-pass extraction** — ad type + content in one LLM call (~50% cost reduction)
- **Confidence scoring** — 0.0–1.0 score based on schema completeness

**Key API:**
```python
# Factory function
client = get_llm_client("anthropic", model="claude-sonnet-4-20250514", max_retries=3)

# AdExtractor (main entry point)
extractor = AdExtractor(provider="anthropic", single_pass=True)
result = extractor.extract(frames, video_duration, audio_context=audio)
print(result["_metadata"]["confidence"])  # 0.0 - 1.0
```

### `prompts.py`
Prompt construction for LLM extraction. Includes temporal context, topic/sentiment taxonomies, and multiple prompt modes.

**Prompt Types:**
- `build_temporal_prompt()` — standard chronological frame prompt
- `build_single_pass_prompt()` — merged type detection + extraction
- `build_segmented_prompt()` — groups frames by scene for narrative context
- `build_type_detection_prompt()` — ad type classification only

### `schema.py`
Extraction schema definitions with adaptive mode (adjusts fields based on detected ad type).
