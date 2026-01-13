#src\extraction\llm_client.py
"""
LLM clients and extraction orchestration with audio context support.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np

from .prompts import FrameForPrompt, prepare_frames_for_prompt, build_temporal_prompt, build_type_detection_prompt
from .schema import get_schema, get_valid_ad_types

logger = logging.getLogger(__name__)


# ============================================================================
# LLM Clients
# ============================================================================

class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def extract(
        self,
        frames: List[FrameForPrompt],
        prompt: str
    ) -> str:
        """Send frames and prompt to LLM, return response text."""
        pass


class AnthropicClient(BaseLLMClient):
    """Claude API client."""
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 2000,
        temperature: float = 0.0
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client
    
    def extract(
        self,
        frames: List[FrameForPrompt],
        prompt: str
    ) -> str:
        client = self._get_client()
        
        # Build content with images
        content = []
        
        for frame in frames:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": frame.base64_image
                }
            })
        
        content.append({
            "type": "text",
            "text": prompt
        })
        
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": content}]
        )
        
        return response.content[0].text


class OpenAIClient(BaseLLMClient):
    """OpenAI GPT-4V client."""
    
    def __init__(
        self,
        model: str = "gpt-4o",
        max_tokens: int = 2000,
        temperature: float = 0.0
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client
    
    def extract(
        self,
        frames: List[FrameForPrompt],
        prompt: str
    ) -> str:
        client = self._get_client()
        
        # Build content with images
        content = []
        
        for frame in frames:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame.base64_image}"
                }
            })
        
        content.append({
            "type": "text",
            "text": prompt
        })
        
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": content}]
        )
        
        return response.choices[0].message.content


class GeminiClient(BaseLLMClient):
    """Google Gemini API client."""
    
    def __init__(
        self,
        model: str = "gemini-3.0-flash-exp",
        max_tokens: int = 2000,
        temperature: float = 0.0
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            import google.generativeai as genai
            import os
            
            # Configure with API key from environment
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
            self._client = genai.GenerativeModel(self.model)
        return self._client
    
    def extract(
        self,
        frames: List[FrameForPrompt],
        prompt: str
    ) -> str:
        import base64
        from PIL import Image
        from io import BytesIO
        
        client = self._get_client()
        
        # Build content with images
        content = []
        
        for frame in frames:
            # Decode base64 to PIL Image
            image_data = base64.b64decode(frame.base64_image)
            pil_image = Image.open(BytesIO(image_data))
            content.append(pil_image)
        
        # Add text prompt at the end
        content.append(prompt)
        
        # Generate response
        response = client.generate_content(
            content,
            generation_config={
                "max_output_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
        )
        
        return response.text


class MockLLMClient(BaseLLMClient):
    """Mock client for testing without API calls."""
    
    def extract(
        self,
        frames: List[FrameForPrompt],
        prompt: str
    ) -> str:
        return json.dumps({
            "brand": {
                "brand_name_text": "Test Brand",
                "logo_visible": True,
                "brand_text_contrast": "high"
            },
            "product": {
                "product_name": "Test Product",
                "industry": "technology"
            },
            "promotion": {
                "promo_present": True,
                "promo_text": "50% off",
                "promo_deadline": "limited time",
                "price_value": "$9.99"
            },
            "call_to_action": {
                "cta_present": True,
                "cta_type": "Sign up button"
            },
            "visual_elements": {
                "text_density": "medium"
            },
            "content_rating": {
                "is_nsfw": False
            },
            "_mock": True,
            "_num_frames": len(frames)
        })


def get_llm_client(
    provider: str,
    model: str,
    max_tokens: int = 2000,
    temperature: float = 0.0
) -> BaseLLMClient:
    """
    Factory function to get LLM client.
    
    Args:
        provider: "anthropic", "openai", "gemini", or "mock"
        model: Model name
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        
    Returns:
        LLM client instance
    """
    if provider == "anthropic":
        return AnthropicClient(model=model, max_tokens=max_tokens, temperature=temperature)
    elif provider == "openai":
        return OpenAIClient(model=model, max_tokens=max_tokens, temperature=temperature)
    elif provider == "gemini":
        return GeminiClient(model=model, max_tokens=max_tokens, temperature=temperature)
    elif provider == "mock":
        return MockLLMClient()
    else:
        raise ValueError(f"Unknown provider: {provider}. Choose from: anthropic, openai, gemini, mock")


# ============================================================================
# Extractor
# ============================================================================

class AdExtractor:
    """
    Main extractor class with adaptive schema support and audio context.
    """
    
    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 2000,
        temperature: float = 0.0,
        schema_mode: str = "adaptive",  # adaptive, fixed, flexible
        temporal_context: bool = True,
        include_timestamps: bool = True,
        include_time_deltas: bool = True,
        include_position_labels: bool = True,
        include_narrative_instructions: bool = True
    ):
        self.client = get_llm_client(provider, model, max_tokens, temperature)
        self.schema_mode = schema_mode
        self.temporal_context = temporal_context
        self.include_timestamps = include_timestamps
        self.include_time_deltas = include_time_deltas
        self.include_position_labels = include_position_labels
        self.include_narrative_instructions = include_narrative_instructions
    
    def detect_ad_type(
        self,
        frames: List[FrameForPrompt]
    ) -> str:
        """
        Detect ad type from frames.
        
        Args:
            frames: Prepared frames
            
        Returns:
            Ad type string
        """
        prompt = build_type_detection_prompt()
        
        try:
            response = self.client.extract(frames, prompt)
            ad_type = response.strip().lower().replace(" ", "_")
            
            # Validate
            valid_types = get_valid_ad_types()
            if ad_type in valid_types:
                return ad_type
            
            # Try to match partial
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
        audio_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Extract structured information from ad frames.
        
        Args:
            frames: List of (timestamp, frame) tuples
            video_duration: Total video duration
            audio_context: Optional dict with audio transcription and features
            
        Returns:
            Extracted information dictionary
        """
        if not frames:
            return {"error": "No frames provided"}
        
        # Prepare frames
        prepared_frames = prepare_frames_for_prompt(
            frames,
            video_duration,
            include_position_labels=self.include_position_labels
        )
        
        # Detect ad type if adaptive
        ad_type = None
        if self.schema_mode == "adaptive":
            ad_type = self.detect_ad_type(prepared_frames)
            logger.info(f"Detected ad type: {ad_type}")
        
        # Get schema
        schema = get_schema(mode=self.schema_mode, ad_type=ad_type)
        
        # Build prompt with audio context
        prompt = build_temporal_prompt(
            prepared_frames,
            video_duration,
            schema,
            include_timestamps=self.include_timestamps,
            include_time_deltas=self.include_time_deltas,
            include_position_labels=self.include_position_labels,
            include_narrative_instructions=self.include_narrative_instructions,
            audio_context=audio_context
        )
        
        # Extract
        try:
            response = self.client.extract(prepared_frames, prompt)
            
            # Parse JSON
            # Handle potential markdown code blocks
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            
            result = json.loads(response)
            
            # Add metadata
            result["_metadata"] = {
                "ad_type": ad_type,
                "schema_mode": self.schema_mode,
                "num_frames": len(frames),
                "video_duration": video_duration,
                "has_audio_context": audio_context is not None
            }
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            return {
                "error": "JSON parse error",
                "raw_response": response[:500]
            }
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            return {"error": str(e)}


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
        include_narrative_instructions=temporal_config.get("include_narrative_instructions", True)
    )