#src\extraction\prompts.py
"""
Prompt building for LLM extraction.
"""

import json
import base64
from io import BytesIO
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from PIL import Image
import cv2


@dataclass
class FrameForPrompt:
    """Frame prepared for LLM prompt."""
    timestamp: float
    base64_image: str
    position_label: Optional[str] = None  # OPENING, CLOSING, etc.


def frame_to_base64(frame: np.ndarray, max_size: int = 512) -> str:
    """
    Convert frame to base64 for API.
    
    Args:
        frame: Frame as BGR numpy array
        max_size: Maximum dimension (resized if larger)
        
    Returns:
        Base64-encoded JPEG string
    """
    # Resize for API efficiency
    h, w = frame.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    
    # Convert to PIL and encode
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=85)
    
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def prepare_frames_for_prompt(
    frames: List[Tuple[float, np.ndarray]],
    video_duration: float,
    include_position_labels: bool = True
) -> List[FrameForPrompt]:
    """
    Prepare frames with metadata for prompt.
    
    Args:
        frames: List of (timestamp, frame) tuples
        video_duration: Total video duration in seconds
        include_position_labels: Whether to add OPENING/MIDDLE/CLOSING labels
        
    Returns:
        List of FrameForPrompt objects
    """
    prepared = []
    
    for ts, frame in frames:
        position_label = None
        
        if include_position_labels:
            position = ts / video_duration if video_duration > 0 else 0
            if position < 0.15:
                position_label = "OPENING"
            elif position > 0.85:
                position_label = "CLOSING"
            elif 0.4 < position < 0.6:
                position_label = "MIDDLE"
        
        prepared.append(FrameForPrompt(
            timestamp=ts,
            base64_image=frame_to_base64(frame),
            position_label=position_label
        ))
    
    return prepared


def build_temporal_prompt(
    frames: List[FrameForPrompt],
    video_duration: float,
    schema: Dict,
    include_timestamps: bool = True,
    include_time_deltas: bool = True,
    include_position_labels: bool = True,
    include_narrative_instructions: bool = True,
    audio_context: Optional[Dict] = None
) -> str:
    """
    Build a temporally-aware prompt for LLM extraction.
    
    Args:
        frames: Prepared frames with base64 images
        video_duration: Total video duration
        schema: Schema dictionary for extraction
        include_timestamps: Whether to show timestamps
        include_time_deltas: Whether to show time gaps between frames
        include_position_labels: Whether to show OPENING/CLOSING labels
        include_narrative_instructions: Whether to include narrative analysis instructions
        audio_context: Optional dict with audio transcription and features
        
    Returns:
        Prompt string
    """
    prompt = f"""You are analyzing a {video_duration:.1f}-second video advertisement through {len(frames)} keyframes.

The frames are in CHRONOLOGICAL ORDER. Analyze both individual frames AND the narrative progression.

"""
    
    if include_narrative_instructions:
        prompt += """ANALYSIS APPROACH:
1. Identify what CHANGES between frames (scene transitions, new elements, text)
2. Track the NARRATIVE ARC (setup → development → conclusion/CTA)
3. Note RECURRING ELEMENTS (logo appearances, product shots, faces)
4. Consider the PACING (fast cuts = energy, slow shots = emotion)

"""
    
    prompt += "TEMPORAL CONTEXT:\n"
    
    prev_ts = 0
    for i, frame in enumerate(frames):
        line = f"Frame {i+1}"
        
        if include_timestamps:
            line += f" @ {frame.timestamp:.1f}s"
        
        if include_time_deltas and i > 0:
            delta = frame.timestamp - prev_ts
            line += f" (Δ{delta:.1f}s)"
        
        if include_position_labels and frame.position_label:
            line += f" [{frame.position_label}]"
        
        prompt += line + "\n"
        prev_ts = frame.timestamp
    
    # Add audio context if available
    if audio_context:
        prompt += "\n\nAUDIO CONTEXT:\n"
        
        if "transcription" in audio_context and audio_context["transcription"]:
            prompt += "Spoken Content:\n"
            for segment in audio_context["transcription"][:10]:  # Limit to avoid token overflow
                prompt += f"- [{segment['start']:.1f}s-{segment['end']:.1f}s]: \"{segment['text']}\"\n"
            prompt += "\n"
        
        if "mood" in audio_context:
            prompt += f"Audio Mood: {audio_context['mood']}\n"
        
        if "key_phrases" in audio_context:
            prompt += "Key Spoken Phrases:\n"
            for phrase in audio_context["key_phrases"]:
                prompt += f"- \"{phrase['text']}\" at {phrase['timestamp']:.1f}s\n"
            prompt += "\n"
    
    prompt += f"""

Extract the following information in JSON format:

{json.dumps(schema, indent=2)}

EXTRACTION GUIDELINES:

Brand Information:
- brand_name_text: The brand/company name as it appears in text or visually
- brand_text_contrast: Rate how prominently the brand name is displayed (low/medium/high)
- industry: Business category (e.g., automotive, food & beverage, technology, retail, finance)

Product Information:
- product_name: Specific product or service being advertised (not just the brand)

Promotional Offers:
- promo_present: Set to true only if there is a specific promotional offer
- promo_text: Extract ONLY the core offer itself (e.g., "50% off", "Buy one get one free", "1 cent to join & get 1 month free")
  DO NOT include full sentences or descriptions - just the essential offer
- promo_deadline: Any time limit mentioned (e.g., "ends today", "48 hours only", "limited time")
- price_value: Specific price if shown (e.g., "$9.99/mo", "$0.01 down", "Free trial")

Call to Action:
- cta_present: true if there is a button, text, or verbal instruction telling viewers what to do
- cta_type: The specific action (e.g., "Sign up button", "Order now", "Learn more", "Download app", "Visit website")

Visual Analysis:
- text_density: Assess overall amount of text on screen
  * low: Minimal text, mostly visuals
  * medium: Moderate text with balanced visuals
  * high: Text-heavy, lots of information displayed
- text_overlays: List all visible text that appears on screen

Content Rating:
- is_nsfw: Set to true ONLY if content contains explicit sexual content, graphic violence, or other not-safe-for-work material
  Most advertisements should be false

IMPORTANT FORMATTING RULES:
- Respond with ONLY valid JSON, no markdown code blocks or explanations
- Use null for fields where information is not available or not applicable
- Be specific and concise in your descriptions
- Do NOT use emojis in any field
- Keep responses professional and factual
- Extract exact text as it appears, maintaining proper spelling and capitalization

JSON Response:"""
    
    return prompt


def build_type_detection_prompt() -> str:
    """
    Build prompt for detecting ad type.
    
    Returns:
        Prompt string for ad type classification
    """
    return """Analyze this advertisement and classify it into exactly ONE category:

- product_demo: Shows product features, usage, or demonstration
- testimonial: Features customer reviews, expert opinions, or endorsements
- brand_awareness: Emotional storytelling focused on brand values, no specific product
- tutorial: Teaches how to do something, instructional content
- entertainment: Comedy, celebrity content, viral/shareable moments

Respond with ONLY the category name, nothing else. Do not use emojis or additional formatting."""