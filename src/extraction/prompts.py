#src\extraction\prompts.py
"""
Prompt building for LLM extraction.
Includes Topic and Sentiment classification from video ad dataset taxonomy.
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


def get_topic_reference() -> str:
    """
    Get the topic categories reference for the prompt.
    
    Returns:
        Formatted string of all topic categories
    """
    return """TOPIC CATEGORIES (select ONE primary topic by ID):
1. Restaurants, cafe, fast food
2. Chocolate, cookies, candy, ice cream
3. Chips, snacks, nuts, fruit, gum, cereal, yogurt, soups
4. Seasoning, condiments, ketchup
5. Pet food
6. Alcohol
7. Coffee, tea
8. Soda, juice, milk, energy drinks, water
9. Cars, automobiles (car sales, auto parts, car insurance, car repair, gas, motor oil, etc.)
10. Electronics (computers, laptops, tablets, cellphones, TVs, etc.)
11. Phone, TV and internet service providers
12. Financial services (banks, credit cards, investment firms, etc.)
13. Education (universities, colleges, kindergarten, online degrees, etc.)
14. Security and safety services (anti-theft, safety courses, etc.)
15. Software (internet radio, streaming, job search website, grammar correction, travel planning, etc.)
16. Other services (dating, tax, legal, loan, religious, printing, catering, etc.)
17. Beauty products and cosmetics (deodorants, toothpaste, makeup, hair products, laser hair removal, etc.)
18. Healthcare and medications (hospitals, health insurance, allergy, cold remedy, home tests, vitamins)
19. Clothing and accessories (jeans, shoes, eye glasses, handbags, watches, jewelry)
20. Baby products (baby food, sippy cups, diapers, etc.)
21. Games and toys (including video and mobile games)
22. Cleaning products (detergents, fabric softeners, soap, tissues, paper towels, etc.)
23. Home improvements and repairs (furniture, decoration, lawn care, plumbing, etc.)
24. Home appliances (coffee makers, dishwashers, cookware, vacuum cleaners, heaters, music players, etc.)
25. Vacation and travel (airlines, cruises, theme parks, hotels, travel agents, etc.)
26. Media and arts (TV shows, movies, musicals, books, audio books, etc.)
27. Sports equipment and activities
28. Shopping (department stores, drug stores, groceries, etc.)
29. Gambling (lotteries, casinos, etc.)
30. Environment, nature, pollution, wildlife
31. Animal rights, animal abuse
32. Human rights
33. Safety, safe driving, fire safety
34. Smoking, alcohol abuse
35. Domestic violence
36. Self esteem, bullying, cyber bullying
37. Political candidates (support or opposition)
38. Charities"""


def get_sentiment_reference() -> str:
    """
    Get the sentiment categories reference for the prompt.
    
    Returns:
        Formatted string of all sentiment categories
    """
    return """SENTIMENT CATEGORIES (select primary sentiment by ID, and up to 3 secondary sentiments):
1. Active (energetic, adventurous, vibrant, enthusiastic, playful)
2. Afraid (horrified, scared, fearful)
3. Alarmed (concerned, worried, anxious, overwhelmed)
4. Alert (attentive, curious)
5. Amazed (surprised, astonished, awed, fascinated, intrigued)
6. Amused (humored, laughing)
7. Angry (annoyed, irritated)
8. Calm (soothed, peaceful, comforted, fulfilled, cozy)
9. Cheerful (delighted, happy, joyful, carefree, optimistic)
10. Confident (assured, strong, healthy)
11. Conscious (aware, thoughtful, prepared)
12. Conscious (aware, thoughtful, prepared)
13. Disturbed (disgusted, shocked)
14. Eager (hungry, thirsty, passionate)
15. Educated (informed, enlightened, smart, savvy, intelligent)
16. Emotional (vulnerable, moved, nostalgic, reminiscent)
17. Empathetic (sympathetic, supportive, understanding, receptive)
18. Fashionable (trendy, elegant, beautiful, attractive, sexy)
19. Feminine (womanly, girlish)
20. Grateful (thankful)
21. Inspired (motivated, ambitious, empowered, hopeful, determined)
22. Jealous
23. Loving (loved, romantic)
24. Manly
25. Persuaded (impressed, enchanted, immersed)
26. Pessimistic (skeptical)
27. Proud (patriotic)
28. Sad (depressed, upset, betrayed, distant)
29. Thrifty (frugal)
30. Youthful (childlike)"""


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
    
    # Add Topic reference
    prompt += f"""

{get_topic_reference()}

{get_sentiment_reference()}

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

Topic Classification:
- topic_id: Select the SINGLE most appropriate topic category ID (1-38) from the list above
- topic_name: Provide the full name of the selected topic category
- topic_confidence: Rate your confidence in the classification (low/medium/high)
  * Consider: Does the ad clearly fit one category, or could it belong to multiple?

Sentiment Classification:
- primary_sentiment_id: Select the SINGLE most dominant sentiment ID (1-30) that the ad is designed to evoke in viewers
- primary_sentiment_name: Provide the full name of the primary sentiment
- secondary_sentiments: List up to 3 additional sentiment IDs that also apply (can be empty list)
- sentiment_intensity: Rate how strongly the sentiment is conveyed (low/medium/high)
  * low: Subtle emotional appeal
  * medium: Clear emotional messaging
  * high: Strong, unmistakable emotional impact

Engagement Metrics:
- is_funny: Rate 0.0 to 1.0 - likelihood the ad uses humor (0.0 = not at all, 1.0 = primarily comedic)
- is_exciting: Rate 0.0 to 1.0 - likelihood the ad is exciting/thrilling (0.0 = calm/mundane, 1.0 = high energy/thrilling)
- effectiveness_score: Rate 1-5 how effective you predict this ad to be
  * 1: Poor - confusing, off-putting, or likely to be ignored
  * 2: Below average - some issues with messaging or execution
  * 3: Average - competent but not memorable
  * 4: Good - clear message, engaging, likely to resonate
  * 5: Excellent - highly compelling, memorable, likely to drive action

IMPORTANT FORMATTING RULES:
- Respond with ONLY valid JSON, no markdown code blocks or explanations
- Use null for fields where information is not available or not applicable
- Be specific and concise in your descriptions
- Do NOT use emojis in any field
- Keep responses professional and factual
- Extract exact text as it appears, maintaining proper spelling and capitalization
- For topic_id and sentiment IDs, use INTEGER values (not strings)

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


def build_topic_only_prompt() -> str:
    """
    Build a prompt specifically for topic classification.
    
    Returns:
        Prompt string for topic classification only
    """
    return f"""Analyze this advertisement and classify it into exactly ONE topic category.

{get_topic_reference()}

Respond with ONLY the topic ID number (1-38), nothing else."""


def build_sentiment_only_prompt() -> str:
    """
    Build a prompt specifically for sentiment classification.
    
    Returns:
        Prompt string for sentiment classification only
    """
    return f"""Analyze this advertisement and identify the PRIMARY sentiment it is designed to evoke in viewers.

{get_sentiment_reference()}

Respond with ONLY the sentiment ID number (1-30), nothing else."""


def build_engagement_prompt() -> str:
    """
    Build a prompt for engagement metrics (funny, exciting, effective).
    
    Returns:
        Prompt string for engagement metrics
    """
    return """Analyze this advertisement and rate the following:

1. FUNNY: How humorous/comedic is this ad? (0.0 to 1.0)
   - 0.0 = No humor at all
   - 0.5 = Some humor elements
   - 1.0 = Primarily a comedy ad

2. EXCITING: How exciting/thrilling is this ad? (0.0 to 1.0)
   - 0.0 = Calm, mundane
   - 0.5 = Moderately engaging
   - 1.0 = High energy, thrilling

3. EFFECTIVE: How effective do you predict this ad to be? (1 to 5)
   - 1 = Poor
   - 2 = Below average
   - 3 = Average
   - 4 = Good
   - 5 = Excellent

Respond with ONLY three numbers separated by commas: funny_score,exciting_score,effectiveness_score
Example: 0.3,0.7,4"""