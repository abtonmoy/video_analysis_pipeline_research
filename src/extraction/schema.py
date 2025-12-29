# src\extraction\schema.py
"""
Schema definitions for ad content extraction.
"""

BASE_SCHEMA = {
    "brand": {
        "name": "string",
        "logo_visible": "boolean",
        "logo_timestamps": "list of floats (seconds when logo appears)"
    },
    "message": {
        "primary_message": "string (main message or value proposition)",
        "call_to_action": "string or null",
        "tagline": "string or null"
    },
    "creative_elements": {
        "dominant_colors": "list of color names",
        "text_overlays": "list of text shown on screen",
        "music_mood": "string or null (e.g., upbeat, dramatic, calm)"
    },
    "target_audience": {
        "age_group": "string (e.g., 18-25, 25-40, all ages)",
        "interests": "list of interests this ad appeals to"
    },
    "persuasion_techniques": "list of techniques used (e.g., social proof, scarcity, emotion)"
}

SCHEMA_EXTENSIONS = {
    "product_demo": {
        "product": {
            "name": "string",
            "category": "string",
            "features_demonstrated": "list of features shown",
            "price_shown": "string or null"
        },
        "demo_steps": "list of demonstration steps shown"
    },
    "testimonial": {
        "testimonial": {
            "speaker_name": "string or null",
            "speaker_role": "string (e.g., customer, expert, celebrity)",
            "key_quotes": "list of notable quotes",
            "credibility_markers": "list (e.g., credentials, affiliations)"
        }
    },
    "brand_awareness": {
        "emotional_appeal": {
            "primary_emotion": "string (e.g., joy, nostalgia, excitement)",
            "storytelling_elements": "list of narrative elements",
            "brand_values_conveyed": "list of values (e.g., innovation, trust)"
        }
    },
    "tutorial": {
        "tutorial": {
            "skill_taught": "string",
            "steps": "list of instructional steps",
            "tools_shown": "list of tools or products used"
        }
    },
    "entertainment": {
        "entertainment": {
            "humor_type": "string or null (e.g., slapstick, wordplay)",
            "celebrity_featured": "string or null",
            "viral_elements": "list of shareable moments"
        }
    }
}

FLEXIBLE_SCHEMA = {
    "brand": {
        "name": "string",
        "logo_visible": "boolean"
    },
    "ad_type": "string (product_demo | testimonial | brand_awareness | tutorial | entertainment)",
    "message": {
        "primary_message": "string",
        "call_to_action": "string or null"
    },
    "narrative": {
        "opening_hook": "string (how the ad grabs attention)",
        "middle_development": "string (main content)",
        "closing_resolution": "string (conclusion and CTA)"
    },
    "key_elements": "list of most important visual/audio elements",
    "persuasion_techniques": "list of techniques used",
    "target_audience": {
        "demographics": "string",
        "interests": "list"
    }
}


def get_schema(mode: str = "adaptive", ad_type: str = None) -> dict:
    """
    Get schema based on mode and ad type.
    
    Args:
        mode: "adaptive", "fixed", or "flexible"
        ad_type: Type of ad (only used in adaptive mode)
        
    Returns:
        Schema dictionary
    """
    if mode == "flexible":
        return FLEXIBLE_SCHEMA
    
    if mode == "fixed" or ad_type is None:
        return BASE_SCHEMA
    
    # Adaptive: combine base with type-specific
    schema = BASE_SCHEMA.copy()
    if ad_type in SCHEMA_EXTENSIONS:
        schema.update(SCHEMA_EXTENSIONS[ad_type])
    
    return schema


def get_valid_ad_types() -> list:
    """Get list of valid ad types."""
    return list(SCHEMA_EXTENSIONS.keys())