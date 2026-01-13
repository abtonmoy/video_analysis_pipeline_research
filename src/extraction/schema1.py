# src\extraction\schema.py
"""
Schema definitions for ad content extraction.
"""

BASE_SCHEMA = {
    "brand": {
        "brand_name_text": "string (the brand/company name as it appears in text)",
        "logo_visible": "boolean",
        "logo_timestamps": "list of floats (seconds when logo appears)",
        "brand_text_contrast": "string (low/medium/high - how much the brand name stands out visually)"
    },
    "product": {
        "product_name": "string (specific product or service being advertised)",
        "industry": "string (business category, e.g., automotive, food & beverage, technology)"
    },
    "promotion": {
        "promo_present": "boolean (true if promotional offer exists)",
        "promo_text": "string or null (ONLY the core offer itself, e.g., '50% off', 'Buy one get one free', NOT full description)",
        "promo_deadline": "string or null (time limit, e.g., 'ends today', '48 hours', 'limited time')",
        "price_value": "string or null (specific price mentioned, e.g., '$9.99/mo', '$0.01 down')"
    },
    "call_to_action": {
        "cta_present": "boolean (true if call-to-action exists)",
        "cta_type": "string or null (action requested, e.g., 'Sign up button', 'Order now', 'Learn more')"
    },
    "message": {
        "primary_message": "string (main message or value proposition)",
        "tagline": "string or null"
    },
    "visual_elements": {
        "text_density": "string (low/medium/high - amount of text on screen)",
        "dominant_colors": "list of color names",
        "text_overlays": "list of text shown on screen"
    },
    "content_rating": {
    "is_nsfw": "boolean (true if contains: explicit/suggestive sexual content, revealing attire used for sex appeal, graphic violence, drug content, or anything inappropriate for workplace viewing)"
    },
    "target_audience": {
        "age_group": "string (e.g., 18-25, 25-40, all ages)",
        "interests": "list of interests this ad appeals to"
    },
    "persuasion_techniques": "list of techniques used (e.g., social proof, scarcity, emotion)"
}

SCHEMA_EXTENSIONS = {
    "product_demo": {
        "demo_details": {
            "features_demonstrated": "list of features shown",
            "demo_steps": "list of demonstration steps shown"
        }
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
        "brand_name_text": "string",
        "logo_visible": "boolean",
        "brand_text_contrast": "string (low/medium/high)"
    },
    "product": {
        "product_name": "string",
        "industry": "string"
    },
    "ad_type": "string (product_demo | testimonial | brand_awareness | tutorial | entertainment)",
    "promotion": {
        "promo_present": "boolean",
        "promo_text": "string or null (core offer only)",
        "promo_deadline": "string or null",
        "price_value": "string or null"
    },
    "call_to_action": {
        "cta_present": "boolean",
        "cta_type": "string or null"
    },
    "message": {
        "primary_message": "string"
    },
    "narrative": {
        "opening_hook": "string (how the ad grabs attention)",
        "middle_development": "string (main content)",
        "closing_resolution": "string (conclusion and CTA)"
    },
    "visual_elements": {
        "text_density": "string (low/medium/high)",
        "key_elements": "list of most important visual elements"
    },
    "content_rating": {
        "is_nsfw": "boolean"
    },
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