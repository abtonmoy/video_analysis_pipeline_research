# src\extraction\schema.py
"""
Schema definitions for ad content extraction.
Includes Topic and Sentiment taxonomies from video ad dataset.
"""

# Topic categories from the video ad dataset
# Keys are 1-indexed to match the dataset
TOPICS = {
    1: "Restaurants, cafe, fast food",
    2: "Chocolate, cookies, candy, ice cream",
    3: "Chips, snacks, nuts, fruit, gum, cereal, yogurt, soups",
    4: "Seasoning, condiments, ketchup",
    5: "Pet food",
    6: "Alcohol",
    7: "Coffee, tea",
    8: "Soda, juice, milk, energy drinks, water",
    9: "Cars, automobiles (car sales, auto parts, car insurance, car repair, gas, motor oil, etc.)",
    10: "Electronics (computers, laptops, tablets, cellphones, TVs, etc.)",
    11: "Phone, TV and internet service providers",
    12: "Financial services (banks, credit cards, investment firms, etc.)",
    13: "Education (universities, colleges, kindergarten, online degrees, etc.)",
    14: "Security and safety services (anti-theft, safety courses, etc.)",
    15: "Software (internet radio, streaming, job search website, grammar correction, travel planning, etc.)",
    16: "Other services (dating, tax, legal, loan, religious, printing, catering, etc.)",
    17: "Beauty products and cosmetics (deodorants, toothpaste, makeup, hair products, laser hair removal, etc.)",
    18: "Healthcare and medications (hospitals, health insurance, allergy, cold remedy, home tests, vitamins)",
    19: "Clothing and accessories (jeans, shoes, eye glasses, handbags, watches, jewelry)",
    20: "Baby products (baby food, sippy cups, diapers, etc.)",
    21: "Games and toys (including video and mobile games)",
    22: "Cleaning products (detergents, fabric softeners, soap, tissues, paper towels, etc.)",
    23: "Home improvements and repairs (furniture, decoration, lawn care, plumbing, etc.)",
    24: "Home appliances (coffee makers, dishwashers, cookware, vacuum cleaners, heaters, music players, etc.)",
    25: "Vacation and travel (airlines, cruises, theme parks, hotels, travel agents, etc.)",
    26: "Media and arts (TV shows, movies, musicals, books, audio books, etc.)",
    27: "Sports equipment and activities",
    28: "Shopping (department stores, drug stores, groceries, etc.)",
    29: "Gambling (lotteries, casinos, etc.)",
    30: "Environment, nature, pollution, wildlife",
    31: "Animal rights, animal abuse",
    32: "Human rights",
    33: "Safety, safe driving, fire safety",
    34: "Smoking, alcohol abuse",
    35: "Domestic violence",
    36: "Self esteem, bullying, cyber bullying",
    37: "Political candidates (support or opposition)",
    38: "Charities",
}

# Topic abbreviations for compact reference
TOPIC_ABBREVIATIONS = {
    1: "restaurant",
    2: "chocolate",
    3: "chips",
    4: "seasoning",
    5: "petfood",
    6: "alcohol",
    7: "coffee",
    8: "soda",
    9: "cars",
    10: "electronics",
    11: "phone_tv_internet_providers",
    12: "financial",
    13: "education",
    14: "security",
    15: "software",
    16: "other_service",
    17: "beauty",
    18: "healthcare",
    19: "clothing",
    20: "baby",
    21: "game",
    22: "cleaning",
    23: "home_improvement",
    24: "home_appliance",
    25: "travel",
    26: "media",
    27: "sports",
    28: "shopping",
    29: "gambling",
    30: "environment",
    31: "animal_right",
    32: "human_right",
    33: "safety",
    34: "smoking_alcohol_abuse",
    35: "domestic_violence",
    36: "self_esteem",
    37: "political",
    38: "charities",
}

# Sentiment categories from the video ad dataset
# Keys are 1-indexed to match the dataset
SENTIMENTS = {
    1: "Active (energetic, adventurous, vibrant, enthusiastic, playful)",
    2: "Afraid (horrified, scared, fearful)",
    3: "Alarmed (concerned, worried, anxious, overwhelmed)",
    4: "Alert (attentive, curious)",
    5: "Amazed (surprised, astonished, awed, fascinated, intrigued)",
    6: "Amused (humored, laughing)",
    7: "Angry (annoyed, irritated)",
    8: "Calm (soothed, peaceful, comforted, fulfilled, cozy)",
    9: "Cheerful (delighted, happy, joyful, carefree, optimistic)",
    10: "Confident (assured, strong, healthy)",
    11: "Conscious (aware, thoughtful, prepared)",
    12: "Creative (inventive, productive)",
    13: "Disturbed (disgusted, shocked)",
    14: "Eager (hungry, thirsty, passionate)",
    15: "Educated (informed, enlightened, smart, savvy, intelligent)",
    16: "Emotional (vulnerable, moved, nostalgic, reminiscent)",
    17: "Empathetic (sympathetic, supportive, understanding, receptive)",
    18: "Fashionable (trendy, elegant, beautiful, attractive, sexy)",
    19: "Feminine (womanly, girlish)",
    20: "Grateful (thankful)",
    21: "Inspired (motivated, ambitious, empowered, hopeful, determined)",
    22: "Jealous",
    23: "Loving (loved, romantic)",
    24: "Manly",
    25: "Persuaded (impressed, enchanted, immersed)",
    26: "Pessimistic (skeptical)",
    27: "Proud (patriotic)",
    28: "Sad (depressed, upset, betrayed, distant)",
    29: "Thrifty (frugal)",
    30: "Youthful (childlike)",
}

# Sentiment abbreviations for compact reference
SENTIMENT_ABBREVIATIONS = {
    1: "active",
    2: "afraid",
    3: "alarmed",
    4: "alert",
    5: "amazed",
    6: "amused",
    7: "angry",
    8: "calm",
    9: "cheerful",
    10: "confident",
    11: "conscious",
    12: "creative",
    13: "disturbed",
    14: "eager",
    15: "educated",
    16: "emotional",
    17: "empathetic",
    18: "fashionable",
    19: "feminine",
    20: "grateful",
    21: "inspired",
    22: "jealous",
    23: "loving",
    24: "manly",
    25: "persuaded",
    26: "pessimistic",
    27: "proud",
    28: "sad",
    29: "thrifty",
    30: "youthful",
}


def get_topic_list() -> list:
    """Get list of all topic names for prompt inclusion."""
    return [f"{k}. {v}" for k, v in TOPICS.items()]


def get_sentiment_list() -> list:
    """Get list of all sentiment names for prompt inclusion."""
    return [f"{k}. {v}" for k, v in SENTIMENTS.items()]


def get_topic_by_id(topic_id: int) -> str:
    """Get topic name by ID."""
    return TOPICS.get(topic_id, "Unknown")


def get_sentiment_by_id(sentiment_id: int) -> str:
    """Get sentiment name by ID."""
    return SENTIMENTS.get(sentiment_id, "Unknown")


def get_topic_abbreviation(topic_id: int) -> str:
    """Get topic abbreviation by ID."""
    return TOPIC_ABBREVIATIONS.get(topic_id, "unknown")


def get_sentiment_abbreviation(sentiment_id: int) -> str:
    """Get sentiment abbreviation by ID."""
    return SENTIMENT_ABBREVIATIONS.get(sentiment_id, "unknown")


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
    "topic": {
        "topic_id": "integer (1-38, the primary topic category ID from the predefined list)",
        "topic_name": "string (the topic category name)",
        "topic_confidence": "string (low/medium/high - confidence in topic classification)"
    },
    "sentiment": {
        "primary_sentiment_id": "integer (1-30, the primary sentiment ID from the predefined list)",
        "primary_sentiment_name": "string (the primary sentiment name)",
        "secondary_sentiments": "list of integers (additional sentiment IDs that apply, max 3)",
        "sentiment_intensity": "string (low/medium/high - how strongly the sentiment is conveyed)"
    },
    "engagement_metrics": {
        "is_funny": "float (0.0-1.0, likelihood the ad is humorous/comedic)",
        "is_exciting": "float (0.0-1.0, likelihood the ad is exciting/thrilling)",
        "effectiveness_score": "integer (1-5, predicted effectiveness of the ad)"
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
    "topic": {
        "topic_id": "integer (1-38)",
        "topic_name": "string"
    },
    "sentiment": {
        "primary_sentiment_id": "integer (1-30)",
        "primary_sentiment_name": "string",
        "secondary_sentiments": "list of integers (max 3)"
    },
    "engagement_metrics": {
        "is_funny": "float (0.0-1.0)",
        "is_exciting": "float (0.0-1.0)",
        "effectiveness_score": "integer (1-5)"
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


def get_all_topics() -> dict:
    """Get all topics as a dictionary."""
    return TOPICS.copy()


def get_all_sentiments() -> dict:
    """Get all sentiments as a dictionary."""
    return SENTIMENTS.copy()