#src\utils\config.py
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)
def load_config(config_path: str, overrides: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file with optional overrides.
    
    Args:
        config_path: Path to YAML config file
        overrides: Dictionary of values to override
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if overrides:
        config = deep_merge(config, overrides)
    
    return config


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_device(preference: str = "auto") -> str:
    """
    Get the best available device.
    
    Args:
        preference: "auto", "cuda", or "cpu"
        
    Returns:
        Device string
    """
    import torch
    
    if preference == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return preference

