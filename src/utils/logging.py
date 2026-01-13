# src\utils\logging.py
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
import cv2
import numpy as np
from PIL import Image


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None
) -> None:
    """Configure logging for the pipeline."""
    
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )