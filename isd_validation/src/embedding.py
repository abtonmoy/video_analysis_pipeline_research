"""
CLIP embedding generation for videos.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str,
    interval_ms: float = 100.0,
    max_frames: int = 300
) -> List[Tuple[float, np.ndarray]]:
    """
    Extract frames at regular intervals from video.
    
    Args:
        video_path: Path to video file
        interval_ms: Interval between frames in milliseconds
        max_frames: Maximum number of frames to extract
    
    Returns:
        List of (timestamp, frame) tuples, and duration
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    # Calculate frame interval
    interval_frames = int(fps * interval_ms / 1000) if fps > 0 else 3
    interval_frames = max(1, interval_frames)
    
    frame_idx = 0
    while len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = frame_idx / fps if fps > 0 else 0
        frames.append((timestamp, frame))
        frame_idx += interval_frames
    
    cap.release()
    return frames, duration


def load_clip_model(model_name: str = "ViT-B-32", device: Optional[str] = None):
    """
    Load CLIP model using open_clip.
    
    Args:
        model_name: CLIP model variant
        device: Device to use (auto-detected if None)
    
    Returns:
        model, preprocess function, device
    """
    try:
        import open_clip
    except ImportError:
        raise ImportError("open_clip not installed. Run: uv pip install open-clip-torch")
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained="openai", device=device
    )
    
    return model, preprocess, device


def compute_clip_embeddings(
    frames: List[Tuple[float, np.ndarray]],
    model,
    preprocess,
    device: str,
    batch_size: int = 32
) -> np.ndarray:
    """
    Compute CLIP embeddings for frames.
    
    Args:
        frames: List of (timestamp, frame) tuples
        model: CLIP model
        preprocess: Preprocessing function
        device: Device string
        batch_size: Batch size for inference
    
    Returns:
        Normalized embeddings array (n_frames, 512)
    """
    embeddings = []
    
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        images = []
        
        for _, frame in batch:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            processed = preprocess(pil_image).unsqueeze(0)
            images.append(processed)
        
        if images:
            image_batch = torch.cat(images).to(device)
            
            with torch.no_grad():
                image_features = model.encode_image(image_batch)
                # Normalize
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embeddings.append(image_features.cpu().numpy())
    
    return np.vstack(embeddings) if embeddings else np.array([])


def process_video(
    video_path: str,
    model,
    preprocess,
    device: str,
    interval_ms: float = 100.0,
    save_frames: bool = False,
    output_dir: Optional[Path] = None
) -> dict:
    """
    Process a single video: extract frames and compute embeddings.
    
    Args:
        video_path: Path to video
        model: CLIP model
        preprocess: Preprocess function
        device: Device
        interval_ms: Frame extraction interval
        save_frames: Whether to save frame images
        output_dir: Directory to save frames
    
    Returns:
        Dictionary with results
    """
    from .isd import compute_isd, compute_semantic_velocity
    
    # Extract frames
    frames, duration = extract_frames(video_path, interval_ms=interval_ms)
    
    if len(frames) < 3:
        raise ValueError(f"Too few frames extracted: {len(frames)}")
    
    # Compute embeddings
    embeddings = compute_clip_embeddings(frames, model, preprocess, device)
    
    if len(embeddings) < 3:
        raise ValueError(f"Too few embeddings: {len(embeddings)}")
    
    # Compute metrics
    isd = compute_isd(embeddings)
    sem_velocity = compute_semantic_velocity(embeddings)
    
    result = {
        "video_name": Path(video_path).name,
        "duration": duration,
        "num_frames": len(frames),
        "isd": isd,
        "semantic_velocity": sem_velocity,
        "embeddings": embeddings,
    }
    
    # Save frames if requested
    if save_frames and output_dir:
        video_name = Path(video_path).stem
        frame_dir = output_dir / "frames" / video_name
        frame_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (ts, frame) in enumerate(frames):
            frame_path = frame_dir / f"frame_{i:04d}_{ts:.2f}s.jpg"
            cv2.imwrite(str(frame_path), frame)
    
    return result
