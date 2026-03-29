#!/usr/bin/env python3
"""
Generate CLIP embeddings for videos to support ISD analysis.

Usage:
    uv run python generate_embeddings.py --video_dir /home/wabashcs/abt/use_data --output_dir results/embeddings
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def extract_frames(video_path: str, interval_ms: float = 100.0, max_frames: int = 300) -> List[Tuple[float, np.ndarray]]:
    """Extract frames at regular intervals."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
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


def load_clip_model(device: str = "auto"):
    """Load CLIP model."""
    try:
        import open_clip
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=device
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        return model, preprocess, tokenizer, device
    except ImportError:
        logger.error("open_clip not installed. Install with: uv pip install open-clip-torch")
        raise


def compute_embeddings(frames: List[Tuple[float, np.ndarray]], model, preprocess, device) -> np.ndarray:
    """Compute CLIP embeddings for frames."""
    from PIL import Image
    
    embeddings = []
    batch_size = 32
    
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
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embeddings.append(image_features.cpu().numpy())
    
    return np.vstack(embeddings) if embeddings else np.array([])


def compute_isd(embeddings: np.ndarray, tau: float = 0.90) -> int:
    """Compute Intrinsic Semantic Dimensionality via SVD."""
    if embeddings.shape[0] < 3:
        return max(1, embeddings.shape[0] // 2)
    
    centered = embeddings - embeddings.mean(axis=0)
    try:
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        var_explained = (s ** 2) / (np.sum(s ** 2) + 1e-9)
        cum_var = np.cumsum(var_explained)
        isd = int(np.argmax(cum_var >= tau)) + 1
        return isd
    except np.linalg.LinAlgError:
        return max(1, embeddings.shape[0] // 5)


def main():
    parser = argparse.ArgumentParser(description="Generate CLIP embeddings for videos")
    parser.add_argument("--video_dir", default="/home/wabashcs/abt/use_data", help="Directory with videos")
    parser.add_argument("--output_dir", default="results/embeddings", help="Output directory")
    parser.add_argument("--videos_csv", default="videos.csv", help="CSV with video names")
    parser.add_argument("--max_videos", type=int, default=None, help="Max videos to process")
    parser.add_argument("--interval_ms", type=float, default=100.0, help="Frame extraction interval")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load video list
    with open(args.videos_csv, 'r') as f:
        video_names = [line.strip() for line in f if line.strip()]
    
    if args.max_videos:
        video_names = video_names[:args.max_videos]
    
    logger.info(f"Processing {len(video_names)} videos")
    
    # Load CLIP model
    logger.info("Loading CLIP model...")
    model, preprocess, tokenizer, device = load_clip_model()
    
    # Process each video
    results = []
    for i, vname in enumerate(video_names):
        vpath = Path(args.video_dir) / vname
        if not vpath.exists():
            logger.warning(f"Video not found: {vpath}")
            continue
        
        logger.info(f"[{i+1}/{len(video_names)}] Processing {vname}")
        
        try:
            # Extract frames
            frames, duration = extract_frames(str(vpath), interval_ms=args.interval_ms)
            
            if len(frames) < 3:
                logger.warning(f"  Too few frames: {len(frames)}")
                continue
            
            # Compute embeddings
            embeddings = compute_embeddings(frames, model, preprocess, device)
            
            if len(embeddings) < 3:
                logger.warning(f"  Too few embeddings: {len(embeddings)}")
                continue
            
            # Compute ISD
            isd = compute_isd(embeddings)
            
            # Compute semantic velocity
            diffs = np.linalg.norm(np.diff(embeddings, axis=0), axis=1)
            sem_velocity = float(np.mean(diffs))
            
            # Save embeddings
            emb_path = output_dir / f"{vname}.npy"
            np.save(emb_path, embeddings)
            
            results.append({
                "video_name": vname,
                "duration": duration,
                "num_frames": len(frames),
                "isd": isd,
                "semantic_velocity": sem_velocity,
                "embedding_path": str(emb_path),
            })
            
            logger.info(f"  ISD={isd}, frames={len(frames)}, velocity={sem_velocity:.3f}")
            
        except Exception as e:
            logger.error(f"  Failed: {e}")
    
    # Save results
    with open(output_dir / "embedding_metadata.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nProcessed {len(results)} videos successfully")
    logger.info(f"Results saved to: {output_dir}")
    
    # Summary stats
    if results:
        isds = [r["isd"] for r in results]
        logger.info(f"\nISD Statistics:")
        logger.info(f"  Mean: {np.mean(isds):.1f}")
        logger.info(f"  Std: {np.std(isds):.1f}")
        logger.info(f"  Range: [{min(isds)}, {max(isds)}]")


if __name__ == "__main__":
    main()
