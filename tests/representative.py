"""
Test for selection/representation.py
"""

import numpy as np
from src.selection.clustering import FrameCandidate
from src.selection.representative import ImportanceScorer, FrameSelector


def test_position_scoring():
    """Test scoring by position in video."""
    print("="*80)
    print("TEST 1: Position-based Importance Scoring")
    print("="*80)
    
    scorer = ImportanceScorer()
    video_duration = 10.0
    
    test_cases = [
        (0.5, "Opening (5%)"),
        (1.0, "Opening (10%)"),
        (5.0, "Middle (50%)"),
        (9.0, "Closing (90%)"),
        (9.5, "Closing (95%)"),
    ]
    
    print(f"\nVideo duration: {video_duration}s")
    print(f"\nPosition scores:")
    
    for timestamp, label in test_cases:
        score = scorer.score_by_position(timestamp, video_duration)
        position_pct = (timestamp / video_duration) * 100
        print(f"  {timestamp:.1f}s ({position_pct:5.1f}%) - {label:20s}: {score:.2f}x")
    
    # Verify
    assert scorer.score_by_position(0.5, 10.0) == 1.5, "Opening should be 1.5x"
    assert scorer.score_by_position(5.0, 10.0) == 1.0, "Middle should be 1.0x"
    assert scorer.score_by_position(9.5, 10.0) == 1.3, "Closing should be 1.3x"
    
    print(f"\n✓ Position scoring works!")


def test_audio_event_scoring():
    """Test scoring by audio events."""
    print("\n" + "="*80)
    print("TEST 2: Audio Event Importance Scoring")
    print("="*80)
    
    scorer = ImportanceScorer()
    
    # Define audio events
    audio_events = {
        "energy_peaks": [1.5, 3.2, 7.8],
        "silence_segments": [(2.0, 2.5), (5.0, 5.3)]
    }
    
    test_cases = [
        (1.5, "At energy peak"),
        (1.7, "Near energy peak (0.2s away)"),
        (2.6, "After silence (0.1s after)"),
        (4.0, "No audio events nearby"),
        (7.9, "Near energy peak (0.1s away)"),
    ]
    
    print(f"\nAudio Events:")
    print(f"  Energy peaks: {audio_events['energy_peaks']}")
    print(f"  Silence: {audio_events['silence_segments']}")
    print(f"\nAudio event scores:")
    
    for timestamp, label in test_cases:
        score = scorer.score_by_audio_events(timestamp, audio_events)
        print(f"  {timestamp:.1f}s - {label:30s}: {score:.2f}x")
    
    # Verify
    assert scorer.score_by_audio_events(1.5, audio_events) > 1.0, "At peak should boost"
    assert scorer.score_by_audio_events(2.6, audio_events) > 1.0, "After silence should boost"
    assert scorer.score_by_audio_events(4.0, audio_events) == 1.0, "No events should be 1.0x"
    
    print(f"\n✓ Audio event scoring works!")


def test_scene_position_scoring():
    """Test scoring by position within scene."""
    print("\n" + "="*80)
    print("TEST 3: Scene Position Importance Scoring")
    print("="*80)
    
    scorer = ImportanceScorer()
    
    scene_start = 2.0
    scene_end = 8.0
    scene_duration = scene_end - scene_start
    
    test_cases = [
        (2.1, "Scene start (1.7%)"),
        (2.5, "Scene start (8.3%)"),
        (5.0, "Scene middle (50%)"),
        (7.5, "Scene end (91.7%)"),
        (7.9, "Scene end (98.3%)"),
    ]
    
    print(f"\nScene: {scene_start}s - {scene_end}s (duration: {scene_duration}s)")
    print(f"\nScene position scores:")
    
    for timestamp, label in test_cases:
        score = scorer.score_by_scene_position(timestamp, scene_start, scene_end)
        position_in_scene = ((timestamp - scene_start) / scene_duration) * 100
        print(f"  {timestamp:.1f}s ({position_in_scene:5.1f}% into scene) - {label:25s}: {score:.2f}x")
    
    # Verify
    assert scorer.score_by_scene_position(2.1, 2.0, 8.0) == 1.4, "Scene start should be 1.4x"
    assert scorer.score_by_scene_position(5.0, 2.0, 8.0) == 1.0, "Scene middle should be 1.0x"
    assert scorer.score_by_scene_position(7.9, 2.0, 8.0) == 1.2, "Scene end should be 1.2x"
    
    print(f"\n✓ Scene position scoring works!")


def test_combined_importance():
    """Test combined importance scoring."""
    print("\n" + "="*80)
    print("TEST 4: Combined Importance Scoring")
    print("="*80)
    
    scorer = ImportanceScorer()
    
    # Setup
    video_duration = 10.0
    scene_boundaries = [(0.0, 3.0), (3.0, 7.0), (7.0, 10.0)]
    audio_events = {
        "energy_peaks": [1.0, 5.5, 8.5],
        "silence_segments": [(2.5, 3.0)]
    }
    
    # Create test frames
    test_frames = [
        (0.2, 0, "Opening + scene start"),
        (1.0, 0, "Opening + at energy peak"),
        (3.1, 1, "After silence + scene start"),
        (5.0, 1, "Middle + neutral"),
        (9.5, 2, "Closing + scene end"),
    ]
    
    print(f"\nVideo: {video_duration}s")
    print(f"Scenes: {len(scene_boundaries)}")
    print(f"\nCombined importance scores:")
    
    for timestamp, scene_id, label in test_frames:
        # Create frame candidate
        frame = FrameCandidate(
            timestamp=timestamp,
            frame=np.zeros((480, 640, 3)),
            scene_id=scene_id
        )
        
        score = scorer.compute_importance(
            frame,
            video_duration,
            scene_boundaries,
            audio_events
        )
        
        print(f"  {timestamp:.1f}s (Scene {scene_id}) - {label:30s}: {score:.2f}x")
    
    # Verify opening frame has highest boost
    frame_opening = FrameCandidate(0.2, np.zeros((480, 640, 3)), scene_id=0)
    frame_middle = FrameCandidate(5.0, np.zeros((480, 640, 3)), scene_id=1)
    
    score_opening = scorer.compute_importance(frame_opening, video_duration, scene_boundaries, audio_events)
    score_middle = scorer.compute_importance(frame_middle, video_duration, scene_boundaries, audio_events)
    
    assert score_opening > score_middle, "Opening frame should score higher than middle"
    
    print(f"\n✓ Combined importance scoring works!")


def test_frame_selector_integration():
    """Test FrameSelector with full pipeline."""
    print("\n" + "="*80)
    print("TEST 5: FrameSelector Integration")
    print("="*80)
    
    # Create synthetic data
    frames = [
        (0.5, np.zeros((480, 640, 3))),
        (1.0, np.zeros((480, 640, 3))),
        (1.5, np.zeros((480, 640, 3))),
        (3.5, np.zeros((480, 640, 3))),
        (4.0, np.zeros((480, 640, 3))),
        (4.5, np.zeros((480, 640, 3))),
        (7.0, np.zeros((480, 640, 3))),
        (8.0, np.zeros((480, 640, 3))),
    ]
    
    # Create synthetic embeddings (different clusters)
    embeddings = np.vstack([
        np.random.randn(3, 512) + 1.0,   # Scene 0 cluster
        np.random.randn(3, 512) - 1.0,   # Scene 1 cluster
        np.random.randn(2, 512) + 0.0,   # Scene 2 cluster
    ])
    
    scene_boundaries = [
        (0.0, 2.5),
        (2.5, 6.0),
        (6.0, 10.0),
    ]
    
    video_duration = 10.0
    
    audio_events = {
        "energy_peaks": [1.0, 4.0, 8.0],
        "silence_segments": []
    }
    
    # Create selector
    selector = FrameSelector(
        max_frames_per_scene=2,
        min_temporal_gap_s=0.5,
        clustering_method="kmeans",
        use_importance_scoring=True
    )
    
    print(f"\nInput:")
    print(f"  Frames: {len(frames)}")
    print(f"  Scenes: {len(scene_boundaries)}")
    print(f"  Config: max_frames_per_scene=2, min_temporal_gap_s=0.5s")
    
    # Run selection
    selected = selector.select(
        frames,
        embeddings,
        scene_boundaries,
        video_duration,
        audio_events
    )
    
    print(f"\nSelected {len(selected)} representatives:")
    for cand in selected:
        print(f"  {cand.timestamp:.1f}s (Scene {cand.scene_id}, importance: {cand.importance_score:.2f})")
    
    # Verify
    assert len(selected) > 0, "Should select at least some frames"
    assert len(selected) <= len(frames), "Should not create new frames"
    
    # Check temporal gaps
    for i in range(1, len(selected)):
        gap = selected[i].timestamp - selected[i-1].timestamp
        assert gap >= 0.5, f"Gap {gap:.1f}s violates minimum 0.5s"
    
    # Check scene distribution
    scene_counts = {}
    for cand in selected:
        scene_counts[cand.scene_id] = scene_counts.get(cand.scene_id, 0) + 1
    
    print(f"\nFrames per scene:")
    for scene_id, count in sorted(scene_counts.items()):
        print(f"  Scene {scene_id}: {count} frames")
    
    for scene_id, count in scene_counts.items():
        assert count <= 2, f"Scene {scene_id} has {count} frames, max is 2"
    
    print(f"\n✓ FrameSelector integration works!")


def test_config_based_creation():
    """Test creating selector from config."""
    print("\n" + "="*80)
    print("TEST 6: Config-based Selector Creation")
    print("="*80)
    
    from src.selection.representative import create_selector
    
    config = {
        "selection": {
            "method": "clustering",
            "max_frames_per_scene": 5,
            "min_temporal_gap_s": 1.0
        }
    }
    
    print(f"\nConfig:")
    print(f"  method: {config['selection']['method']}")
    print(f"  max_frames_per_scene: {config['selection']['max_frames_per_scene']}")
    print(f"  min_temporal_gap_s: {config['selection']['min_temporal_gap_s']}")
    
    selector = create_selector(config)
    
    print(f"\nCreated selector:")
    print(f"  max_frames_per_scene: {selector.clusterer.max_frames_per_scene}")
    print(f"  min_temporal_gap_s: {selector.clusterer.min_temporal_gap_s}")
    print(f"  has scorer: {selector.scorer is not None}")
    
    assert selector.clusterer.max_frames_per_scene == 5
    assert selector.clusterer.min_temporal_gap_s == 1.0
    assert selector.scorer is not None
    
    print(f"\n✓ Config-based creation works!")


def main():
    print("Representation Module Tests\n")
    
    # Test 1: Position scoring
    test_position_scoring()
    
    # Test 2: Audio event scoring
    test_audio_event_scoring()
    
    # Test 3: Scene position scoring
    test_scene_position_scoring()
    
    # Test 4: Combined importance
    test_combined_importance()
    
    # Test 5: Full integration
    test_frame_selector_integration()
    
    # Test 6: Config creation
    test_config_based_creation()
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED! ✓")
    print("="*80)


if __name__ == "__main__":
    main()