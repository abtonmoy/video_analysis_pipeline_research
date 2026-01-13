"""
Test for selection/clustering.py
"""

import numpy as np
from src.selection.clustering import FrameCandidate, TemporalClusterer


def test_scene_assignment():
    """Test assigning frames to scenes."""
    print("="*80)
    print("TEST 1: Scene Assignment")
    print("="*80)
    
    # Create dummy frames
    frames = [
        (0.5, np.zeros((480, 640, 3))),
        (1.5, np.zeros((480, 640, 3))),
        (3.0, np.zeros((480, 640, 3))),
        (5.5, np.zeros((480, 640, 3))),
        (7.0, np.zeros((480, 640, 3))),
        (9.5, np.zeros((480, 640, 3))),
    ]
    
    # Define scene boundaries
    scene_boundaries = [
        (0.0, 2.5),   # Scene 0
        (2.5, 6.0),   # Scene 1
        (6.0, 10.0),  # Scene 2
    ]
    
    # Test scene assignment
    clusterer = TemporalClusterer()
    candidates = clusterer.assign_scenes(frames, scene_boundaries)
    
    print(f"\nScene Boundaries:")
    for i, (start, end) in enumerate(scene_boundaries):
        print(f"  Scene {i}: {start:.1f}s - {end:.1f}s")
    
    print(f"\nFrame → Scene Assignment:")
    for cand in candidates:
        print(f"  Frame at {cand.timestamp:.1f}s → Scene {cand.scene_id}")
    
    # Verify
    expected_scenes = [0, 0, 1, 1, 2, 2]
    actual_scenes = [c.scene_id for c in candidates]
    
    assert actual_scenes == expected_scenes, f"Expected {expected_scenes}, got {actual_scenes}"
    print(f"\n✓ Scene assignment correct!")
    
    return candidates


def test_uniform_selection():
    """Test uniform frame selection."""
    print("\n" + "="*80)
    print("TEST 2: Uniform Selection (No Embeddings)")
    print("="*80)
    
    # Create many frames in one scene
    frames = [(i * 0.1, np.zeros((480, 640, 3))) for i in range(20)]
    scene_boundaries = [(0.0, 2.0)]
    
    clusterer = TemporalClusterer(max_frames_per_scene=3)
    candidates = clusterer.assign_scenes(frames, scene_boundaries)
    
    print(f"\nInput: {len(candidates)} frames")
    print(f"Config: max_frames_per_scene = 3")
    
    # Select without embeddings (uses uniform selection)
    selected = clusterer.cluster_and_select(candidates, embeddings=None)
    
    print(f"\nSelected {len(selected)} frames:")
    for cand in selected:
        print(f"  {cand.timestamp:.1f}s (is_representative: {cand.is_representative})")
    
    assert len(selected) == 3, f"Expected 3 frames, got {len(selected)}"
    assert all(c.is_representative for c in selected), "All should be marked as representatives"
    print(f"\n✓ Uniform selection works!")
    
    return selected


def test_kmeans_selection():
    """Test K-means clustering selection."""
    print("\n" + "="*80)
    print("TEST 3: K-means Selection (With Embeddings)")
    print("="*80)
    
    # Create frames with synthetic embeddings
    # Simulate 3 clusters of similar content
    frames = []
    embeddings_list = []
    
    # Cluster 0: Frames 0-5 (person talking)
    for i in range(6):
        frames.append((i * 0.2, np.zeros((480, 640, 3))))
        embeddings_list.append(np.random.randn(512) + np.array([1.0] * 512))  # Cluster around [1,1,1...]
    
    # Cluster 1: Frames 6-10 (product shot)
    for i in range(6, 11):
        frames.append((i * 0.2, np.zeros((480, 640, 3))))
        embeddings_list.append(np.random.randn(512) + np.array([-1.0] * 512))  # Cluster around [-1,-1,-1...]
    
    # Cluster 2: Frames 11-15 (text overlay)
    for i in range(11, 16):
        frames.append((i * 0.2, np.zeros((480, 640, 3))))
        embeddings_list.append(np.random.randn(512) + np.array([0.0] * 512))  # Cluster around [0,0,0...]
    
    embeddings = np.array(embeddings_list)
    
    scene_boundaries = [(0.0, 3.0)]
    
    clusterer = TemporalClusterer(
        max_frames_per_scene=3,
        clustering_method="kmeans"
    )
    candidates = clusterer.assign_scenes(frames, scene_boundaries)
    
    print(f"\nInput: {len(candidates)} frames with embeddings")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Config: max_frames_per_scene = 3, method = kmeans")
    
    # Select with K-means
    selected = clusterer.cluster_and_select(candidates, embeddings)
    
    print(f"\nSelected {len(selected)} representatives:")
    for cand in selected:
        print(f"  Frame {cand.timestamp:.1f}s (cluster_id: {cand.cluster_id})")
    
    assert len(selected) == 3, f"Expected 3 frames, got {len(selected)}"
    
    # Check that selected frames come from different clusters
    cluster_ids = [c.cluster_id for c in selected]
    assert len(set(cluster_ids)) == 3, f"Expected 3 different clusters, got {set(cluster_ids)}"
    
    print(f"\n✓ K-means clustering works!")
    
    return selected


def test_temporal_gap_enforcement():
    """Test minimum temporal gap enforcement."""
    print("\n" + "="*80)
    print("TEST 4: Temporal Gap Enforcement")
    print("="*80)
    
    # Create frames very close together
    frames = [
        (0.0, np.zeros((480, 640, 3))),
        (0.2, np.zeros((480, 640, 3))),  # Too close!
        (0.4, np.zeros((480, 640, 3))),  # Too close!
        (1.0, np.zeros((480, 640, 3))),  # OK gap
        (1.3, np.zeros((480, 640, 3))),  # Too close!
        (2.5, np.zeros((480, 640, 3))),  # OK gap
    ]
    
    scene_boundaries = [(0.0, 3.0)]
    
    clusterer = TemporalClusterer(
        max_frames_per_scene=10,  # Keep all
        min_temporal_gap_s=0.5    # Minimum 0.5s gap
    )
    candidates = clusterer.assign_scenes(frames, scene_boundaries)
    
    print(f"\nInput frames:")
    for cand in candidates:
        print(f"  {cand.timestamp:.1f}s")
    
    print(f"\nConfig: min_temporal_gap_s = 0.5s")
    
    selected = clusterer.cluster_and_select(candidates, embeddings=None)
    
    print(f"\nAfter temporal gap enforcement:")
    for cand in selected:
        print(f"  {cand.timestamp:.1f}s")
    
    # Verify gaps
    for i in range(1, len(selected)):
        gap = selected[i].timestamp - selected[i-1].timestamp
        assert gap >= 0.5, f"Gap {gap:.1f}s is less than minimum 0.5s"
    
    print(f"\n✓ All gaps are at least 0.5s!")
    
    return selected


def test_multi_scene_clustering():
    """Test clustering across multiple scenes."""
    print("\n" + "="*80)
    print("TEST 5: Multi-Scene Clustering")
    print("="*80)
    
    # Create frames across 3 scenes
    frames = []
    
    # Scene 0: 5 frames
    for i in range(5):
        frames.append((0.5 + i * 0.3, np.zeros((480, 640, 3))))
    
    # Scene 1: 8 frames (should be clustered)
    for i in range(8):
        frames.append((3.0 + i * 0.2, np.zeros((480, 640, 3))))
    
    # Scene 2: 2 frames (should keep all)
    for i in range(2):
        frames.append((7.0 + i * 0.5, np.zeros((480, 640, 3))))
    
    scene_boundaries = [
        (0.0, 2.5),   # Scene 0
        (2.5, 6.0),   # Scene 1
        (6.0, 10.0),  # Scene 2
    ]
    
    clusterer = TemporalClusterer(max_frames_per_scene=3)
    candidates = clusterer.assign_scenes(frames, scene_boundaries)
    
    print(f"\nInput:")
    for i in range(3):
        scene_frames = [c for c in candidates if c.scene_id == i]
        print(f"  Scene {i}: {len(scene_frames)} frames")
    
    print(f"\nConfig: max_frames_per_scene = 3")
    
    selected = clusterer.cluster_and_select(candidates, embeddings=None)
    
    print(f"\nAfter clustering:")
    for i in range(3):
        scene_frames = [c for c in selected if c.scene_id == i]
        print(f"  Scene {i}: {len(scene_frames)} frames")
        for cand in scene_frames:
            print(f"    {cand.timestamp:.1f}s")
    
    # Verify
    scene_counts = {}
    for cand in selected:
        scene_counts[cand.scene_id] = scene_counts.get(cand.scene_id, 0) + 1
    
    assert scene_counts[0] <= 3, "Scene 0 should have max 3 frames"
    assert scene_counts[1] <= 3, "Scene 1 should have max 3 frames"
    assert scene_counts[2] == 2, "Scene 2 should keep both frames (<=3)"
    
    print(f"\n✓ Multi-scene clustering works!")
    
    return selected


def main():
    print("Clustering Module Tests\n")
    
    # Test 1: Scene assignment
    test_scene_assignment()
    
    # Test 2: Uniform selection
    test_uniform_selection()
    
    # Test 3: K-means selection
    test_kmeans_selection()
    
    # Test 4: Temporal gap
    test_temporal_gap_enforcement()
    
    # Test 5: Multi-scene
    test_multi_scene_clustering()
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED! ✓")
    print("="*80)


if __name__ == "__main__":
    main()