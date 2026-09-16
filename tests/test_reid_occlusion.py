"""
tests/test_reid_occlusion.py
Simulates a 5-second complete visual occlusion and verifies that
the Re-ID gallery cosine metric recovers the original entity identity.
Compatible with Python 3.12.5 on Windows.
"""

from __future__ import annotations

import time
import numpy as np
from src.perception.reid_tracker import TrackAppearanceGallery


def test_gallery_occlusion_persistence() -> None:
    print("=" * 65)
    print(" Testing Re-ID Gallery Persistence Across 5-Second Occlusion")
    print("=" * 65)

    gallery = TrackAppearanceGallery(alpha=0.9, max_lost_age_sec=8.0)
    current_time = time.time()

    # Generate a deterministic synthetic appearance feature vector (Entity #42)
    np.random.seed(1337)
    base_feature = np.random.randn(512).astype(np.float32)
    base_feature /= np.linalg.norm(base_feature)

    # Initial Observation at t=0
    gallery.update(track_id=42, new_embedding=base_feature, timestamp=current_time)
    print(f"[+] Track #42 registered in appearance gallery at t={current_time:.1f}s")

    # Simulate 5-second occlusion
    simulated_future_time = current_time + 5.0
    gallery.purge_expired(simulated_future_time)
    assert 42 in gallery.gallery, "[X] Error: Track expired prematurely during 5-second occlusion!"

    # Entity reappears with slight noise (illumination change)
    noise = np.random.normal(0, 0.05, 512).astype(np.float32)
    reappeared_feature = base_feature + noise
    reappeared_feature /= np.linalg.norm(reappeared_feature)

    stored_embedding = gallery.gallery[42]
    cosine_distance = 1.0 - float(np.dot(reappeared_feature, stored_embedding))

    print(f"[+] Reappearance Cosine Distance : {cosine_distance:.4f} (Threshold: <= 0.35)")
    assert cosine_distance <= 0.35, f"[X] Error: Cosine distance too high: {cosine_distance}"

    print("[+] SUCCESS: Re-ID Appearance Gallery passed 5-second occlusion test!")
    print("=" * 65)


if __name__ == "__main__":
    test_gallery_occlusion_persistence()