import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import shutil
from unittest.mock import MagicMock, patch
import cv2
import numpy as np

from src.perception.add_hard_negatives import inject_hard_negatives
from src.vlm.schemas import IncidentSeverity
from src.vlm.vlm_engine import GroundedVLMAuditor


def test_remote_vlm_payload_headers_and_fallback() -> None:
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    cv2.line(frame, (200, 100), (320, 380), (20, 20, 20), 5)

    old_key = os.environ.get("VLM_API_KEY")
    os.environ["VLM_API_KEY"] = "test-secret-key"
    try:
        auditor = GroundedVLMAuditor(
            mode="remote",
            remote_api_url="https://example.trycloudflare.com/v1/chat/completions"
        )
        assert auditor.active_backend == "qwen2.5-vl-remote"

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "choices": [{
                "message": {
                    "content": (
                        '{"scene_description": "Diagonal structural shear crack across plaster.", '
                        '"box_2d": [190, 300, 810, 520], '
                        '"severity": "HIGH", '
                        '"recommended_action": "Inspect load-bearing column immediately."}'
                    )
                }
            }]
        }

        with patch("src.vlm.vlm_engine.requests.post", return_value=mock_resp) as mock_post:
            rep = auditor.audit_frame(
                image_bgr=frame,
                incident_id="inc_remote_ok",
                zone_id="zone_alpha_danger",
                track_id=5,
                entity_class="crack"
            )
            assert rep.audit_backend == "qwen2.5-vl-remote"
            assert rep.severity == IncidentSeverity.HIGH
            assert "shear crack" in rep.vlm_scene_description.lower()
            sent_headers = mock_post.call_args.kwargs["headers"]
            assert sent_headers["Authorization"] == "Bearer test-secret-key"
            assert sent_headers["ngrok-skip-browser-warning"] == "true"

        with patch("src.vlm.vlm_engine.requests.post", side_effect=TimeoutError("tunnel timeout")):
            rep_fb = auditor.audit_frame(
                image_bgr=frame,
                incident_id="inc_remote_fb",
                zone_id="zone_alpha_danger",
                track_id=6,
                entity_class="crack"
            )
            assert rep_fb.audit_backend == "edge-cv-crack-morph"
            assert rep_fb.severity in {
                IncidentSeverity.LOW,
                IncidentSeverity.MEDIUM,
                IncidentSeverity.HIGH,
                IncidentSeverity.CRITICAL
            }
    finally:
        if old_key is None:
            os.environ.pop("VLM_API_KEY", None)
        else:
            os.environ["VLM_API_KEY"] = old_key


def test_seg_mask_millimeter_calibration() -> None:
    frame = np.full((480, 640, 3), 210, dtype=np.uint8)
    seg_mask = np.zeros((480, 640), dtype=np.uint8)
    cv2.rectangle(seg_mask, (280, 120), (290, 320), 255, -1)

    old_cal = os.environ.get("SENTINEL_PIXELS_PER_MM")
    os.environ["SENTINEL_PIXELS_PER_MM"] = "10.0"
    try:
        auditor = GroundedVLMAuditor(mode="auto", remote_api_url="")
        rep = auditor.audit_frame(
            image_bgr=frame,
            incident_id="inc_mm_test",
            zone_id="zone_alpha_danger",
            track_id=2,
            entity_class="crack",
            hint_bbox_xyxy=(250.0, 100.0, 320.0, 340.0),
            seg_mask=seg_mask
        )
        assert "mm" in rep.vlm_scene_description.lower()
        assert rep.severity in {IncidentSeverity.HIGH, IncidentSeverity.CRITICAL}
    finally:
        if old_cal is None:
            os.environ.pop("SENTINEL_PIXELS_PER_MM", None)
        else:
            os.environ["SENTINEL_PIXELS_PER_MM"] = old_cal


def test_hard_negative_injector_creates_empty_labels() -> None:
    tmp_root = PROJECT_ROOT / "data" / "raw" / "incidents" / "_hn_test_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)

    neg_dir = tmp_root / "hard_negatives"
    ds_dir = tmp_root / "processed"
    neg_dir.mkdir(parents=True, exist_ok=True)
    (ds_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (ds_dir / "valid" / "images").mkdir(parents=True, exist_ok=True)

    for i in range(5):
        img = np.full((64, 64, 3), 150, dtype=np.uint8)
        cv2.imwrite(str(neg_dir / f"window_{i}.jpg"), img)

    stats = inject_hard_negatives(neg_dir, ds_dir, val_ratio=0.2)
    assert stats["train_added"] == 4
    assert stats["valid_added"] == 1

    train_txts = list((ds_dir / "train" / "labels").glob("hn_*.txt"))
    val_txts = list((ds_dir / "valid" / "labels").glob("hn_*.txt"))
    assert len(train_txts) == 4 and len(val_txts) == 1
    assert all(p.read_text(encoding="utf-8") == "" for p in train_txts + val_txts)

    shutil.rmtree(tmp_root)