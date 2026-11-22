import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import cv2
import numpy as np
from src.vlm.vlm_engine import GroundedVLMAuditor


def build_sample_crack() -> np.ndarray:
    rng = np.random.default_rng(7)
    img = rng.integers(190, 212, (480, 640, 3), dtype=np.uint8)
    pts = np.array([[210, 70], [235, 165], [275, 250], [315, 345], [340, 420]], dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=False, color=(20, 20, 20), thickness=5)
    return img


if __name__ == "__main__":
    url = os.getenv("VLM_API_URL", "").strip()
    if not url:
        print("[!] VLM_API_URL is not set. Testing local edge-cv-crack-morph fallback.")
    else:
        print(f"[*] Testing remote VLM endpoint: {url}")

    sample_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if sample_path and sample_path.exists():
        frame = cv2.imread(str(sample_path))
    else:
        frame = build_sample_crack()

    auditor = GroundedVLMAuditor(mode="auto")
    report = auditor.audit_frame(
        image_bgr=frame,
        incident_id="remote_vlm_probe",
        zone_id="zone_alpha_danger",
        track_id=1,
        entity_class="crack"
    )
    print(f"[+] Backend  : {report.audit_backend}")
    print(f"[+] Latency  : {report.audit_latency_ms} ms")
    print(f"[+] Severity : {report.severity.value}")
    print(f"[+] Box 2D   : {report.hazard_grounding_box.model_dump()}")
    print(f"[+] Finding  : {report.vlm_scene_description}")
    print(f"[+] Action   : {report.recommended_action}")