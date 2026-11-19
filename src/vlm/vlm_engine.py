import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import base64
import gc
import json
import os
import re
import time
from typing import Any

import cv2
import numpy as np
import psutil
import requests
from PIL import Image
from pydantic import ValidationError

from src.vlm.schemas import (
    GroundingBoundingBox,
    IncidentAuditReport,
    IncidentSeverity,
    RawVLMOutput
)


class GroundedVLMAuditor:
    REMOTE_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
    LOCAL_LIGHT_MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
    MIN_FREE_RAM_GB_FOR_LOCAL_VLM = 2.2

    def __init__(
        self,
        mode: str = "auto",
        remote_api_url: str | None = None,
        remote_model_name: str | None = None,
        local_model_id: str | None = None,
        request_timeout_sec: float = 20.0
    ) -> None:
        self.mode = mode.lower()
        self.remote_api_url = remote_api_url or os.getenv("VLM_API_URL", "").strip()
        self.remote_model_name = remote_model_name or os.getenv("VLM_MODEL_NAME", self.REMOTE_MODEL_NAME)
        self.local_model_id = local_model_id or self.LOCAL_LIGHT_MODEL_ID
        self.request_timeout_sec = request_timeout_sec

        self.processor = None
        self.model = None
        self.active_backend = "edge-cv-fallback"

        self._select_backend()

    def _get_free_ram_gb(self) -> float:
        return psutil.virtual_memory().available / (1024 ** 3)

    def _select_backend(self) -> None:
        if self.mode in ("auto", "remote") and self.remote_api_url:
            self.active_backend = "qwen2.5-vl-remote"
            print(f"[+] VLM Auditor configured for Remote Qwen2.5-VL endpoint: {self.remote_api_url}")
            return

        if self.mode == "local":
            free_gb = self._get_free_ram_gb()
            if free_gb < self.MIN_FREE_RAM_GB_FOR_LOCAL_VLM:
                print(
                    f"[!] RAM Guard Triggered: Only {free_gb:.2f} GB RAM free "
                    f"(need >= {self.MIN_FREE_RAM_GB_FOR_LOCAL_VLM:.1f} GB). "
                    "Using 8GB-safe edge-cv-fallback."
                )
                self.active_backend = "edge-cv-fallback"
                return
            self._load_local_vlm()
            return

        self.active_backend = "edge-cv-fallback"
        print(
            f"[+] VLM Auditor initialized in 8GB-safe mode ({self.active_backend}). "
            "Set VLM_API_URL for cloud Qwen2.5-VL-3B or pass mode='local' when >=2.2GB RAM is free."
        )

    def _load_local_vlm(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor
            try:
                from transformers import AutoModelForImageTextToText as AutoVLMModel
            except ImportError:
                from transformers import AutoModelForVision2Seq as AutoVLMModel

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

            print(f"[*] Loading lightweight local VLM: {self.local_model_id} on {device}...")
            self.processor = AutoProcessor.from_pretrained(self.local_model_id)
            self.model = AutoVLMModel.from_pretrained(
                self.local_model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            ).to(device).eval()

            self.active_backend = f"local-vlm:{self.local_model_id.split('/')[-1]}"
            print(f"[+] Local VLM loaded cleanly: {self.active_backend}")
        except Exception as err:
            print(f"[!] Local VLM load skipped ({err}). Falling back to edge-cv-fallback.")
            self.processor = None
            self.model = None
            self.active_backend = "edge-cv-fallback"
            gc.collect()

    def _build_prompt(self, entity_class: str, zone_id: str) -> str:
        return (
            "You are a structural crack and industrial safety auditor. "
            f"Inspect the detected '{entity_class}' inside zone '{zone_id}'. "
            "Respond ONLY with a valid JSON object matching this exact schema: "
            '{"scene_description": "concise 1-sentence finding", '
            '"box_2d": [ymin, xmin, ymax, xmax], '
            '"severity": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL", '
            '"recommended_action": "action under 10 words"}. '
            "box_2d coordinates must be integers from 0 to 1000."
        )

    def _encode_resized_jpeg_b64(self, image_bgr: np.ndarray, max_dim: int = 512) -> str:
        h, w = image_bgr.shape[:2]
        scale = min(1.0, max_dim / float(max(h, w)))
        if scale < 1.0:
            resized = cv2.resize(
                image_bgr,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA
            )
        else:
            resized = image_bgr

        ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            raise RuntimeError("Failed to encode snapshot for VLM transport.")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def parse_and_validate_json(self, raw_text: str, fallback_box: list[int]) -> RawVLMOutput:
        candidate: dict[str, Any] | None = None

        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if fenced:
            try:
                candidate = json.loads(fenced.group(1))
            except json.JSONDecodeError:
                candidate = None

        if candidate is None:
            braced = re.search(r"(\{.*\})", raw_text, re.DOTALL)
            if braced:
                try:
                    candidate = json.loads(braced.group(1))
                except json.JSONDecodeError:
                    candidate = None

        if candidate is None:
            candidate = {}

        raw_box = candidate.get("box_2d", fallback_box)
        if not (isinstance(raw_box, list) and len(raw_box) == 4):
            raw_box = fallback_box
        clamped_box = [int(np.clip(float(v), 0, 1000)) for v in raw_box]

        raw_sev = str(candidate.get("severity", "MEDIUM")).upper()
        if "CRITICAL" in raw_sev:
            sev = IncidentSeverity.CRITICAL
        elif "HIGH" in raw_sev:
            sev = IncidentSeverity.HIGH
        elif "LOW" in raw_sev:
            sev = IncidentSeverity.LOW
        else:
            sev = IncidentSeverity.MEDIUM

        desc = str(candidate.get("scene_description", "")).strip()
        if len(desc) < 3:
            cleaned = re.sub(r"\s+", " ", raw_text).strip()
            desc = cleaned[:180] if len(cleaned) >= 3 else "Structural surface anomaly detected in monitored zone."

        action = str(candidate.get("recommended_action", "")).strip()
        if len(action) < 3:
            action = "Inspect defect area and log measurement."

        try:
            return RawVLMOutput(
                scene_description=desc[:300],
                box_2d=clamped_box,
                severity=sev,
                recommended_action=action[:120]
            )
        except ValidationError:
            return RawVLMOutput(
                scene_description="Structural defect detected; schema fallback applied.",
                box_2d=fallback_box,
                severity=IncidentSeverity.MEDIUM,
                recommended_action="Schedule manual structural inspection."
            )

    def _run_remote_qwen(
        self,
        image_bgr: np.ndarray,
        entity_class: str,
        zone_id: str,
        fallback_box: list[int]
    ) -> RawVLMOutput:
        img_b64 = self._encode_resized_jpeg_b64(image_bgr, max_dim=512)
        prompt = self._build_prompt(entity_class, zone_id)

        payload = {
            "model": self.remote_model_name,
            "temperature": 0.0,
            "max_tokens": 220,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                        }
                    ]
                }
            ]
        }

        resp = requests.post(
            self.remote_api_url,
            json=payload,
            timeout=self.request_timeout_sec
        )
        resp.raise_for_status()
        data = resp.json()
        raw_text = data["choices"][0]["message"]["content"]
        return self.parse_and_validate_json(raw_text, fallback_box)

    def _run_local_vlm(
        self,
        image_bgr: np.ndarray,
        entity_class: str,
        zone_id: str,
        fallback_box: list[int]
    ) -> RawVLMOutput:
        import torch

        h, w = image_bgr.shape[:2]
        scale = min(1.0, 512.0 / float(max(h, w)))
        if scale < 1.0:
            small_bgr = cv2.resize(
                image_bgr,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA
            )
        else:
            small_bgr = image_bgr

        rgb_pil = Image.fromarray(cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB))
        prompt = self._build_prompt(entity_class, zone_id)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        chat_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=chat_text, images=[rgb_pil], return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            out_ids = self.model.generate(**inputs, max_new_tokens=160, do_sample=False)

        prompt_len = inputs["input_ids"].shape[1]
        trimmed = out_ids[:, prompt_len:]
        raw_text = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        return self.parse_and_validate_json(raw_text, fallback_box)

    def _run_edge_cv_audit(
        self,
        image_bgr: np.ndarray,
        entity_class: str,
        zone_id: str,
        hint_bbox_xyxy: tuple[float, float, float, float] | None = None
    ) -> RawVLMOutput:
        h, w = image_bgr.shape[:2]

        if hint_bbox_xyxy is not None:
            x1, y1, x2, y2 = [int(v) for v in hint_bbox_xyxy]
            x1 = max(0, min(x1, w - 2))
            y1 = max(0, min(y1, h - 2))
            x2 = max(x1 + 2, min(x2, w))
            y2 = max(y1 + 2, min(y2, h))
        else:
            x1, y1, x2, y2 = 0, 0, w, h

        roi = image_bgr[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            cx, cy, cw, ch = cv2.boundingRect(largest)
            gx1 = x1 + cx
            gy1 = y1 + cy
            gx2 = x1 + cx + cw
            gy2 = y1 + cy + ch
            edge_density = float(np.count_nonzero(edges)) / float(max(1, roi.shape[0] * roi.shape[1]))
        else:
            gx1, gy1, gx2, gy2 = x1, y1, x2, y2
            edge_density = 0.02

        norm_box = [
            int(np.clip((gy1 / float(h)) * 1000.0, 0, 1000)),
            int(np.clip((gx1 / float(w)) * 1000.0, 0, 1000)),
            int(np.clip((gy2 / float(h)) * 1000.0, 0, 1000)),
            int(np.clip((gx2 / float(w)) * 1000.0, 0, 1000))
        ]

        box_area_ratio = float((gx2 - gx1) * (gy2 - gy1)) / float(max(1, w * h))
        if box_area_ratio >= 0.18 or edge_density >= 0.12:
            severity = IncidentSeverity.CRITICAL
            action = "Halt scan and schedule urgent structural reinforcement."
        elif box_area_ratio >= 0.08 or edge_density >= 0.06:
            severity = IncidentSeverity.HIGH
            action = "Log fissure width and inspect load-bearing section."
        elif box_area_ratio >= 0.03:
            severity = IncidentSeverity.MEDIUM
            action = "Monitor crack propagation on next inspection pass."
        else:
            severity = IncidentSeverity.LOW
            action = "Record minor surface crack for baseline tracking."

        desc = (
            f"Detected {entity_class} in {zone_id} spanning "
            f"{gx2 - gx1}x{gy2 - gy1}px ({box_area_ratio * 100:.1f}% frame area, "
            f"edge density {edge_density * 100:.1f}%)."
        )

        return RawVLMOutput(
            scene_description=desc,
            box_2d=norm_box,
            severity=severity,
            recommended_action=action
        )

    def audit_frame(
        self,
        image_bgr: np.ndarray,
        incident_id: str,
        zone_id: str,
        track_id: int,
        entity_class: str,
        hint_bbox_xyxy: tuple[float, float, float, float] | None = None
    ) -> IncidentAuditReport:
        t0 = time.perf_counter()
        h, w = image_bgr.shape[:2]

        if hint_bbox_xyxy is not None:
            fallback_box = [
                int(np.clip((hint_bbox_xyxy[1] / float(h)) * 1000.0, 0, 1000)),
                int(np.clip((hint_bbox_xyxy[0] / float(w)) * 1000.0, 0, 1000)),
                int(np.clip((hint_bbox_xyxy[3] / float(h)) * 1000.0, 0, 1000)),
                int(np.clip((hint_bbox_xyxy[2] / float(w)) * 1000.0, 0, 1000))
            ]
        else:
            fallback_box = [150, 150, 850, 850]

        used_backend = self.active_backend
        try:
            if self.active_backend == "qwen2.5-vl-remote":
                parsed = self._run_remote_qwen(image_bgr, entity_class, zone_id, fallback_box)
            elif self.model is not None and self.processor is not None:
                parsed = self._run_local_vlm(image_bgr, entity_class, zone_id, fallback_box)
            else:
                parsed = self._run_edge_cv_audit(image_bgr, entity_class, zone_id, hint_bbox_xyxy)
        except Exception as err:
            print(f"[!] Primary VLM backend ({self.active_backend}) failed: {err}. Switching to edge-cv-fallback.")
            used_backend = "edge-cv-fallback"
            parsed = self._run_edge_cv_audit(image_bgr, entity_class, zone_id, hint_bbox_xyxy)

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        grounding_box = GroundingBoundingBox(
            ymin=parsed.box_2d[0],
            xmin=parsed.box_2d[1],
            ymax=parsed.box_2d[2],
            xmax=parsed.box_2d[3]
        )

        return IncidentAuditReport(
            incident_id=incident_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            zone_id=zone_id,
            track_id=track_id,
            entity_class=entity_class,
            vlm_scene_description=parsed.scene_description,
            hazard_grounding_box=grounding_box,
            severity=parsed.severity,
            recommended_action=parsed.recommended_action,
            audit_backend=used_backend,
            audit_latency_ms=latency_ms
        )

    def audit_incident(
        self,
        image_path: Path,
        incident_id: str,
        zone_id: str,
        track_id: int,
        entity_class: str,
        hint_bbox_xyxy: tuple[float, float, float, float] | None = None
    ) -> IncidentAuditReport:
        if not image_path.exists():
            raise FileNotFoundError(f"[X] Snapshot not found: {image_path}")
        frame_bgr = cv2.imread(str(image_path))
        if frame_bgr is None:
            raise ValueError(f"[X] Could not decode image: {image_path}")
        return self.audit_frame(
            image_bgr=frame_bgr,
            incident_id=incident_id,
            zone_id=zone_id,
            track_id=track_id,
            entity_class=entity_class,
            hint_bbox_xyxy=hint_bbox_xyxy
        )