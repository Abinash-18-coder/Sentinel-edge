"""
src/perception/reid_extractor.py
Deep visual appearance feature extractor generating L2-normalized 512-dim embeddings.
Employs a lightweight CNN backbone compatible with PyTorch 2.4+ and Python 3.12.5 on Windows.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models


class AppearanceEmbeddingExtractor:
    """
    Extracts L2-normalized 512-dimensional appearance vectors from cropped entity images.
    """

    def __init__(self, device: str | None = None) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[*] Initializing Appearance Re-ID Extractor on device: {self.device}...")

        # Load MobileNetV3-Small as a high-speed feature extractor backbone
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = backbone.classifier[0].in_features

        # Replace classification head with 512-dim projection embedding layer
        backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512)
        )

        self.model = backbone.to(self.device).eval()

        # Standard Re-ID input normalization (256x128 resolution)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("[+] Re-ID Appearance Extractor initialized successfully.")

    @torch.no_grad()
    def extract_crops(self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
        """
        Extracts L2-normalized embeddings for a list of bounding boxes from a single frame.
        
        Args:
            frame_bgr: Full raw BGR image frame (H, W, C).
            boxes_xyxy: (N, 4) array of coordinates [x1, y1, x2, y2].

        Returns:
            (N, 512) float32 numpy array of normalized feature embeddings.
        """
        if len(boxes_xyxy) == 0:
            return np.empty((0, 512), dtype=np.float32)

        h, w = frame_bgr.shape[:2]
        crop_tensors: list[torch.Tensor] = []

        for box in boxes_xyxy:
            x1, y1, x2, y2 = box.astype(int)
            # Clip bounds defensively to prevent out-of-frame slicing errors
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Check for degenerate zero-area crop
            if (x2 - x1) < 2 or (y2 - y1) < 2:
                # Provide a synthetic neutral patch
                crop = np.zeros((256, 128, 3), dtype=np.uint8)
            else:
                crop = frame_bgr[y1:y2, x1:x2]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            crop_tensors.append(self.transform(crop))

        batch = torch.stack(crop_tensors, dim=0).to(self.device)
        features: torch.Tensor = self.model(batch)

        # Mathematical L2 Normalization across the 512 dimension
        norm_features = torch.nn.functional.normalize(features, p=2, dim=1)
        return norm_features.cpu().numpy().astype(np.float32)