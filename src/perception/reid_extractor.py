import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torchvision.models as models


class AppearanceEmbeddingExtractor:
    def __init__(self, device: str | None = None, num_threads: int = 4) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if self.device.type == "cpu":
            torch.set_num_threads(num_threads)

        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
        self.features = backbone.features.to(self.device).eval()
        self.avgpool = backbone.avgpool.to(self.device).eval()
        self.embedding_dim = 576

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        self.input_wh = (128, 128)

        print(f"[+] Re-ID Extractor ready on {self.device} (576-dim deterministic MobileNetV3-Small)")

    def _preprocess_crop(self, crop_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(crop_bgr, self.input_wh, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        chw = np.transpose(rgb, (2, 0, 1))
        return (chw - self.mean) / self.std

    def extract_crops(
        self,
        frame_bgr: np.ndarray,
        boxes_xyxy: np.ndarray,
        max_batch: int = 16
    ) -> np.ndarray:
        if len(boxes_xyxy) == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        h, w = frame_bgr.shape[:2]
        preprocessed: list[np.ndarray] = []

        for box in boxes_xyxy:
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            if (x2 - x1) < 4 or (y2 - y1) < 4:
                crop = np.zeros((self.input_wh[1], self.input_wh[0], 3), dtype=np.uint8)
            else:
                crop = frame_bgr[y1:y2, x1:x2]

            preprocessed.append(self._preprocess_crop(crop))

        all_embeddings: list[np.ndarray] = []

        with torch.inference_mode():
            for start in range(0, len(preprocessed), max_batch):
                chunk_np = np.stack(preprocessed[start:start + max_batch], axis=0)
                batch_tensor = torch.from_numpy(chunk_np).to(self.device)
                feat_map = self.features(batch_tensor)
                pooled = self.avgpool(feat_map)
                flat = torch.flatten(pooled, 1)
                normed = torch.nn.functional.normalize(flat, p=2, dim=1)
                all_embeddings.append(normed.cpu().numpy().astype(np.float32))

        return np.vstack(all_embeddings)