import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import threading
import cv2
import numpy as np


class SpatialHeatmapEngine:
    def __init__(
        self,
        resolution_wh: tuple[int, int] = (640, 480),
        blur_kernel: int = 41,
        point_radius: int = 12,
        overlay_alpha: float = 0.58
    ) -> None:
        self.width, self.height = int(resolution_wh[0]), int(resolution_wh[1])
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.point_radius = max(2, int(point_radius))
        self.overlay_alpha = float(np.clip(overlay_alpha, 0.1, 0.9))

        self._lock = threading.Lock()
        self.density_grid = np.zeros((self.height, self.width), dtype=np.float32)
        self.total_samples = 0

    def add_point(
        self,
        cx: float,
        cy: float,
        source_w: int | None = None,
        source_h: int | None = None,
        weight: float = 1.0
    ) -> None:
        sw = source_w if source_w and source_w > 0 else self.width
        sh = source_h if source_h and source_h > 0 else self.height

        gx = int(np.clip((cx / float(sw)) * self.width, 0, self.width - 1))
        gy = int(np.clip((cy / float(sh)) * self.height, 0, self.height - 1))

        with self._lock:
            cv2.circle(
                self.density_grid,
                (gx, gy),
                self.point_radius,
                float(weight),
                thickness=-1
            )
            self.total_samples += 1

    def add_points_batch(self, points: list[tuple[float, float, int, int]], weight: float = 1.0) -> None:
        if not points:
            return
        stamp = np.zeros((self.height, self.width), dtype=np.float32)
        for cx, cy, sw, sh in points:
            src_w = sw if sw > 0 else self.width
            src_h = sh if sh > 0 else self.height
            gx = int(np.clip((cx / float(src_w)) * self.width, 0, self.width - 1))
            gy = int(np.clip((cy / float(src_h)) * self.height, 0, self.height - 1))
            cv2.circle(stamp, (gx, gy), self.point_radius, float(weight), thickness=-1)

        with self._lock:
            self.density_grid += stamp
            self.total_samples += len(points)

    def reset(self) -> None:
        with self._lock:
            self.density_grid.fill(0.0)
            self.total_samples = 0

    def render_overlay(self, background_bgr: np.ndarray | None = None) -> np.ndarray:
        if background_bgr is not None:
            base = cv2.resize(background_bgr, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        else:
            base = np.full((self.height, self.width, 3), 18, dtype=np.uint8)
            for gx in range(0, self.width, 64):
                cv2.line(base, (gx, 0), (gx, self.height), (32, 40, 56), 1)
            for gy in range(0, self.height, 64):
                cv2.line(base, (0, gy), (self.width, gy), (32, 40, 56), 1)

        with self._lock:
            grid_copy = self.density_grid.copy()
            samples = self.total_samples

        peak = float(np.max(grid_copy))
        if peak <= 0.0:
            cv2.putText(
                base,
                "NO SPATIAL DENSITY DATA YET",
                (155, self.height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (148, 163, 184),
                2,
                cv2.LINE_AA
            )
            return base

        smoothed = cv2.GaussianBlur(grid_copy, (self.blur_kernel, self.blur_kernel), 0)
        log_scaled = np.log1p(smoothed)
        max_val = float(np.max(log_scaled))
        if max_val <= 1e-8:
            return base

        norm_u8 = np.clip((log_scaled / max_val) * 255.0, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)

        mask = (norm_u8 > 14).astype(np.float32) * self.overlay_alpha
        mask_3ch = np.dstack([mask, mask, mask])

        blended = (
            base.astype(np.float32) * (1.0 - mask_3ch)
            + colored.astype(np.float32) * mask_3ch
        ).astype(np.uint8)

        cv2.rectangle(blended, (10, 10), (320, 44), (15, 23, 42), -1)
        cv2.putText(
            blended,
            f"SPATIAL HEATMAP | SAMPLES: {samples}",
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (56, 189, 248),
            1,
            cv2.LINE_AA
        )

        bar_x, bar_y, bar_w, bar_h = self.width - 170, 14, 150, 14
        grad = np.linspace(0, 255, bar_w, dtype=np.uint8).reshape(1, bar_w)
        grad_bar = cv2.applyColorMap(np.repeat(grad, bar_h, axis=0), cv2.COLORMAP_JET)
        blended[bar_y:bar_y + bar_h, bar_x:bar_x + bar_w] = grad_bar
        cv2.putText(blended, "LOW", (bar_x, bar_y + bar_h + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (203, 213, 225), 1, cv2.LINE_AA)
        cv2.putText(blended, "HIGH", (bar_x + bar_w - 28, bar_y + bar_h + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (248, 113, 113), 1, cv2.LINE_AA)

        return blended

    def render_jpeg(self, background_bgr: np.ndarray | None = None, quality: int = 85) -> bytes:
        img = self.render_overlay(background_bgr=background_bgr)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("[X] Failed to JPEG-encode spatial heatmap.")
        return buf.tobytes()