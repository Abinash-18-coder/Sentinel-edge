import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import threading
import time
import cv2
import numpy as np


def parse_camera_source(raw_source: int | str) -> int | str:
    if isinstance(raw_source, int):
        return raw_source
    cleaned = str(raw_source).strip()
    if cleaned.isdigit():
        return int(cleaned)
    return cleaned


def classify_source_type(source: int | str) -> str:
    if isinstance(source, int):
        return "usb_webcam"
    lower = source.lower()
    if lower.startswith("rtsp://") or lower.startswith("rtsps://"):
        return "rtsp_robot"
    if lower.startswith("http://") or lower.startswith("https://"):
        return "http_mjpeg"
    return "video_file"


def format_source_label(source: int | str) -> str:
    src_type = classify_source_type(source)
    if src_type == "usb_webcam":
        return f"cam:{source}"
    if src_type == "video_file":
        return Path(str(source)).name
    text = str(source)
    if len(text) > 36:
        return f"{text[:18]}...{text[-14:]}"
    return text


class UniversalCameraStream:
    def __init__(
        self,
        source: int | str = 0,
        max_width: int = 640,
        reconnect_delay_sec: float = 1.5
    ) -> None:
        self.source: int | str = parse_camera_source(source)
        self.source_type: str = classify_source_type(self.source)
        self.max_width = max(160, int(max_width))
        self.reconnect_delay_sec = max(0.3, float(reconnect_delay_sec))

        self._frame_lock = threading.Lock()
        self._ctrl_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_ts: float = 0.0
        self._frame_wh: tuple[int, int] = (640, 480)

        self._pending_source: int | str | None = None
        self._switch_requested: bool = False
        self._running: bool = False
        self._thread: threading.Thread | None = None

        self.status: str = "initializing"
        self.is_connected: bool = False
        self.reconnect_count: int = 0

    def start(self) -> "UniversalCameraStream":
        if self._running:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()
        return self

    def switch_source(self, new_source: int | str) -> None:
        parsed = parse_camera_source(new_source)
        with self._ctrl_lock:
            self._pending_source = parsed
            self._switch_requested = True

    def get_source_info(self) -> dict[str, str | bool | int]:
        return {
            "source": str(self.source),
            "label": format_source_label(self.source),
            "source_type": self.source_type,
            "status": self.status,
            "connected": self.is_connected,
            "reconnects": self.reconnect_count
        }

    def read_latest(self) -> tuple[bool, np.ndarray | None]:
        with self._frame_lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    def wait_for_first_frame(self, timeout_sec: float = 4.0) -> tuple[bool, np.ndarray | None]:
        deadline = time.perf_counter() + timeout_sec
        while time.perf_counter() < deadline and self._running:
            ok, frame = self.read_latest()
            if ok and frame is not None:
                return True, frame
            time.sleep(0.05)
        return False, None

    def _resize_if_needed(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w <= self.max_width:
            return frame
        scale = self.max_width / float(w)
        new_h = max(16, int(round(h * scale)))
        new_h = new_h if new_h % 2 == 0 else new_h + 1
        return cv2.resize(frame, (self.max_width, new_h), interpolation=cv2.INTER_AREA)

    def _open_capture(self, source: int | str, src_type: str) -> cv2.VideoCapture:
        if src_type == "rtsp_robot":
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;4000000"
            cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
        elif src_type == "http_mjpeg":
            cap = cv2.VideoCapture(str(source))
        elif src_type == "video_file":
            cap = cv2.VideoCapture(str(source))
        else:
            cap = cv2.VideoCapture(int(source))
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.max_width))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480.0)

        if cap.isOpened() and src_type in ("usb_webcam", "rtsp_robot", "http_mjpeg"):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            try:
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3500)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3500)
            except Exception:
                pass

        return cap

    def _interruptible_sleep(self, seconds: float) -> None:
        end_t = time.perf_counter() + seconds
        while time.perf_counter() < end_t and self._running:
            with self._ctrl_lock:
                if self._switch_requested:
                    return
            time.sleep(0.05)

    def _grab_loop(self) -> None:
        cap: cv2.VideoCapture | None = None
        consecutive_failures = 0
        file_frame_delay = 0.033

        try:
            while self._running:
                with self._ctrl_lock:
                    if self._switch_requested and self._pending_source is not None:
                        if cap is not None:
                            cap.release()
                            cap = None
                        self.source = self._pending_source
                        self.source_type = classify_source_type(self.source)
                        self._pending_source = None
                        self._switch_requested = False
                        self.is_connected = False
                        self.status = "connecting"
                        consecutive_failures = 0
                        with self._frame_lock:
                            self._latest_frame = None
                        print(f"[*] [CAMERA] Switching stream to {self.source_type}: {self.source}")

                if cap is None or not cap.isOpened():
                    self.status = "connecting"
                    cap = self._open_capture(self.source, self.source_type)
                    if not cap.isOpened():
                        cap.release()
                        cap = None
                        self.is_connected = False
                        self.status = "reconnecting"
                        self.reconnect_count += 1
                        self._interruptible_sleep(self.reconnect_delay_sec)
                        continue

                    self.is_connected = True
                    self.status = "connected"
                    consecutive_failures = 0
                    if self.source_type == "video_file":
                        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
                        native_fps = float(np.clip(native_fps, 5.0, 60.0))
                        file_frame_delay = 1.0 / native_fps
                    print(f"[+] [CAMERA] Connected ({self.source_type}): {format_source_label(self.source)}")

                ret, frame = cap.read()
                if not ret or frame is None:
                    if self.source_type == "video_file":
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        time.sleep(0.02)
                        continue

                    consecutive_failures += 1
                    if consecutive_failures >= 8:
                        print(f"[!] [CAMERA] Stream dropped ({format_source_label(self.source)}). Reconnecting...")
                        cap.release()
                        cap = None
                        self.is_connected = False
                        self.status = "reconnecting"
                        self.reconnect_count += 1
                        self._interruptible_sleep(self.reconnect_delay_sec)
                    else:
                        time.sleep(0.02)
                    continue

                consecutive_failures = 0
                resized = self._resize_if_needed(frame)
                rh, rw = resized.shape[:2]
                with self._frame_lock:
                    self._latest_frame = resized
                    self._latest_ts = time.perf_counter()
                    self._frame_wh = (rw, rh)

                if self.source_type == "video_file":
                    self._interruptible_sleep(file_frame_delay)
                else:
                    time.sleep(0.001)
        finally:
            if cap is not None:
                cap.release()
            self.is_connected = False
            self.status = "stopped"

    def stop(self, timeout_sec: float = 5.0) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=timeout_sec)
            self._thread = None