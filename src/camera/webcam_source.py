import os
import threading
import cv2
import yaml
import numpy as np
from .base import CameraSource

DEFAULT_CONFIG_PATH = "configs/camera_webcam.yaml"


class WebcamSource(CameraSource):
    """
    Threaded capture: a background thread continuously reads frames and
    always keeps only the latest one. The main pipeline loop never blocks
    waiting on the camera, and never processes a backlog of stale frames.
    """

    def __init__(self, device_id: int = 0, width: int = 1080, height: int = 720,
                 target_fps: int =60, config_path: str = DEFAULT_CONFIG_PATH):
        self.cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)

        # MJPG matters: most laptop webcams (including this one) only sustain
        # 1080p30 in compressed MJPEG mode. The default raw YUY2 mode caps out
        # much lower (often 720p10-15) even if you request 1080p30 — the request
        # silently gets ignored without this line.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, target_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # don't let the driver queue stale frames

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam device {device_id}")

        # Log actual negotiated settings — the camera can silently ignore requests
        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"[WebcamSource] Requested {width}x{height}@{target_fps} — "
              f"got {actual_w:.0f}x{actual_h:.0f}@{actual_fps:.0f}")

        self.width = width
        self.height = height
        self.focal_length_px = self._load_or_default(config_path, width)

        self._latest_frame = None
        self.frame_id = 0  # increments for every new camera frame
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        while self._running:
            ok, frame = self.cap.read()
            if ok:
                with self._lock:
                    self._latest_frame = frame
                    self.frame_id += 1

    @staticmethod
    def _load_or_default(config_path: str, width: int) -> float:
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            if cfg:
                focal = cfg.get("calibration", {}).get("focal_length_px") or cfg.get("focal_length_px")
                if focal:
                    return float(focal)
        return float(width)

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    def get_focal_length_px(self) -> float:
        return self.focal_length_px

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=1)
        self.cap.release()