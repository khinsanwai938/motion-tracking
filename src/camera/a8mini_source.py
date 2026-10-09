import os
import time
import threading
import cv2
import yaml
import numpy as np
from .base import CameraSource

DEFAULT_CONFIG_PATH = "configs/camera_a8mini.yaml"

# Force RTSP over TCP instead of UDP — UDP drops packets over WiFi/long links
# and causes visible corruption/green blocks; TCP is more reliable for this use case.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


class A8MiniSource(CameraSource):
    """
    RTSP source for the SIYI A8 Mini gimbal camera.
    Same CameraSource interface as WebcamSource — the rest of the pipeline
    (tracker, distance/speed estimators, worker) doesn't know or care which
    camera is feeding it frames.
    """

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.rtsp_url = cfg["camera"]["rtsp_url"]
        self.width = cfg["camera"]["width"]
        self.height = cfg["camera"]["height"]
        self.focal_length_px = float(
            cfg.get("calibration", {}).get("focal_length_px") or self.width
        )
        self.reconnect_attempts = cfg.get("connection", {}).get("reconnect_attempts", 5)
        self.reconnect_delay_sec = cfg.get("connection", {}).get("reconnect_delay_sec", 2)

        self.cap = self._connect()

        self._latest_frame = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _connect(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize latency — don't let stale frames queue up
        if not cap.isOpened():
            raise ConnectionError(
                f"Could not connect to A8 Mini at {self.rtsp_url}. "
                f"Check: (1) Ethernet connected, (2) PC has a static IP on 192.168.144.x, "
                f"(3) camera powered on and streaming."
            )
        return cap

    def _capture_loop(self) -> None:
        attempts = 0
        while self._running:
            ok, frame = self.cap.read()
            if ok:
                attempts = 0
                with self._lock:
                    self._latest_frame = frame
            else:
                # RTSP streams drop occasionally (WiFi/link glitches) — reconnect instead of dying
                attempts += 1
                if attempts > self.reconnect_attempts:
                    print(f"[A8MiniSource] Lost connection after {attempts} attempts. Stopping.")
                    self._running = False
                    break
                print(f"[A8MiniSource] Frame read failed, reconnecting (attempt {attempts})...")
                time.sleep(self.reconnect_delay_sec)
                self.cap.release()
                try:
                    self.cap = self._connect()
                except ConnectionError as e:
                    print(f"[A8MiniSource] Reconnect failed: {e}")

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    def get_focal_length_px(self) -> float:
        return self.focal_length_px

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=2)
        self.cap.release()