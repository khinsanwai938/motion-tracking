import os
import re
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from src.schemas import Track

# OpenCV's own face models (no extra pip packages needed).
#   YuNet = face detector, SFace = face embedding / recognition.
MODEL_URLS = {
    "face_detection_yunet_2023mar.onnx": [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
    ],
    "face_recognition_sface_2021dec.onnx": [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
    ],
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
UNKNOWN = "Unknown"


def _ensure_model(models_dir: Path, filename: str) -> str:
    path = models_dir / filename
    if path.exists() and path.stat().st_size > 100_000:
        return str(path)
    models_dir.mkdir(parents=True, exist_ok=True)
    last_err = None
    for url in MODEL_URLS[filename]:
        try:
            print(f"[FaceIdentifier] Downloading {filename} ...")
            urllib.request.urlretrieve(url, path)
            if path.stat().st_size > 100_000:
                return str(path)
        except Exception as e:  # try next mirror
            last_err = e
    raise RuntimeError(
        f"Could not download {filename}. Download it manually into '{models_dir}'. "
        f"URLs: {MODEL_URLS[filename]}. Last error: {last_err}"
    )


class FaceIdentifier:
    """
    Puts a real name on each tracked person.

    How it works:
      1. At startup, every photo in `known_faces/` is turned into a face
         embedding. File name = person name ("Aung.jpg", "Aung_2.jpg" -> "Aung").
      2. For each tracked person we crop the head/shoulder area of the box,
         find the face, compute its embedding and compare to enrolled ones.
      3. Once the same name wins `confirm_hits` times in a row, the name is
         locked to that track ID, so recognition stops costing time for them.
         Unrecognised people stay "Unknown" and are retried every
         `retry_every_n_frames` frames.

    Names are tied to the tracker's ID. If ByteTrack loses someone and gives
    them a new ID, they are simply recognised again.
    """

    def __init__(self, known_dir: str = "known_faces", models_dir: str = "models",
                 match_threshold: float = 0.363, retry_every_n_frames: int = 10,
                 confirm_hits: int = 2, min_face_score: float = 0.7):
        models = Path(models_dir)
        det_path = _ensure_model(models, "face_detection_yunet_2023mar.onnx")
        rec_path = _ensure_model(models, "face_recognition_sface_2021dec.onnx")

        self.detector = cv2.FaceDetectorYN.create(det_path, "", (320, 320), min_face_score, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(rec_path, "")
        self.match_threshold = match_threshold   # 0.363 = cosine threshold recommended for SFace
        self.retry_every_n_frames = retry_every_n_frames
        self.confirm_hits = confirm_hits

        self._known: list[tuple[str, np.ndarray]] = []
        self._locked: dict[int, str] = {}                    # track_id -> confirmed name
        self._pending: dict[int, tuple[str, int]] = {}       # track_id -> (name, consecutive hits)
        self._last_try: dict[int, int] = {}                  # track_id -> frame index of last attempt
        self._frame_idx = 0

        self._enroll(Path(known_dir))

    @property
    def has_known_faces(self) -> bool:
        return bool(self._known)

    # ---------- enrollment ----------
    def _enroll(self, known_dir: Path) -> None:
        if not known_dir.is_dir():
            print(f"[FaceIdentifier] '{known_dir}' not found - no one enrolled.")
            return
        for img_path in sorted(known_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            name = re.sub(r"[_\-\s]*\d+$", "", img_path.stem) or img_path.stem
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[FaceIdentifier] Could not read {img_path.name}, skipped.")
                continue
            feat = self._embed_largest_face(img)
            if feat is None:
                print(f"[FaceIdentifier] No face found in {img_path.name}, skipped.")
                continue
            self._known.append((name, feat))
        names = sorted({n for n, _ in self._known})
        print(f"[FaceIdentifier] Enrolled {len(self._known)} photo(s): {names}")

    # ---------- face helpers ----------
    def _detect_faces(self, img: np.ndarray) -> np.ndarray | None:
        h, w = img.shape[:2]
        if h < 20 or w < 20:
            return None
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img)
        return faces

    def _embed_largest_face(self, img: np.ndarray) -> np.ndarray | None:
        faces = self._detect_faces(img)
        if faces is None or len(faces) == 0:
            return None
        face = max(faces, key=lambda f: f[2] * f[3])
        aligned = self.recognizer.alignCrop(img, face)
        return self.recognizer.feature(aligned)

    def _match(self, feat: np.ndarray) -> tuple[str, float]:
        best_name, best_score = UNKNOWN, -1.0
        for name, known_feat in self._known:
            score = self.recognizer.match(feat, known_feat, cv2.FaceRecognizerSF_FR_COSINE)
            if score > best_score:
                best_name, best_score = name, score
        if best_score < self.match_threshold:
            return UNKNOWN, best_score
        return best_name, best_score

    # ---------- per-frame ----------
    def identify(self, frame: np.ndarray, tracks: list[Track]) -> list[Track]:
        self._frame_idx += 1
        fh, fw = frame.shape[:2]

        for t in tracks:
            if t.track_id in self._locked:
                t.name = self._locked[t.track_id]
                continue

            t.name = UNKNOWN if self._known else None
            if not self._known:
                continue

            if self._frame_idx - self._last_try.get(t.track_id, -10**9) < self.retry_every_n_frames:
                continue
            self._last_try[t.track_id] = self._frame_idx

            # Face lives in the top part of a person box - crop it (faster + fewer false faces)
            x1, y1, x2, y2 = t.bbox
            box_h = y2 - y1
            cx1, cx2 = max(0, int(x1)), min(fw, int(x2))
            cy1, cy2 = max(0, int(y1)), min(fh, int(y1 + box_h * 0.5))
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            feat = self._embed_largest_face(crop)
            if feat is None:
                continue

            name, _score = self._match(feat)
            if name == UNKNOWN:
                self._pending.pop(t.track_id, None)
                continue

            prev_name, hits = self._pending.get(t.track_id, (name, 0))
            hits = hits + 1 if prev_name == name else 1
            self._pending[t.track_id] = (name, hits)
            if hits >= self.confirm_hits:
                self._locked[t.track_id] = name
                self._pending.pop(t.track_id, None)
                t.name = name

        self._prune({t.track_id for t in tracks})
        return tracks

    def _prune(self, active_ids: set[int]) -> None:
        # keep memory bounded on long runs; locked names are kept in case the ID reappears
        if len(self._last_try) > 500:
            for tid in [i for i in self._last_try if i not in active_ids and i not in self._locked]:
                self._last_try.pop(tid, None)
                self._pending.pop(tid, None)
