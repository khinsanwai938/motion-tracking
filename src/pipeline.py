import time
import cv2
from src.camera.webcam_source import WebcamSource
from src.detection_tracking.tracker import PersonTracker
from src.metrics.distance_estimator import DistanceEstimator, HeightDistanceEstimator
from src.metrics.speed_estimator import SpeedEstimator
from src.recognition import FaceIdentifier
from src.visualization.overlay import draw_tracks


class TrackingPipeline:
    def __init__(self, camera, tracker, distance_estimator, speed_estimator, depth_every_n_frames: int = 1,
                 identifier=None):
        self.camera = camera
        self.tracker = tracker
        self.distance_estimator = distance_estimator
        self.speed_estimator = speed_estimator
        self.depth_every_n_frames = depth_every_n_frames
        self.identifier = identifier  # optional FaceIdentifier; None = no names
        self._frame_count = 0

    def run(self):
        last_frame_id = -1
        fps, yolo_ms, face_ms, depth_ms = 0.0, 0.0, 0.0, 0.0
        t_prev = time.time()

        while True:
            ok, frame = self.camera.read()
            if not ok:
                continue

            # The camera thread may not have produced a new image yet - don't re-process the old one
            frame_id = getattr(self.camera, "frame_id", None)
            if frame_id is not None:
                if frame_id == last_frame_id:
                    time.sleep(0.001)
                    continue
                last_frame_id = frame_id

            # Depth runs in a background thread; this call returns immediately
            if self._frame_count % self.depth_every_n_frames == 0:
                self.distance_estimator.set_frame(frame)

            t1 = time.time()
            tracks = self.tracker.update(frame)
            t2 = time.time()

            tracks = [self.distance_estimator.estimate(t) for t in tracks]
            tracks = [self.speed_estimator.update(t) for t in tracks]
            t3 = time.time()
            if self.identifier is not None:
                tracks = self.identifier.identify(frame, tracks)
            t4 = time.time()

            self._frame_count += 1

            # Smoothed FPS + per-stage timing. If FPS drops when you move, this shows WHICH stage got slow.
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
            t_prev = now
            yolo_ms = 0.9 * yolo_ms + 0.1 * (t2 - t1) * 1000
            face_ms = 0.9 * face_ms + 0.1 * (t4 - t3) * 1000
            depth_ms = self.distance_estimator.last_depth_ms

            frame = draw_tracks(frame, tracks)
            cv2.putText(frame, f"FPS {fps:.0f} | YOLO {yolo_ms:.0f}ms | Face {face_ms:.0f}ms | Depth(bg) {depth_ms:.0f}ms",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Human Tracking Demo (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.camera.release()
        cv2.destroyAllWindows()

def build_pipeline(single_person_mode: bool = True, recognize_names: bool = True,
                   distance_mode: str = "height") -> TrackingPipeline:
    camera = WebcamSource()
    focal_length_px = camera.get_focal_length_px()  # still needed by SpeedEstimator below

    tracker = PersonTracker(max_people=1 if single_person_mode else None)
    if distance_mode == "depth":
        # accurate but heavy: ~2.5 s per depth map on a CPU. Fine with a GPU.
        from src.metrics.depth_model import MonocularDepthEstimator
        distance_estimator = DistanceEstimator(depth_model=MonocularDepthEstimator(model_size="small"))
    else:
        # "height": geometry-based, ~0 ms, recommended on CPU
        distance_estimator = HeightDistanceEstimator(focal_length_px=focal_length_px)
    speed_estimator = SpeedEstimator(focal_length_px=focal_length_px)

    identifier = FaceIdentifier(known_dir="known_faces") if recognize_names else None
    return TrackingPipeline(camera, tracker, distance_estimator, speed_estimator,
                            depth_every_n_frames=6 if distance_mode == "depth" else 1,
                            identifier=identifier)