# motionTrace — Object Tracking, Distance & Speed Estimation

**Project brief for engineering handoff**
Status: Phase 0 (laptop webcam) working demo. Phase 1 (A8 Mini RTSP) scaffolded, not field-tested.

---

## 1. Goal

Detect and track humans in a live camera feed, estimate each tracked person's **distance from the camera** and **movement speed**, with persistent IDs across frames. Built as a demo to validate the approach before committing to a production build.

**End use case:** track a person of interest (e.g. running/fleeing) via drone footage, reporting real-time distance and speed. This repo is the foundation — later phases add the A8 Mini gimbal camera and, eventually, a moving drone platform.

## 2. Phased Roadmap

| Phase | Camera | Status | Key difference from previous phase |
|---|---|---|---|
| **P0** | Laptop webcam, static | ✅ Working | Baseline: detection, tracking, distance, speed |
| **P1** | SIYI A8 Mini, static/tripod mount, RTSP | 🔧 Code scaffolded, not tested on hardware | Different lens (needs own calibration), network stream instead of local device |

**Important constraint to keep in mind:** everything built so far assumes **the camera does not move**. P2 is a materially harder problem, not a config change — don't let it get folded into P0/P1 scope silently.

## 3. Architecture

```
Camera Source → Detector+Tracker → Distance Estimator → Speed Estimator → Overlay/Display
   (webcam/          (YOLO26 +          (monocular         (per-track
    A8 Mini RTSP)     ByteTrack)         depth model)        history)
```

Design principle: **the core pipeline is pure Python with no web framework dependency.** It's a library that can run as a local demo (`main.py`, OpenCV window) today, and later gets wrapped by a service layer (FastAPI — not yet built) without changing any of this core code.

### Data contract between stages
```python
@dataclass
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    timestamp: float
    distance_m: float | None
    speed_kmh: float | None
```
Every stage consumes and returns a list of `Track` objects. This is the interface — if you're extending any stage, don't break this shape.

## 4. Key Technical Decisions (and why)

| Decision | Choice | Reasoning |
|---|---|---|
| Detection + tracking | YOLO26 (nano/small) + ByteTrack via `ultralytics` | Off-the-shelf, persistent IDs out of the box, no custom tracker to maintain |
| Distance estimation | **Monocular depth model** (Depth Anything V2, Metric-Indoor checkpoint) | No manual calibration/floor-point input required; works regardless of person's pose/height, unlike the simpler pinhole/bbox-height method we started with and abandoned |
| Speed estimation | Frame-to-frame displacement (using distance + bbox motion), smoothed over a rolling window | Raw frame-to-frame speed is noisy; smoothing avoids jumpy numbers on demo day |
| Camera abstraction | Common `CameraSource` interface (`webcam_source.py`, `a8mini_source.py`) | Swapping cameras later requires zero changes to the tracking/metrics pipeline |
| Calibration | One-time automatic checkerboard capture (`scripts/auto_calibrate.py`) | No manual point-picking or typing distances — still needed for the **speed** calculation's lateral-motion conversion, even though distance itself no longer needs it |

## 5. Project Structure

```
motionTrace/
├── configs/
│   ├── camera_webcam.yaml          # gitignored — per-machine calibration
│   └── camera_a8mini.yaml          # SIYI A8 Mini RTSP + calibration config
├── src/
│   ├── camera/
│   │   ├── base.py                 # CameraSource interface
│   │   ├── webcam_source.py
│   │   └── a8mini_source.py        # RTSP, threaded capture, auto-reconnect
│   ├── detection_tracking/
│   │   └── tracker.py              # YOLO26 + ByteTrack wrapper
│   ├── metrics/
│   │   ├── depth_model.py          # Depth Anything V2 wrapper
│   │   ├── distance_estimator.py
│   │   └── speed_estimator.py      # owns its own per-track_id history (see §7)
│   ├── visualization/
│   │   └── overlay.py
│   ├── schemas.py                  # Track dataclass — the shared data contract
│   ├── pipeline.py                 # orchestrates the full loop; build_pipeline() factory
│   └── worker.py                   # background-thread wrapper (for future API integration)
├── tests/
├── main.py                         # entry point — run this to launch the demo
├── requirements.txt
└── .gitignore
```

## 6. Setup & Running

```bash
python -m venv venv
venv\Scripts\activate

# GPU install (adjust cu-tag to your CUDA setup — see pytorch.org/get-started/locally)
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```


Run the demo (laptop webcam):
```bash
python main.py
```
Press `q` to quit. To switch to the A8 Mini once hardware is connected, change `camera_type="webcam"` to `camera_type="a8mini"` in `main.py` — see §8 for network setup required first.

## 7. Known Bugs Already Fixed (context for anyone reading old commits/history)

- **Depth field bug:** the HF `depth-estimation` pipeline returns `result["depth"]` (a normalized 0–255 image for *display only*) and `result["predicted_depth"]` (actual metric depth). We were initially reading the wrong one — fixed to use `predicted_depth`.
- **Speed always 0.0 bug:** `Track` objects are recreated fresh every frame by the tracker, so history stored *on* the `Track` object reset every frame. Fixed by moving history into `SpeedEstimator`, keyed by `track_id`, persisting across frames.
- **1080p lag:** fixed via threaded camera capture (always serves the latest frame instead of blocking on a growing OS buffer), MJPG fourcc for webcams, throttled depth-model inference (not every frame), and `imgsz=640` for YOLO instead of full resolution.

## 8. A8 Mini (Phase 1) Network Setup — required before RTSP will connect

- Camera default IP: `192.168.144.25`
- RTSP URL: `rtsp://192.168.144.25:8554/main.264`
- PC needs a **static IP** on the same subnet (e.g. `192.168.144.20`), connected via Ethernet
- Test with `ping 192.168.144.25`, then verify the raw stream opens in VLC before touching the Python code — isolates network issues from code issues
- **Not yet tested on real hardware** — this is the next validation step

## 9. Known Limitations (be upfront about these in any demo/review)

- Distance/speed assume a **static, non-panning camera**. If the A8 Mini's gimbal moves during tracking, or once this moves to a drone, the math breaks — that's separate, unsolved future work.
- The depth model's "Metric-Indoor" checkpoint is trained mostly on indoor benchmark data — expect good accuracy at typical room/hallway range, more drift at longer outdoor distances.
- Zoom changes on the A8 Mini invalidate its calibrated focal length — keep zoom fixed during any test/demo.
- This is an **estimation system**, not measurement-grade — appropriate for an analytics/demo use case, not something to promise centimeter/kph precision on without a depth sensor or stereo rig.

## 10. Suggested Task Split

**Engineer A — Detection & Tracking**
- `src/camera/`, `src/detection_tracking/`
- Multi-person tracking validation (ID stability across occlusion/re-entry)
- A8 Mini hardware integration/testing (network setup, RTSP stream stability, reconnect logic)

**Engineer B — Metrics & Calibration**
- `src/metrics/`, `scripts/auto_calibrate.py`
- Accuracy tuning: smoothing window size, depth model checkpoint choice, calibration validation
- Eventually: production service layer (`worker.py` is the scaffold for this — background thread + queue, ready to wrap in an API later)

---

*Questions or architecture decisions worth revisiting should be raised before Phase 1 hardware testing begins — some of the calibration/accuracy assumptions here are demo-appropriate but not yet validated against real walking/running data.*


## Person names (face recognition)

1. Put photos in `known_faces/` - file name is the displayed name (`Aung.jpg`, `Aung_2.jpg` -> "Aung").
2. Run `python main.py`. On first run the two small OpenCV face models (~40 MB total) download into `models/`.
3. The overlay shows `Name (ID n) | distance | speed`. Orange box = face seen but not matched ("Unknown").
   If no face is visible yet, only `ID n` is shown.

Notes: needs a reasonably front-facing face (head-and-shoulders visible); a name sticks to the tracker ID, so after a long
occlusion the person is re-recognised. Disable with `build_pipeline(recognize_names=False)`. Tune `match_threshold` in
`FaceIdentifier` (higher = stricter). Only enrol people who consented.
