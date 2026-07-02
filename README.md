# Sentinel — AI Video Analytics Console

A full-stack warehouse security & operations platform built for **Technomak**.
A single-page control-room UI backed by a **FastAPI** server running real
computer-vision pipelines (ANPR, face recognition, PPE & worker-activity
detection, vehicle detection).

Every analytics tile works on **uploaded video, photos, or screenshots** — no
special camera hardware is required to demo or use the system.

---

## Modules

| Module | Capability |
|--------|-----------|
| **Operations overview** | 4 module-aligned camera tiles (ANPR · Face Recognition · PPE · Activity), a live event feed, and pop-up alerts |
| **Gate & Access** | ANPR number-plate detection + whitelist authorization (grant/deny), face recognition (grant/deny), and an editable authorised-people list |
| **People & Safety** | PPE compliance (hard-hat / vest), worker-activity monitoring (working / on-phone / chatting / resting), a live PPE-compliance gauge, an events feed with per-row delete, and 4 warehouse camera tiles |
| **Activity Monitor** | Detailed per-worker activity classification |
| **Biometric Auth** | Register people (webcam / photo / video), verify identity, editable personnel list, access log |
| **Vehicle & Logistics** | Vehicle detection camera tile, turnaround-time tracking, manual gate override |
| **Alerts** | Every access-denied / violation event (PPE not worn, biometric denied, vehicle denied) raises a bottom-right pop-up **and** an entry in the Alerts tab |

---

## Models & AI pipelines

| Task | Engine(s) |
|------|-----------|
| **Number plates (ANPR)** | **PaddleOCR** (primary) with **EasyOCR** fallback; YOLOv8n for the surrounding vehicle box; cross-frame majority voting for accuracy |
| **Face recognition** | **DeepFace / Facenet** — 128-d embedding (primary); **OpenCV Haar-cascade face-crop** embedding as a TensorFlow-free fallback. Cosine-similarity matching against stored, engine-tagged encodings |
| **PPE (hard-hat / vest)** | Person boxes from **YOLOv8n** (COCO); PPE model auto-downloaded from HuggingFace (`keremberke/yolov8s-hard-hat-detection`) |
| **Worker activity** | **YOLOv8n-pose** (resting) + YOLOv8n COCO cell-phone class (on-phone) + proximity heuristic (chatting) |

Model weights (`*.pt`) and OCR/face models download automatically on first use
and are **not** stored in git.

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Backend / API | **Python** — FastAPI + Uvicorn (REST/JSON + MJPEG) |
| Computer vision | OpenCV, Ultralytics (YOLOv8 / YOLOv8-pose), PaddleOCR, EasyOCR, DeepFace (tf-keras) |
| Database | **SQLite** (`db.py`, WAL mode) + JSON for face encodings |
| Frontend | Single self-contained **HTML + CSS + vanilla JavaScript** file — no framework, no build step |

---

## Project structure

```
Ai Camera/
├── main.py                                   # FastAPI app; serves the console at /console
├── db.py                                     # SQLite schema + connection helper
├── detector.py                               # Person/vehicle YOLO detector wrapper
├── technomak-video-analytics-console.html    # Entire frontend UI
├── requirements.txt
├── start.bat / start_server.bat              # Windows launchers
│
├── routers/
│   ├── anpr.py          # ANPR video job, single-image scan, whitelist, access log
│   ├── biometric.py     # Face register/verify (image + video), persons CRUD, log
│   ├── safety.py        # PPE + activity analyze-image/analyze-video, events log
│   ├── streams.py       # MJPEG camera stream endpoints (synthetic placeholders)
│   └── vehicles.py      # Vehicle entry/exit, turnaround, demo detection
│
├── services/
│   ├── anpr_service.py       # PaddleOCR (EasyOCR fallback) plate extraction
│   ├── bio_service.py        # DeepFace/Facenet + OpenCV embedding, verify
│   ├── ppe_service.py        # Person + hard-hat/vest detection & compliance
│   ├── activity_service.py   # Pose + phone + proximity activity classification
│   └── yolo_service.py       # Shared YOLO vehicle detector
│
├── models/              # YOLO weights (auto-downloaded; gitignored)
└── data/                # sentinel.db, encodings.json, faces/ (runtime; gitignored)
```

---

## Setup & run

**Requirements:** Python 3.10+

```bash
# 1. Virtual environment
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

# 2. Dependencies
pip install -r requirements.txt

# 3. Start the server
python main.py                   # or: uvicorn main:app --host 0.0.0.0 --port 8000
```

> First run downloads the OCR / face / YOLO model weights automatically
> (several hundred MB total). PPE and pose models download on first detection.

**Open the console:** <http://127.0.0.1:8000/console>

Use the `/console` URL (not the raw `file://` path) so browser features that need
a secure context — e.g. the webcam for biometric enrolment — work reliably.

Interactive API docs: <http://127.0.0.1:8000/docs>

---

## Key API endpoints

### ANPR — `/api/v1/anpr`
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload` | Upload gate video → background OCR job |
| `GET` | `/job/{id}` | Poll job status + detected plates |
| `GET` | `/job/{id}/track` · `/frame` · `/video` | Moving detection-box track / annotated frame / annotated video |
| `POST` | `/scan-image` | Instant single-image / screenshot plate scan |
| `GET`/`POST` | `/authorized` | List / add whitelisted vehicles |
| `PUT`/`DELETE` | `/authorized/{plate}` | Update / remove a whitelisted vehicle |
| `GET` | `/log` | Recent access log |

### Biometric — `/api/v1/biometric`
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/register` · `/register-video` | Enrol a person from a photo or a video clip |
| `POST` | `/verify` · `/verify-video` | Verify identity from a photo or a clip → grant/deny |
| `GET` | `/persons` | List registered people |
| `PATCH`/`DELETE` | `/persons/{employee_id}` | Edit / remove a person |
| `GET`/`POST` | `/log` | Access log |

### People & Safety — `/api/v1/safety`
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/status` | Model availability (hat / vest) |
| `POST` | `/analyze-image` · `/analyze-video` | PPE + activity detection → annotated frame + per-person results |
| `GET` | `/log` | PPE / activity event feed |
| `DELETE` | `/log/{id}` · `/log?kind=ppe\|activity` | Remove one event / clear a kind / clear all |

### Vehicles — `/api/v1/vehicles`
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/entry` · `/exit` | Record vehicle entry / exit (turnaround) |
| `POST` | `/demo-upload` → `GET /demo-job/{id}` | Vehicle-detection job on an uploaded clip |

### Streams
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/stream/{cam_id}` | MJPEG stream (`cam01`, `cam02`, `cam05`, `cam08`, `cam12`) — synthetic placeholder frames |

---

## Pipelines at a glance

**ANPR** — sample frames (early-stop on a confident read) → downscale → PaddleOCR/EasyOCR
→ clean & validate against a plate regex, correcting OCR-confusion characters
(O↔0, I↔1, S↔5…) → cross-frame majority vote → fuzzy-match against the whitelist
→ log GRANTED / DENIED. A single image uses the same logic in one pass via `/scan-image`.

**Biometric** — register: detect face → compute embedding → store photo + engine-tagged
vector. Verify: compute the query embedding → cosine-similarity vs stored encodings of
the *same* engine → grant if ≥ threshold (Facenet 0.72 / OpenCV 0.80).

**PPE & activity** — detect each person (YOLOv8n, tuned for high recall) → overlay
hard-hat/vest signal boxes on head/torso regions to decide compliance → classify
activity from pose + phone proximity → return an annotated frame, aggregate counts,
and per-person detail.

---

## Objectives

- **Automated access control** — check plates and faces against a database and
  grant/deny without manual verification.
- **Proactive safety** — instantly flag missing PPE and off-task workers.
- **Single pane of glass** — plates, faces, PPE, activity, and vehicles in one console,
  with editable authorized-vehicle and authorized-people lists and a full audit log.

---

## Current status

**Working:** ANPR (video + image), biometric register/verify (image + video), PPE &
activity detection, dynamic PPE-compliance gauge, per-camera tiles across
Operations / Gate / Safety / Vehicle, editable whitelists & personnel, pop-up alerts +
Alerts tab, local-time logs.

**Demo / placeholder (not yet real):** camera "streams" are synthetic MJPEG frames
(no live RTSP wired in — detection runs on uploaded media); the **restricted-zone
intrusion** and **fall / motionless** safety panels are static illustrative demos.

**Notes / limits:** PPE vest detection depends on the loaded HF model's classes
(hat-only unless swapped for a hat+vest model); person detection uses CPU YOLOv8n
(tuned for recall at `imgsz=960`, but heavy occlusion can still cause misses — a larger
model would improve this at the cost of speed).

---

## Licence

Internal project — Technomak. Not for public distribution.
