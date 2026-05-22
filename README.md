# Yard Monitor — Real-time LPR, Gate Automation & Asset Tracking

A complete computer-vision yard management system. It watches one or more
camera feeds, **detects vehicles**, **reads their license plates** in real
time, decides whether each vehicle is authorized, and **opens the gate**
automatically. A second camera type monitors the yard interior to maintain
an inventory of **assets** (vehicles, containers, etc.) currently on-site.

Everything is wrapped in a live web dashboard so the operator can see what's
happening and manage the whitelist.

---

## What it does

| Capability | How it works |
|---|---|
| **Vehicle detection + tracking** | YOLOv8 with the built-in ByteTrack tracker — stable track IDs across frames |
| **License plate recognition (LPR)** | Plate detector (YOLO) → crop → EasyOCR with plate-specific preprocessing |
| **Gate automation** | Plate looked up in SQLite whitelist → if authorized, gate driver actuates (simulated / Raspberry Pi GPIO / HTTP relay) |
| **Direction inference** | Virtual "direction line" on the gate camera. Crossings top-to-bottom = entry, bottom-to-top = exit |
| **Asset tracking** | Yard cameras keep an inventory of assets currently present, with snapshots, last-seen timestamps, and last-known location |
| **Live dashboard** | FastAPI + WebSocket. KPIs, live MJPEG streams, recent events, whitelist CRUD, manual gate override |
| **Audit log** | Every gate open/close and whitelist change is logged |

---

## Architecture

```
            ┌────────────┐     ┌────────────┐     ┌────────────┐
RTSP / mp4 →│ CameraPipe │ →   │  Detector  │ →   │ PlateOCR   │
            │ (1 thread  │     │  (YOLOv8)  │     │ (EasyOCR)  │
            │  per cam)  │     └────────────┘     └─────┬──────┘
            └──────┬─────┘                              │
                   │ events                             ▼
                   ▼                              GateController
            ┌────────────┐                       (sim | gpio | http)
            │  SQLite    │
            │ (events,   │←──── REST /api/* ────┐
            │  whitelist,│                       │
            │  assets)   │                       │
            └─────┬──────┘                       │
                  │                              │
                  ▼                              │
            WebSocket /ws   ────────────► Dashboard (HTML/JS)
            MJPEG /stream/{id}
```

---

## Quick start (5 minutes)

```bash
# 1. Clone & enter project
cd yard-monitor

# 2. Create venv and install
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Download YOLO models
python scripts/download_models.py

# 4. Seed the demo whitelist
python scripts/seed.py

# 5. (Optional) Make a synthetic gate video for first-launch sanity-check
python scripts/make_sample_video.py

# 6. Launch
python main.py
```

Then open **http://localhost:8000** and log in with `admin123`
(change this in `config.yaml`).

> **For a real demo**: replace `data/sample_gate.mp4` and `data/sample_yard.mp4`
> with a recording of vehicles passing your gate. The synthetic video only
> confirms plumbing works; YOLO can't detect colour rectangles as vehicles.

---

## Hooking up real cameras

Edit `config.yaml`:

```yaml
sources:
  - id: "gate_cam"
    name: "Main Gate"
    role: "gate"
    uri: "rtsp://admin:password@192.168.1.10:554/stream1"
    fps_target: 8
    direction_line_y: 0.55       # adjust until the line sits where vehicles cross

  - id: "yard_cam"
    name: "Yard Overview"
    role: "yard"
    uri: "rtsp://admin:password@192.168.1.11:554/stream1"
    fps_target: 4
```

Supported URIs:
- `0`, `1`, ... → USB webcam
- `rtsp://...` → IP camera (most CCTV NVRs export RTSP)
- `/path/to/file.mp4` → recorded video (loops)
- `http://...` → MJPEG over HTTP

### Hooking up a real gate

In `config.yaml`, change the `gate.driver`:

- `simulated` — for demos. The dashboard shows OPEN/CLOSED.
- `gpio` — Raspberry Pi GPIO relay. Wire pin 17 (BCM) to your relay control input. Install `RPi.GPIO` on the Pi.
- `http` — any HTTP-controllable gate. Set the `open_url` and `close_url` it should POST to.

---

## Project layout

```
yard-monitor/
├── main.py                  # entrypoint
├── config.yaml              # all settings
├── requirements.txt
├── Dockerfile
├── core/
│   ├── config.py            # YAML loader
│   ├── database.py          # SQLAlchemy models + helpers
│   ├── detector.py          # YOLOv8 wrapper (vehicle + plate)
│   ├── ocr.py               # EasyOCR + plate preprocessing
│   ├── gate_controller.py   # decision + driver (sim/gpio/http)
│   ├── asset_tracker.py     # yard inventory
│   └── pipeline.py          # per-camera processing thread
├── api/
│   └── main.py              # FastAPI app, REST, WebSocket, MJPEG
├── static/
│   ├── index.html           # dashboard
│   ├── app.js               # WebSocket + UI logic
│   └── style.css
├── scripts/
│   ├── seed.py              # populate demo whitelist
│   ├── download_models.py   # fetch YOLO weights
│   └── make_sample_video.py # synthetic demo clip
└── tests/
    └── test_basic.py        # pytest sanity tests
```

---

## Configuration reference

All knobs live in `config.yaml`. Most important ones:

| Path | Purpose |
|---|---|
| `sources[].uri` | Camera URI |
| `sources[].fps_target` | Frames/sec to process (lower = less CPU) |
| `sources[].direction_line_y` | Fraction of frame height for the entry/exit line |
| `models.vehicle_detector` | YOLO model for vehicles (default `yolov8n.pt`) |
| `models.plate_detector` | YOLO model for plates (download or train your own) |
| `thresholds.vehicle_conf` | Min confidence for a vehicle detection |
| `thresholds.plate_conf` | Min confidence for a plate detection |
| `thresholds.ocr_conf` | Min confidence for an OCR result |
| `thresholds.duplicate_window_sec` | Debounce same plate seen again |
| `gate.driver` | `simulated` / `gpio` / `http` |
| `gate.open_duration_sec` | How long the gate stays open |
| `gate.strict_whitelist` | If true, only whitelisted plates auto-open the gate |

---

## API reference

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/api/login` | — | Returns admin token |
| GET | `/api/stats` | — | KPI counters for today |
| GET | `/api/cameras` | — | List of configured cameras |
| GET | `/api/events?limit=50` | — | Recent vehicle events |
| GET | `/api/assets?present_only=true` | — | Asset inventory |
| GET | `/api/whitelist` | — | List authorized vehicles |
| POST | `/api/whitelist` | ✓ | Add / upsert a vehicle |
| DELETE | `/api/whitelist/{plate}` | ✓ | Remove a vehicle |
| GET | `/api/gate/status` | — | Current gate state |
| POST | `/api/gate/open` | ✓ | Manual open |
| POST | `/api/gate/close` | ✓ | Manual close |
| GET | `/stream/{camera_id}` | — | MJPEG live preview |
| WS | `/ws` | — | Real-time events |

Authenticated requests must include the header `X-Admin-Token: <password>`.

---

## Docker

```bash
docker build -t yard-monitor .
docker run --rm -p 8000:8000 \
  -v $PWD/data:/app/data \
  -v $PWD/config.yaml:/app/config.yaml \
  yard-monitor
```

For Raspberry Pi GPIO support, run with `--device /dev/gpiomem` and use a Pi-compatible base image.

---

## Performance notes

- On a modest laptop (i5, no GPU), YOLOv8n + EasyOCR at 4–8 FPS per camera is comfortable.
- Set `YM_USE_GPU=1` to enable GPU OCR (requires CUDA-build of PyTorch).
- Lower `fps_target` if CPU is pegged. Vehicle gate footage doesn't need 30 FPS — 6–10 is plenty.
- For more accurate plate detection on Indian-format plates, retrain `license_plate.pt` on a labelled dataset (Roboflow Universe has free ones).

---

## Tests

```bash
pytest -v
```

---

## Roadmap / nice extensions for extra marks

- [ ] Per-camera ROI masks to ignore irrelevant zones
- [ ] Email / Telegram alerts on denied entries
- [ ] Day-wise / week-wise charts in the dashboard
- [ ] Plate de-skew / homography for sharply-angled cameras
- [ ] Multi-user accounts with roles (operator / admin / auditor)
- [ ] On-prem export of events as CSV / PDF report

---

## License

MIT — do whatever you need to deliver this to your client.
