"""
FastAPI application — REST + WebSocket + MJPEG streaming.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core import config, database as db
from core.gate_controller import get_gate
from core.pipeline import PipelineManager

log = logging.getLogger("yard_monitor.api")


# ----------------------------------------------------------------------------
# Real-time event bus
# ----------------------------------------------------------------------------

class EventBus:
    """In-process pub/sub for dashboard WebSocket clients."""

    def __init__(self):
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            self._clients.discard(ws)

    async def _broadcast(self, message: dict):
        dead = []
        async with self._lock:
            clients = list(self._clients)
        for ws in clients:
            try:
                await ws.send_text(json.dumps(message, default=str))
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._clients.discard(ws)

    def publish_threadsafe(self, message: dict):
        """Callable from non-async threads (the pipeline)."""
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(message), self._loop)


bus = EventBus()
manager: Optional[PipelineManager] = None


# ----------------------------------------------------------------------------
# Lifespan: spin up pipelines on startup
# ----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=config.get("system", "log_level", default="INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    db.init_db()
    bus.bind_loop(asyncio.get_running_loop())

    global manager
    manager = PipelineManager(
        on_event=lambda ev: bus.publish_threadsafe({"type": "event",         "data": ev}),
        on_update=lambda ev: bus.publish_threadsafe({"type": "event_updated", "data": ev}),
    )

    # Start pipelines in a background thread — model loading takes time
    def _bootstrap():
        try:
            manager.start()
            bus.publish_threadsafe({"type": "system", "data": {"msg": "Cameras online"}})
        except Exception as e:
            log.exception("Pipeline startup failed: %s", e)
            bus.publish_threadsafe({"type": "system", "data": {"msg": f"Startup failed: {e}"}})

    import threading
    threading.Thread(target=_bootstrap, daemon=True).start()

    yield

    if manager:
        manager.stop()


app = FastAPI(title="Yard Monitor", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"],
)


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------

class WhitelistAddRequest(BaseModel):
    plate: str
    owner_name: Optional[str] = None
    owner_phone: Optional[str] = None
    vehicle_type: Optional[str] = None
    company: Optional[str] = None
    notes: Optional[str] = None


class WhitelistUpdateRequest(BaseModel):
    owner_name: Optional[str] = None
    owner_phone: Optional[str] = None
    vehicle_type: Optional[str] = None
    company: Optional[str] = None
    notes: Optional[str] = None


class LoginRequest(BaseModel):
    password: str


class CameraSourceRequest(BaseModel):
    uri: int | str   # 0 = laptop cam, 1 = iVCam, or an RTSP/HTTP URL


# ----------------------------------------------------------------------------
# Very-light "auth" — single shared password
# ----------------------------------------------------------------------------

def _check_auth(request: Request):
    token = request.headers.get("x-admin-token") or request.query_params.get("token")
    expected = config.get("web", "admin_password", default="admin123")
    if token != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid or missing admin token")


# ----------------------------------------------------------------------------
# Static / pages
# ----------------------------------------------------------------------------

STATIC_DIR = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Serve plate / asset snapshots. Create the directories up front so FastAPI's
# StaticFiles mount doesn't blow up on a fresh checkout.
DATA_DIR = Path(__file__).parent.parent / "data"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
(SNAPSHOT_DIR / "assets").mkdir(parents=True, exist_ok=True)
app.mount("/snapshots", StaticFiles(directory=SNAPSHOT_DIR), name="snapshots")


@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(STATIC_DIR / "index.html")


# ----------------------------------------------------------------------------
# REST endpoints
# ----------------------------------------------------------------------------

@app.post("/api/login")
def login(req: LoginRequest):
    expected = config.get("web", "admin_password", default="admin123")
    if req.password != expected:
        raise HTTPException(401, "Wrong password")
    return {"token": expected}


@app.get("/api/stats")
def stats():
    return db.stats_today()


@app.get("/api/cameras")
def cameras():
    return manager.cameras() if manager else []


@app.get("/api/events")
def events(limit: int = 50):
    return db.recent_events(limit=limit)


@app.get("/api/assets")
def assets(present_only: bool = False):
    return db.list_assets(present_only=present_only)


@app.get("/api/whitelist")
def get_whitelist():
    return db.list_whitelist()


@app.post("/api/whitelist", dependencies=[Depends(_check_auth)])
def add_whitelist(req: WhitelistAddRequest):
    rec = db.add_to_whitelist(
        plate=req.plate,
        owner_name=req.owner_name,
        owner_phone=req.owner_phone,
        vehicle_type=req.vehicle_type,
        company=req.company,
        notes=req.notes,
    )
    db.audit("whitelist_add", actor="admin", plate=req.plate)
    bus.publish_threadsafe({"type": "whitelist_update", "data": rec})
    return rec


@app.put("/api/whitelist/{plate}", dependencies=[Depends(_check_auth)])
def update_whitelist(plate: str, req: WhitelistUpdateRequest):
    rec = db.update_whitelist(
        plate=plate,
        owner_name=req.owner_name,
        owner_phone=req.owner_phone,
        vehicle_type=req.vehicle_type,
        company=req.company,
        notes=req.notes,
    )
    if not rec:
        raise HTTPException(404, "Plate not found")
    db.audit("whitelist_edit", actor="admin", plate=plate)
    bus.publish_threadsafe({"type": "whitelist_update", "data": rec})
    return rec


@app.delete("/api/whitelist/{plate}", dependencies=[Depends(_check_auth)])
def del_whitelist(plate: str):
    ok = db.remove_from_whitelist(plate)
    if not ok:
        raise HTTPException(404, "Plate not found")
    db.audit("whitelist_remove", actor="admin", plate=plate)
    bus.publish_threadsafe({"type": "whitelist_update", "data": {"plate": plate, "removed": True}})
    return {"ok": True}


@app.post("/api/camera/{camera_id}/source", dependencies=[Depends(_check_auth)])
def switch_camera_source(camera_id: str, req: CameraSourceRequest):
    if manager is None:
        raise HTTPException(503, "Pipeline not ready")
    ok = manager.switch_camera_source(camera_id, req.uri)
    if not ok:
        raise HTTPException(404, "Camera not found")
    bus.publish_threadsafe({"type": "camera_source", "data": {"id": camera_id, "uri": req.uri}})
    return {"ok": True, "camera_id": camera_id, "uri": req.uri}


@app.delete("/api/events/{event_id}", dependencies=[Depends(_check_auth)])
def delete_event(event_id: int):
    ok = db.delete_event(event_id)
    if not ok:
        raise HTTPException(404, "Event not found")
    db.audit("event_delete", actor="admin", event_id=event_id)
    bus.publish_threadsafe({"type": "event_deleted", "data": {"id": event_id}})
    return {"ok": True}


@app.post("/api/gate/open", dependencies=[Depends(_check_auth)])
def gate_open():
    get_gate().manual_open(actor="admin")
    bus.publish_threadsafe({"type": "gate", "data": {"action": "manual_open"}})
    return {"ok": True, "is_open": True}


@app.post("/api/gate/close", dependencies=[Depends(_check_auth)])
def gate_close():
    get_gate().force_close(actor="admin")
    bus.publish_threadsafe({"type": "gate", "data": {"action": "manual_close"}})
    return {"ok": True, "is_open": False}


@app.get("/api/gate/status")
def gate_status():
    return {"is_open": get_gate().is_open}


# ----------------------------------------------------------------------------
# Demo mode — upload a video file and run it through the pipeline
# ----------------------------------------------------------------------------

DEMO_VIDEO = DATA_DIR / "demo" / "uploaded_demo.mp4"


@app.post("/api/demo/upload", dependencies=[Depends(_check_auth)])
async def demo_upload(file: UploadFile = File(...)):
    """Receive a video file and save it as the demo source."""
    DEMO_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    if not content:
        raise HTTPException(400, "Uploaded file is empty")
    with open(DEMO_VIDEO, "wb") as f:
        f.write(content)
    size_mb = round(len(content) / 1_048_576, 1)
    log.info("Demo video saved: %s (%.1f MB)", DEMO_VIDEO, size_mb)
    return {"ok": True, "filename": file.filename, "size_mb": size_mb}


@app.post("/api/demo/start", dependencies=[Depends(_check_auth)])
def demo_start():
    """Start (or restart) the demo pipeline with the uploaded video."""
    if manager is None:
        raise HTTPException(503, "Pipeline not ready — models still loading")
    if not DEMO_VIDEO.exists():
        raise HTTPException(404, "No demo video found. Upload one first via /api/demo/upload")
    cam_id = manager.start_demo(str(DEMO_VIDEO))
    bus.publish_threadsafe({
        "type": "cameras_updated",
        "data": {"cameras": manager.cameras()},
    })
    return {"ok": True, "camera_id": cam_id}


@app.delete("/api/demo/stop", dependencies=[Depends(_check_auth)])
def demo_stop():
    """Stop the demo pipeline."""
    if manager is None:
        raise HTTPException(503, "Pipeline not ready")
    ok = manager.stop_demo()
    bus.publish_threadsafe({
        "type": "cameras_updated",
        "data": {"cameras": manager.cameras()},
    })
    return {"ok": ok}


# ----------------------------------------------------------------------------
# MJPEG streaming
# ----------------------------------------------------------------------------

@app.get("/stream/{camera_id}")
async def mjpeg(camera_id: str):
    if manager is None or camera_id not in manager.pipelines:
        raise HTTPException(404, "Camera not found")

    async def gen():
        boundary = b"--frame"
        last_sent = None
        while True:
            jpeg = manager.get_jpeg(camera_id)
            # Only yield when there's a new frame to avoid blasting bytes for
            # nothing — also lets the event loop service other connections.
            if jpeg and jpeg is not last_sent:
                last_sent = jpeg
                yield (boundary + b"\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                       + jpeg + b"\r\n")
            await asyncio.sleep(0.08)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


# ----------------------------------------------------------------------------
# WebSocket — real-time updates
# ----------------------------------------------------------------------------

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await bus.connect(websocket)
    try:
        # Send initial snapshot so the dashboard renders fast
        await websocket.send_text(json.dumps({"type": "snapshot", "data": {
            "stats": db.stats_today(),
            "events": db.recent_events(limit=25),
            "assets": db.list_assets(present_only=True),
            "cameras": manager.cameras() if manager else [],
            "gate_open": get_gate().is_open,
        }}, default=str))
        # Keep alive
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                # Could handle ping or filter requests; for now just ignore.
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        pass
    finally:
        await bus.disconnect(websocket)
