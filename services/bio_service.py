"""
Face encoding and verification.

Primary engine: **DeepFace with an ArcFace embedding** — a state-of-the-art face
model that stays discriminative across glasses on/off, lighting, expression, and
moderate pose changes. Faces are located with the strongest detector available
(RetinaFace → YOLOv8 → SSD → OpenCV, whichever imports successfully).

Robustness comes from two things:
  1. ArcFace embeddings (512-d), which separate identities well even with glasses.
  2. A **gallery** of many embeddings per person, captured from the enrolment
     video (different frames / angles / with-and-without glasses). Verification
     takes the BEST (max) cosine similarity across the whole gallery, so a single
     good frame is enough to recognise the person.

When DeepFace/TensorFlow is unavailable we fall back to a lightweight OpenCV
face-crop embedding so the feature still works end-to-end. Encodings are
engine-tagged; a query is only compared against stored encodings from the *same*
engine. Photos live in data/faces/, encodings in data/encodings.json.
"""
from __future__ import annotations

import json
import os
import tempfile
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional

FACES_DIR      = Path("data/faces")
ENCODINGS_FILE = Path("data/encodings.json")

# DeepFace embedding model + detector preference (tried in order; missing ones skipped).
_FACE_MODEL         = "ArcFace"
_DETECTOR_BACKENDS  = ["retinaface", "yolov8", "ssd", "opencv"]
_ENGINE_TAG         = _FACE_MODEL.lower()          # "arcface"
_MAX_GALLERY        = 24                            # embeddings kept per person (more = more robust)
_DEDUP_SIM          = 0.94                          # keep more DIVERSE views (glasses/makeup/pose)

# Cosine-similarity match thresholds, per engine.
THRESHOLDS = {
    "arcface":    0.38,   # ArcFace 512-d — a touch lenient so glasses/makeup still match
    "facenet512": 0.70,
    "facenet":    0.60,   # legacy DeepFace/Facenet 128-d
    "opencv":     0.80,   # raw face-crop fallback
}
DEFAULT_THRESHOLD = 0.45

# Cached lazy state
_deepface = None            # the DeepFace module (or False if unavailable)
_working_backend: Optional[str] = None


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def save_photo(employee_id: str, image_bytes: bytes) -> str:
    FACES_DIR.mkdir(parents=True, exist_ok=True)
    path = FACES_DIR / f"{employee_id}.jpg"
    path.write_bytes(image_bytes)
    return str(path)


def _load_encodings() -> Dict[str, dict]:
    if ENCODINGS_FILE.exists():
        try:
            return json.loads(ENCODINGS_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_encodings(data: Dict[str, dict]):
    ENCODINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    ENCODINGS_FILE.write_text(json.dumps(data))


def _normalize_enc(enc) -> Optional[dict]:
    """Accept every historical form and return {engine, vecs:[...]}:
      * new gallery:  {"engine", "vecs": [[...], ...]}
      * single vec:   {"engine", "vec": [...]}
      * legacy list:  [...]  (was always Facenet)
    """
    if isinstance(enc, dict):
        if isinstance(enc.get("vecs"), list):
            return {"engine": enc.get("engine", _ENGINE_TAG), "vecs": enc["vecs"]}
        if isinstance(enc.get("vec"), list):
            return {"engine": enc.get("engine", "facenet"), "vecs": [enc["vec"]]}
    if isinstance(enc, list):
        return {"engine": "facenet", "vecs": [enc]}
    return None


def save_encoding(employee_id: str, embedding: dict):
    """Store an engine-tagged gallery. Accepts {engine, vecs:[...]} or {engine, vec}."""
    vecs = embedding.get("vecs")
    if vecs is None and "vec" in embedding:
        vecs = [embedding["vec"]]
    enc = _load_encodings()
    enc[employee_id] = {"engine": embedding.get("engine", _ENGINE_TAG), "vecs": vecs or []}
    _save_encodings(enc)


def delete_encoding(employee_id: str):
    enc = _load_encodings()
    enc.pop(employee_id, None)
    _save_encodings(enc)


# ---------------------------------------------------------------------------
# Embedding engines
# ---------------------------------------------------------------------------

def _get_deepface():
    global _deepface
    if _deepface is None:
        try:
            from deepface import DeepFace
            _deepface = DeepFace
        except Exception as exc:
            print(f"[bio] DeepFace unavailable, using OpenCV fallback: {exc}")
            _deepface = False
    return _deepface


def _arcface_embedding(image_bytes: bytes) -> Optional[List[float]]:
    """ArcFace embedding via DeepFace using the best available detector.
    Returns the vector, or None if no face is found / DeepFace is unavailable."""
    global _working_backend
    DeepFace = _get_deepface()
    if not DeepFace:
        return None

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(image_bytes)
            tmp_path = f.name

        # Try the last-known-good backend first, then the rest.
        order = ([_working_backend] if _working_backend else []) + \
                [b for b in _DETECTOR_BACKENDS if b != _working_backend]
        for backend in order:
            try:
                result = DeepFace.represent(
                    img_path=tmp_path,
                    model_name=_FACE_MODEL,
                    enforce_detection=True,     # only accept a real, detected face
                    detector_backend=backend,
                    align=True,
                )
                if result:
                    _working_backend = backend
                    return result[0]["embedding"]
            except Exception:
                continue  # backend missing or no face with this detector — try next
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return None


_CASCADE = None


def _get_cascade():
    global _CASCADE
    if _CASCADE is None:
        _CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _CASCADE


def _opencv_embedding(image_bytes: bytes) -> Optional[dict]:
    """Lightweight, TensorFlow-free face embedding: detect the largest face,
    crop, normalise to a fixed-size equalised grayscale vector. Deterministic."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_found = False
    try:
        faces = _get_cascade().detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
    except Exception:
        faces = []

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad = int(0.12 * w)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(gray.shape[1], x + w + pad), min(gray.shape[0], y + h + pad)
        crop = gray[y0:y1, x0:x1]
        face_found = True
    else:
        crop = gray  # whole image — still lets identical photos match

    if crop.size == 0:
        return None

    crop = cv2.resize(crop, (120, 120))
    crop = cv2.equalizeHist(crop)
    v = crop.astype(np.float32).flatten()
    v -= v.mean()
    norm = float(np.linalg.norm(v))
    if norm < 1e-6:
        return None
    return {"engine": "opencv", "vec": (v / norm).tolist(), "face": face_found}


def enhance_bytes(image_bytes: bytes) -> Optional[bytes]:
    """Clean up an unclear / low-light / small frame so the face is easier to
    detect and embed: upscale tiny frames, apply CLAHE (adaptive contrast) on the
    luminance channel, and a mild unsharp mask. Returns enhanced JPEG bytes."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) < 640:                                    # upscale small/blurry frames
        s = 640.0 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        blur = cv2.GaussianBlur(img, (0, 0), 2)            # mild sharpen
        img = cv2.addWeighted(img, 1.4, blur, -0.4, 0)
    except Exception:
        pass
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes() if ok else None


def compute_embedding(image_bytes: bytes) -> Optional[dict]:
    """Return an engine-tagged embedding dict {engine, vec, face} or None.

    Tries ArcFace on the raw frame; if no face is found (unclear / low-light /
    small face), retries once on an enhanced copy. When DeepFace is unavailable,
    falls back to the OpenCV crop embedding. Returns None when DeepFace works but
    no face is detectable (so the caller can skip that frame)."""
    if _get_deepface():
        vec = _arcface_embedding(image_bytes)
        if vec is None:
            enhanced = enhance_bytes(image_bytes)          # second try on hard frames
            if enhanced is not None:
                vec = _arcface_embedding(enhanced)
        if vec is not None:
            return {"engine": _ENGINE_TAG, "vec": vec, "face": True}
        return None  # DeepFace works but no face in this frame → skip
    return _opencv_embedding(image_bytes)


def _cosine_sim(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return 0.0
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 1e-9 else 0.0


def add_to_gallery(gallery: dict, emb: dict) -> None:
    """Append an embedding to an in-progress gallery dict {engine, vecs}, skipping
    near-duplicates so the stored set stays diverse. Mutates `gallery`."""
    if not emb or "vec" not in emb:
        return
    gallery.setdefault("engine", emb["engine"])
    gallery.setdefault("vecs", [])
    if gallery["engine"] != emb["engine"]:
        return
    if len(gallery["vecs"]) >= _MAX_GALLERY:
        return
    v = emb["vec"]
    if all(_cosine_sim(v, e) < _DEDUP_SIM for e in gallery["vecs"]):
        gallery["vecs"].append(v)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_face(image_bytes: bytes, persons: List[Dict]) -> Dict:
    query = compute_embedding(image_bytes)
    if query is None:
        return {"matched": False, "confidence": 0.0, "person": None,
                "engine": "none", "face": False}

    encodings   = _load_encodings()
    best_sim    = 0.0
    best_person = None

    for p in persons:
        enc = _normalize_enc(encodings.get(p["employee_id"]))
        if enc is None or enc["engine"] != query["engine"]:
            continue  # only compare like-with-like engines
        # Best match across the person's whole gallery
        for vec in enc["vecs"]:
            sim = _cosine_sim(query["vec"], vec)
            if sim > best_sim:
                best_sim, best_person = sim, p

    threshold    = THRESHOLDS.get(query["engine"], DEFAULT_THRESHOLD)
    matched      = best_sim >= threshold and best_person is not None
    engine_label = {"arcface": "DeepFace/ArcFace",
                    "facenet": "DeepFace/Facenet"}.get(query["engine"],
                                                       "OpenCV/face-crop")

    return {
        "matched":    matched,
        "confidence": round(best_sim, 4),
        "person":     best_person if matched else None,
        "engine":     engine_label,
        "face":       bool(query.get("face")),
    }
