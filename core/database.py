"""
Database layer for Yard Monitor.

Tables
------
authorized_vehicle  whitelist of plates allowed to enter
vehicle_event       every entry/exit + authorization decision
asset               persistent assets being tracked in the yard
asset_snapshot      time-series presence of an asset
audit_log           free-form audit trail (gate opens, manual overrides, ...)
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

from . import config


class Base(DeclarativeBase):
    pass


class AuthorizedVehicle(Base):
    __tablename__ = "authorized_vehicle"

    id = Column(Integer, primary_key=True)
    plate = Column(String(32), unique=True, nullable=False, index=True)
    owner_name = Column(String(128))
    owner_phone = Column(String(32))
    vehicle_type = Column(String(32))     # e.g. "Truck", "Sedan"
    company = Column(String(128))
    notes = Column(Text)
    active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "plate": self.plate,
            "owner_name": self.owner_name,
            "owner_phone": self.owner_phone,
            "vehicle_type": self.vehicle_type,
            "company": self.company,
            "notes": self.notes,
            "active": self.active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class VehicleEvent(Base):
    __tablename__ = "vehicle_event"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    camera_id = Column(String(64), nullable=False)
    plate = Column(String(32), index=True)
    plate_confidence = Column(Float)
    direction = Column(String(8))           # "entry" | "exit" | "unknown"
    authorized = Column(Boolean, default=False, nullable=False)
    gate_opened = Column(Boolean, default=False, nullable=False)
    snapshot_path = Column(String(255))     # cropped plate / full frame
    vehicle_type = Column(String(32))
    track_id = Column(Integer)
    notes = Column(Text)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "camera_id": self.camera_id,
            "plate": self.plate,
            "plate_confidence": self.plate_confidence,
            "direction": self.direction,
            "authorized": self.authorized,
            "gate_opened": self.gate_opened,
            "snapshot_path": self.snapshot_path,
            "vehicle_type": self.vehicle_type,
            "track_id": self.track_id,
            "notes": self.notes,
        }


class Asset(Base):
    __tablename__ = "asset"

    id = Column(Integer, primary_key=True)
    asset_code = Column(String(64), unique=True, nullable=False)   # human label
    asset_type = Column(String(32))                                # car / truck / container ...
    plate = Column(String(32))                                     # nullable
    description = Column(Text)
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)
    last_camera = Column(String(64))
    last_bbox = Column(String(64))                                 # "x1,y1,x2,y2"
    present = Column(Boolean, default=True, nullable=False)

    snapshots = relationship("AssetSnapshot", back_populates="asset",
                             cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "asset_code": self.asset_code,
            "asset_type": self.asset_type,
            "plate": self.plate,
            "description": self.description,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "last_camera": self.last_camera,
            "last_bbox": self.last_bbox,
            "present": self.present,
        }


class AssetSnapshot(Base):
    __tablename__ = "asset_snapshot"

    id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey("asset.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    camera_id = Column(String(64))
    bbox = Column(String(64))
    image_path = Column(String(255))

    asset = relationship("Asset", back_populates="snapshots")


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    actor = Column(String(64))                # "system" | "admin" | user name
    action = Column(String(64))               # "gate_open", "whitelist_add", ...
    payload = Column(Text)                    # JSON-encoded extras

    def to_dict(self) -> dict:
        data = {}
        if self.payload:
            try:
                data = json.loads(self.payload)
            except Exception:
                data = {"raw": self.payload}
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "actor": self.actor,
            "action": self.action,
            "data": data,
        }


# ----------------------------------------------------------------------------
# Engine / session management
# ----------------------------------------------------------------------------

_engine = None
_SessionLocal = None


def _make_engine():
    db_url = config.get("storage", "db_url", default="sqlite:///data/yard_monitor.db")
    # Ensure parent dir exists for sqlite
    if db_url.startswith("sqlite:///"):
        Path(db_url.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False so the FastAPI threadpool can share connections
    return create_engine(
        db_url,
        future=True,
        connect_args={"check_same_thread": False} if db_url.startswith("sqlite") else {},
    )


def init_db() -> None:
    """Create tables if they don't exist."""
    global _engine, _SessionLocal
    if _engine is None:
        _engine = _make_engine()
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, autoflush=False)
    Base.metadata.create_all(_engine)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session context."""
    if _SessionLocal is None:
        init_db()
    s = _SessionLocal()  # type: ignore[misc]
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


# ----------------------------------------------------------------------------
# Convenience helpers
# ----------------------------------------------------------------------------

def lookup_plate(plate: str) -> Optional[dict]:
    """Return whitelist record for a plate, or None."""
    if not plate:
        return None
    with session_scope() as s:
        v = s.scalar(
            select(AuthorizedVehicle).where(
                AuthorizedVehicle.plate == plate.upper(),
                AuthorizedVehicle.active.is_(True),
            )
        )
        return v.to_dict() if v else None


def add_event(**kwargs) -> dict:
    with session_scope() as s:
        ev = VehicleEvent(**kwargs)
        s.add(ev)
        s.flush()
        return ev.to_dict()


def recent_events(limit: int = 50) -> list[dict]:
    with session_scope() as s:
        rows = s.scalars(
            select(VehicleEvent).order_by(VehicleEvent.timestamp.desc()).limit(limit)
        ).all()
        return [r.to_dict() for r in rows]


def recent_event_for_plate(plate: str, within_sec: int) -> Optional[dict]:
    """Used to debounce duplicate detections of the same plate."""
    if not plate:
        return None
    cutoff = datetime.utcnow() - timedelta(seconds=within_sec)
    with session_scope() as s:
        ev = s.scalar(
            select(VehicleEvent)
            .where(VehicleEvent.plate == plate.upper(),
                   VehicleEvent.timestamp >= cutoff)
            .order_by(VehicleEvent.timestamp.desc())
        )
        return ev.to_dict() if ev else None


def upsert_asset(asset_code: str, **fields) -> dict:
    with session_scope() as s:
        a = s.scalar(select(Asset).where(Asset.asset_code == asset_code))
        if a is None:
            a = Asset(asset_code=asset_code, **fields)
            s.add(a)
        else:
            for k, v in fields.items():
                if v is not None:
                    setattr(a, k, v)
            a.last_seen = datetime.utcnow()
            a.present = True
        s.flush()
        return a.to_dict()


def mark_assets_absent(camera_id: str, seen_codes: Iterable[str], grace_sec: int = 60):
    """Anything not seen for `grace_sec` on this camera is marked not present."""
    seen = set(seen_codes)
    cutoff = datetime.utcnow() - timedelta(seconds=grace_sec)
    with session_scope() as s:
        rows = s.scalars(
            select(Asset).where(Asset.last_camera == camera_id, Asset.present.is_(True))
        ).all()
        for a in rows:
            if a.asset_code not in seen and a.last_seen < cutoff:
                a.present = False


def list_assets(present_only: bool = False) -> list[dict]:
    with session_scope() as s:
        stmt = select(Asset).order_by(Asset.last_seen.desc())
        if present_only:
            stmt = stmt.where(Asset.present.is_(True))
        return [a.to_dict() for a in s.scalars(stmt).all()]


def list_whitelist() -> list[dict]:
    with session_scope() as s:
        rows = s.scalars(select(AuthorizedVehicle).order_by(AuthorizedVehicle.plate)).all()
        return [r.to_dict() for r in rows]


def add_to_whitelist(plate: str, **fields) -> dict:
    plate = plate.strip().upper()
    with session_scope() as s:
        existing = s.scalar(select(AuthorizedVehicle).where(AuthorizedVehicle.plate == plate))
        if existing:
            for k, v in fields.items():
                if v is not None:
                    setattr(existing, k, v)
            existing.active = True
            s.flush()
            return existing.to_dict()
        v = AuthorizedVehicle(plate=plate, **fields)
        s.add(v)
        s.flush()
        return v.to_dict()


def remove_from_whitelist(plate: str) -> bool:
    plate = plate.strip().upper()
    with session_scope() as s:
        v = s.scalar(select(AuthorizedVehicle).where(AuthorizedVehicle.plate == plate))
        if not v:
            return False
        v.active = False
        return True


def audit(action: str, actor: str = "system", **payload):
    with session_scope() as s:
        s.add(AuditLog(action=action, actor=actor, payload=json.dumps(payload, default=str)))


def delete_event(event_id: int) -> bool:
    with session_scope() as s:
        ev = s.scalar(select(VehicleEvent).where(VehicleEvent.id == event_id))
        if not ev:
            return False
        s.delete(ev)
        return True


def stats_today() -> dict:
    """Return basic counters used by the dashboard header."""
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    with session_scope() as s:
        total = s.scalar(select(func.count(VehicleEvent.id)).where(VehicleEvent.timestamp >= today_start)) or 0
        entries = s.scalar(select(func.count(VehicleEvent.id)).where(
            VehicleEvent.timestamp >= today_start,
            VehicleEvent.direction == "entry",
        )) or 0
        exits = s.scalar(select(func.count(VehicleEvent.id)).where(
            VehicleEvent.timestamp >= today_start,
            VehicleEvent.direction == "exit",
        )) or 0
        denied = s.scalar(select(func.count(VehicleEvent.id)).where(
            VehicleEvent.timestamp >= today_start,
            VehicleEvent.authorized.is_(False),
        )) or 0
        whitelist_size = s.scalar(select(func.count(AuthorizedVehicle.id)).where(
            AuthorizedVehicle.active.is_(True))) or 0
        assets_present = s.scalar(select(func.count(Asset.id)).where(Asset.present.is_(True))) or 0
        return {
            "total_today": total,
            "entries_today": entries,
            "exits_today": exits,
            "denied_today": denied,
            "whitelist_size": whitelist_size,
            "assets_present": assets_present,
        }
