"""
Minimal sanity tests. Run with:  pytest -v
"""
import os
import sys
from pathlib import Path

# Make project root importable when running pytest from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("YM_CONFIG", str(Path(__file__).resolve().parent.parent / "config.yaml"))


def test_config_loads():
    from core import config
    cfg = config.load_config()
    assert "system" in cfg
    assert "sources" in cfg
    assert isinstance(cfg["sources"], list)


def test_db_init_and_whitelist():
    # Use an isolated test DB
    import tempfile
    from core import database as db
    tmpdir = tempfile.mkdtemp()
    db_file = Path(tmpdir) / "test.db"
    # Override engine
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    db._engine = create_engine(f"sqlite:///{db_file}", future=True,
                               connect_args={"check_same_thread": False})
    db._SessionLocal = sessionmaker(bind=db._engine, expire_on_commit=False, autoflush=False)
    db.Base.metadata.create_all(db._engine)

    rec = db.add_to_whitelist(plate="ABC123", owner_name="Test", vehicle_type="Car")
    assert rec["plate"] == "ABC123"
    assert db.lookup_plate("ABC123") is not None
    assert db.lookup_plate("XYZ999") is None

    ok = db.remove_from_whitelist("ABC123")
    assert ok is True
    assert db.lookup_plate("ABC123") is None


def test_gate_decision_strict():
    import tempfile
    from core import database as db, gate_controller
    tmpdir = tempfile.mkdtemp()
    db_file = Path(tmpdir) / "test.db"
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    db._engine = create_engine(f"sqlite:///{db_file}", future=True,
                               connect_args={"check_same_thread": False})
    db._SessionLocal = sessionmaker(bind=db._engine, expire_on_commit=False, autoflush=False)
    db.Base.metadata.create_all(db._engine)

    db.add_to_whitelist(plate="GOOD123", owner_name="Owner")
    gate = gate_controller.GateController()
    assert gate.decide("GOOD123").will_open is True
    assert gate.decide("BAD999").will_open is False
    assert gate.decide(None).will_open is False
