import sqlite3
from pathlib import Path

DB_PATH = Path("data/sentinel.db")


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=15, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS authorized_vehicles (
            plate        TEXT PRIMARY KEY,
            owner        TEXT NOT NULL DEFAULT 'Unknown',
            vehicle_type TEXT NOT NULL DEFAULT 'Car',
            created_at   TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS anpr_jobs (
            job_id     TEXT PRIMARY KEY,
            status     TEXT NOT NULL DEFAULT 'pending',
            progress   REAL NOT NULL DEFAULT 0,
            error      TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS anpr_plates (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id       TEXT NOT NULL,
            plate        TEXT NOT NULL,
            confidence   REAL NOT NULL DEFAULT 0,
            authorized   INTEGER NOT NULL DEFAULT 0,
            owner        TEXT NOT NULL DEFAULT '',
            vehicle_type TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS anpr_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT DEFAULT (datetime('now')),
            plate      TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0,
            authorized INTEGER NOT NULL DEFAULT 0,
            decision   TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS persons (
            employee_id    TEXT PRIMARY KEY,
            name           TEXT NOT NULL,
            department     TEXT NOT NULL DEFAULT 'General',
            clearance_level TEXT NOT NULL DEFAULT 'L1',
            photo_path     TEXT,
            created_at     TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS bio_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT DEFAULT (datetime('now')),
            person_name TEXT NOT NULL DEFAULT 'Unknown',
            confidence  REAL NOT NULL DEFAULT 0,
            decision    TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS vehicle_visits (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            plate            TEXT NOT NULL,
            owner            TEXT NOT NULL DEFAULT 'Unknown',
            entry_time       TEXT NOT NULL DEFAULT (datetime('now')),
            exit_time        TEXT,
            duration_minutes INTEGER
        );
        CREATE TABLE IF NOT EXISTS vehicle_demo_jobs (
            job_id      TEXT PRIMARY KEY,
            status      TEXT NOT NULL DEFAULT 'pending',
            progress    REAL NOT NULL DEFAULT 0,
            error       TEXT,
            result_json TEXT,
            created_at  TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS safety_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT DEFAULT (datetime('now')),
            kind       TEXT NOT NULL DEFAULT 'ppe',   -- 'ppe' | 'activity'
            detail     TEXT NOT NULL DEFAULT '',
            people     INTEGER NOT NULL DEFAULT 0,
            violations TEXT NOT NULL DEFAULT '',
            location   TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS allowed_materials (
            material   TEXT PRIMARY KEY,             -- normalised material name (lowercase)
            label      TEXT NOT NULL DEFAULT '',     -- display label
            note       TEXT NOT NULL DEFAULT '',
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS material_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT DEFAULT (datetime('now')),
            plate      TEXT NOT NULL DEFAULT '',     -- optional linked vehicle / DN
            detected   TEXT NOT NULL DEFAULT '',     -- comma-separated materials found
            disallowed TEXT NOT NULL DEFAULT '',     -- materials not on the allow-list
            decision   TEXT NOT NULL DEFAULT 'GRANTED',  -- 'GRANTED' | 'DENIED'
            location   TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS assets (
            asset_id   TEXT PRIMARY KEY,             -- company asset tag, e.g. GEN-042
            name       TEXT NOT NULL,                -- display name, e.g. Diesel Generator
            category   TEXT NOT NULL DEFAULT '',     -- detected asset category this maps to
            note       TEXT NOT NULL DEFAULT '',
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS asset_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp  TEXT DEFAULT (datetime('now')),
            plate      TEXT NOT NULL DEFAULT '',     -- optional linked truck / DN
            source     TEXT NOT NULL DEFAULT 'upload',  -- 'upload' | 'live' | 'cam'
            detected   TEXT NOT NULL DEFAULT '',     -- comma-separated categories detected
            matched    TEXT NOT NULL DEFAULT '',     -- comma-separated registered asset ids matched
            item_count INTEGER NOT NULL DEFAULT 0,
            location   TEXT NOT NULL DEFAULT ''
        );
    """)

    existing = conn.execute("SELECT COUNT(*) FROM authorized_vehicles").fetchone()[0]
    if existing == 0:
        seeds = [
            # Indian plates
            ("KL07CK4521", "Arun Menon",          "Car"),
            ("KA09AB1234", "Meridian Logistics",   "Truck"),
            ("TN32CD5678", "Coastal Freight",      "Van"),
            ("MH12EF9012", "Vector Supply",        "Truck"),
            ("KL07CK0001", "Technomak Security",   "Car"),
            ("AP28CZ1122", "Warehouse Ops",        "Van"),
            # Dubai plates
            ("A12345",  "Sheikh Mohammed",         "Car"),
            ("B1234",   "Dubai Police",            "Car"),
            ("CD123",   "Emirates Logistics",      "Truck"),
            ("X99999",  "Technomak UAE",           "Car"),
            ("AB5678",  "Al Futtaim Transport",    "Van"),
        ]
        for plate, owner, vtype in seeds:
            conn.execute(
                "INSERT OR IGNORE INTO authorized_vehicles (plate, owner, vehicle_type) VALUES (?,?,?)",
                (plate, owner, vtype),
            )

    # Seed a default allow-list of materials permitted into the yard.
    mat_existing = conn.execute("SELECT COUNT(*) FROM allowed_materials").fetchone()[0]
    if mat_existing == 0:
        material_seeds = [
            # material (normalised)   display label       note
            ("bottle",     "Bottle / Container",  "Sealed liquid containers"),
            ("box",        "Box / Carton",        "Packaged goods"),
            ("carton",     "Carton",              "Packaged goods"),
            ("pallet",     "Pallet",              "Wooden / plastic pallet"),
            ("barrel",     "Barrel / Drum",       "Industrial drum"),
            ("sack",       "Sack / Bag",          "Bagged material"),
            ("suitcase",   "Crate / Case",        "Hard case"),
            ("backpack",   "Bag",                 "Small bag"),
            ("book",       "Documents",           "Paperwork / manuals"),
        ]
        for material, label, note in material_seeds:
            conn.execute(
                "INSERT OR IGNORE INTO allowed_materials (material, label, note) VALUES (?,?,?)",
                (material, label, note),
            )

    # Seed a small register of company assets. The `category` is the *specific*
    # item the object detector recognises (its COCO class), so a detection maps
    # straight back to a known asset tag. These are classes a camera can trigger
    # easily for a demo (bed, laptop, suitcase, backpack, bottle…).
    asset_existing = conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
    if asset_existing == 0:
        asset_seeds = [
            # asset_id   name                  category (specific class)  note
            ("BED-900", "Bed / Mattress Unit", "bed",       "Freight bed / mattress"),
            ("LAP-100", "Field Laptop",        "laptop",    "Rugged inspection laptop"),
            ("CS-204",  "Tool Case",           "suitcase",  "Portable hard tool case"),
            ("BAG-055", "Technician Kit Bag",  "backpack",  "Field technician kit"),
            ("CNT-300", "Coolant Container",   "bottle",    "Sealed liquid container"),
        ]
        for asset_id, name, category, note in asset_seeds:
            conn.execute(
                "INSERT OR IGNORE INTO assets (asset_id, name, category, note) VALUES (?,?,?,?)",
                (asset_id, name, category, note),
            )

    # Migration: earlier builds seeded broad categories (electronics/case/…).
    # Asset tracking now matches on the specific detected class, so realign the
    # known seed rows — but only when they still hold the old value, so any
    # category an operator has since edited is left untouched. Also add the
    # 'bed' example if it is missing.
    _legacy_fix = [
        # asset_id   old_category   new_category
        ("LAP-100", "electronics", "laptop"),
        ("CS-204",  "case",        "suitcase"),
        ("BAG-055", "bag",         "backpack"),
        ("CNT-300", "container",   "bottle"),
    ]
    for asset_id, old, new in _legacy_fix:
        conn.execute(
            "UPDATE assets SET category=? WHERE asset_id=? AND category=?",
            (new, asset_id, old),
        )
    conn.execute(
        "INSERT OR IGNORE INTO assets (asset_id, name, category, note) VALUES (?,?,?,?)",
        ("BED-900", "Bed / Mattress Unit", "bed", "Freight bed / mattress"),
    )
    conn.commit()
    conn.close()
