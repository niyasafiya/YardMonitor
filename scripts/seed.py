"""
Seed the database with a handful of authorized vehicles so you have something
to demo right away. Run once:

    python scripts/seed.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `core` importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import database as db

SAMPLE_VEHICLES = [
    {"plate": "KL07BX1234", "owner_name": "Rahul Menon",   "owner_phone": "+91 98765 43210",
     "vehicle_type": "Car",   "company": "Acme Logistics"},
    {"plate": "TN09AB4567", "owner_name": "Priya Iyer",    "owner_phone": "+91 90000 11122",
     "vehicle_type": "Sedan", "company": "Acme Logistics"},
    {"plate": "KA01MJ7890", "owner_name": "Suresh Kumar",  "owner_phone": "+91 99887 76655",
     "vehicle_type": "Truck", "company": "Speedway Cargo"},
    {"plate": "DL3CAF9012", "owner_name": "Anita Sharma",  "owner_phone": "+91 91234 56789",
     "vehicle_type": "Truck", "company": "BlueDart"},
    {"plate": "MH12HX3456", "owner_name": "Vikram Patil",  "owner_phone": "+91 98765 12345",
     "vehicle_type": "Van",   "company": "Acme Logistics"},
]


def main():
    db.init_db()
    print("Seeding whitelist...")
    for v in SAMPLE_VEHICLES:
        db.add_to_whitelist(**v)
        print(f"  ✓ {v['plate']:12s}  {v['owner_name']}")
    print(f"\nDone. {len(SAMPLE_VEHICLES)} vehicles in whitelist.")


if __name__ == "__main__":
    main()
