"""
Yard Monitor entry point.

Usage:
    python main.py                # start dashboard + pipelines
    python main.py --headless     # only run pipelines (no API server)
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

from core import config, database as db


def run_headless() -> None:
    from core.pipeline import PipelineManager

    db.init_db()
    pm = PipelineManager()
    pm.start()

    stop = False

    def _handle(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    print("Running in headless mode. Ctrl-C to stop.")
    while not stop:
        time.sleep(0.5)

    pm.stop()


def run_server() -> None:
    import uvicorn

    host = config.get("web", "host", default="0.0.0.0")
    port = int(config.get("web", "port", default=8000))

    uvicorn.run("api.main:app", host=host, port=port,
                reload=False, log_level="info")


def main():
    parser = argparse.ArgumentParser(description="Yard Monitor")
    parser.add_argument("--headless", action="store_true",
                        help="Run only pipelines without web UI")
    parser.add_argument("--config", default=None,
                        help="Path to alternate config.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=config.get("system", "log_level", default="INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.config:
        config.load_config(args.config)

    db.init_db()

    if args.headless:
        run_headless()
    else:
        run_server()


if __name__ == "__main__":
    main()
