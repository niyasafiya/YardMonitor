"""
Gate controller — decides whether to open the gate and actuates it.

Three drivers are provided:

* `simulated` — just logs, ideal for demos and the dashboard
* `gpio`      — Raspberry Pi GPIO relay (only loaded if RPi.GPIO is installed)
* `http`      — POSTs to a relay/HTTP-controlled gate device

The decision logic is the same regardless of driver:

1. Look up the plate in the whitelist.
2. If found AND strict_whitelist OR not strict: open the gate.
3. If not found AND not strict: open & log as "guest".
4. If not found AND strict: deny.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

from . import config, database as db

log = logging.getLogger(__name__)


@dataclass
class GateDecision:
    plate: Optional[str]
    authorized: bool
    will_open: bool
    reason: str
    owner: Optional[dict] = None


class _BaseDriver:
    name = "base"
    def open(self) -> None: ...
    def close(self) -> None: ...


class SimulatedDriver(_BaseDriver):
    name = "simulated"

    def __init__(self):
        self._open = False

    def open(self) -> None:
        self._open = True
        log.info("[GATE-SIM] OPEN")

    def close(self) -> None:
        self._open = False
        log.info("[GATE-SIM] CLOSE")


class GPIODriver(_BaseDriver):
    name = "gpio"

    def __init__(self, pin: int = 17):
        import RPi.GPIO as GPIO  # type: ignore  # available only on Pi
        self.GPIO = GPIO
        self.pin = pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)

    def open(self) -> None:
        self.GPIO.output(self.pin, self.GPIO.HIGH)
        log.info("[GATE-GPIO] HIGH pin %s", self.pin)

    def close(self) -> None:
        self.GPIO.output(self.pin, self.GPIO.LOW)
        log.info("[GATE-GPIO] LOW pin %s", self.pin)


class HTTPDriver(_BaseDriver):
    name = "http"

    def __init__(self, open_url: str, close_url: str):
        import urllib.request
        self._open_url = open_url
        self._close_url = close_url
        self._req = urllib.request

    def open(self) -> None:
        try:
            self._req.urlopen(self._open_url, timeout=3)
            log.info("[GATE-HTTP] open OK")
        except Exception as e:
            log.error("[GATE-HTTP] open failed: %s", e)

    def close(self) -> None:
        try:
            self._req.urlopen(self._close_url, timeout=3)
            log.info("[GATE-HTTP] close OK")
        except Exception as e:
            log.error("[GATE-HTTP] close failed: %s", e)


def _make_driver() -> _BaseDriver:
    kind = config.get("gate", "driver", default="simulated")
    if kind == "gpio":
        try:
            return GPIODriver()
        except Exception as e:
            log.warning("Falling back to simulated gate — GPIO error: %s", e)
            return SimulatedDriver()
    if kind == "http":
        return HTTPDriver(
            open_url=config.get("gate", "http", "open_url"),
            close_url=config.get("gate", "http", "close_url"),
        )
    return SimulatedDriver()


class GateController:
    """Stateful gate logic — single global instance is fine."""

    def __init__(self):
        self.driver = _make_driver()
        self.strict = bool(config.get("gate", "strict_whitelist", default=True))
        self.open_sec = int(config.get("gate", "open_duration_sec", default=6))
        self._lock = threading.Lock()
        self._is_open = False
        self._auto_close_thread: Optional[threading.Thread] = None
        log.info("Gate controller ready — driver=%s, strict=%s",
                 self.driver.name, self.strict)

    # ------------------------------------------------------------------

    def decide(self, plate: Optional[str]) -> GateDecision:
        """Authorize based on whitelist, but DON'T actuate yet."""
        if not plate:
            return GateDecision(
                plate=None, authorized=False,
                will_open=not self.strict and False,   # never open on unknown plate
                reason="no_plate_detected",
            )

        owner = db.lookup_plate(plate)
        if owner:
            return GateDecision(plate=plate, authorized=True, will_open=True,
                                reason="whitelisted", owner=owner)
        if self.strict:
            return GateDecision(plate=plate, authorized=False, will_open=False,
                                reason="not_whitelisted")
        return GateDecision(plate=plate, authorized=False, will_open=True,
                            reason="guest_open_mode")

    def actuate(self, decision: GateDecision, manual: bool = False) -> bool:
        """Actually open the gate (and schedule a close)."""
        if not decision.will_open and not manual:
            return False
        with self._lock:
            if self._is_open:
                log.debug("Gate already open — extending hold")
            else:
                self.driver.open()
                self._is_open = True
                db.audit("gate_open",
                         actor="admin" if manual else "system",
                         plate=decision.plate, reason=decision.reason)
            self._schedule_close()
        return True

    def manual_open(self, actor: str = "admin") -> None:
        with self._lock:
            self.driver.open()
            self._is_open = True
            db.audit("gate_open_manual", actor=actor)
            self._schedule_close()

    def force_close(self, actor: str = "admin") -> None:
        with self._lock:
            self.driver.close()
            self._is_open = False
            db.audit("gate_close_manual", actor=actor)

    @property
    def is_open(self) -> bool:
        return self._is_open

    # ------------------------------------------------------------------

    def _schedule_close(self):
        if self._auto_close_thread and self._auto_close_thread.is_alive():
            return
        def _runner():
            time.sleep(self.open_sec)
            with self._lock:
                self.driver.close()
                self._is_open = False
        t = threading.Thread(target=_runner, daemon=True)
        self._auto_close_thread = t
        t.start()


# Module-level singleton (lazy)
_gate: Optional[GateController] = None


def get_gate() -> GateController:
    global _gate
    if _gate is None:
        _gate = GateController()
    return _gate
