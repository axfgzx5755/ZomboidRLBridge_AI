"""Validated, race-tolerant access to the Lua telemetry snapshot."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ZOMBOID_USER_DIR = Path(__file__).resolve().parents[2]
DEFAULT_STATE_PATH = ZOMBOID_USER_DIR / "Lua" / "pz_state.txt"
REQUIRED_FIELDS = ("x", "y", "z", "health")


class TelemetryError(RuntimeError):
    pass


@dataclass(frozen=True)
class TelemetrySample:
    x: float
    y: float
    z: float
    health: float
    modified_ns: int
    navigation: dict | None = None

    def as_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z, "health": self.health}


def _validated_number(payload: dict[str, Any], field: str) -> float:
    try:
        value = float(payload[field])
    except (KeyError, TypeError, ValueError) as error:
        raise TelemetryError(f"invalid telemetry field {field!r}") from error
    if not math.isfinite(value):
        raise TelemetryError(f"non-finite telemetry field {field!r}: {value}")
    return value


class TelemetryReader:
    def __init__(
        self,
        path: Path | str = DEFAULT_STATE_PATH,
        *,
        retries: int = 3,
        retry_delay: float = 0.02,
    ) -> None:
        self.path = Path(path)
        self.retries = retries
        self.retry_delay = retry_delay

    def read(self) -> TelemetrySample:
        """Read a complete snapshot, retrying if Lua is replacing the file."""
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                stat = self.path.stat()
                text = self.path.read_text(encoding="utf-8").strip()
                if not text:
                    raise TelemetryError("telemetry file is empty")
                payload = json.loads(text)
                if not isinstance(payload, dict):
                    raise TelemetryError("telemetry payload is not an object")
                if payload.get("missingPlayer") or payload.get("invalidPlayerValues"):
                    raise TelemetryError("no valid active player")
                sample = TelemetrySample(
                    x=_validated_number(payload, "x"),
                    y=_validated_number(payload, "y"),
                    z=_validated_number(payload, "z"),
                    health=_validated_number(payload, "health"),
                    modified_ns=stat.st_mtime_ns,
                    navigation=payload.get("navigation"),
                )
                if not 0.0 <= sample.health <= 1.0:
                    raise TelemetryError(f"health out of range: {sample.health}")
                return sample
            except (OSError, json.JSONDecodeError, TelemetryError) as error:
                last_error = error
                if attempt < self.retries:
                    time.sleep(self.retry_delay)
        raise TelemetryError(f"could not read {self.path}: {last_error}") from last_error

    def wait_for_update(
        self,
        previous: TelemetrySample,
        *,
        timeout: float = 1.5,
        poll_interval: float = 0.02,
    ) -> TelemetrySample:
        """Wait for a snapshot written after ``previous``."""
        deadline = time.monotonic() + timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                current = self.read()
                if current.modified_ns > previous.modified_ns:
                    return current
            except TelemetryError as error:
                last_error = error
            time.sleep(poll_interval)
        suffix = f"; last error: {last_error}" if last_error else ""
        raise TelemetryError(f"telemetry did not update within {timeout:.2f}s{suffix}")
