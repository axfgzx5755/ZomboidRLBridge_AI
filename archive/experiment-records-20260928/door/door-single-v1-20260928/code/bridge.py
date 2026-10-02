"""Single-client, expiring request/acknowledgement protocol for the Lua mod."""

import json
import math
import os
import time
import uuid
from pathlib import Path

from telemetry import DEFAULT_STATE_PATH, TelemetryError


class BridgeError(RuntimeError):
    pass


def atomic_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        for attempt in range(10):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.02)
    finally:
        temporary.unlink(missing_ok=True)


class TrainingBridge:
    def __init__(self, directory=DEFAULT_STATE_PATH.parent, timeout=8.0):
        self.directory = Path(directory)
        self.timeout = timeout
        self.session = uuid.uuid4().hex
        self.lease = self.directory / "pz_training_lease.txt"
        self.lock = self.directory / "pz_training.lock"
        self._lock_file = None
        self._last_heartbeat = None

    def acquire(self):
        # OS byte lock is released even if Python crashes; the file can remain.
        import msvcrt
        self._lock_file = self.lock.open("a+b")
        self._lock_file.write(b"0")
        self._lock_file.flush()
        self._lock_file.seek(0)
        try:
            msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            self._lock_file.close()
            self._lock_file = None
            raise BridgeError("another training process owns the game") from error

    def heartbeat(self):
        if self._last_heartbeat is not None and time.monotonic() - self._last_heartbeat >= 25:
            raise BridgeError("training lease expired during inactivity; restart from the saved checkpoint")
        atomic_text(self.lease, f"{self.session} {time.time() + 30:.3f}\n")
        self._last_heartbeat = time.monotonic()

    def request(self, operation, arena):
        request_id = uuid.uuid4().hex
        self.heartbeat()
        values = [arena.x, arena.y, arena.z, arena.radius]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("arena coordinates must be finite")
        command = " ".join([request_id, self.session, f"{time.time() + self.timeout:.3f}",
                            operation, *(str(v) for v in values)])
        atomic_text(self.directory / "pz_training_command.txt", command + "\n")
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                ack = json.loads((self.directory / "pz_training_ack.txt").read_text())
                if isinstance(ack, dict) and ack.get("id") == request_id:
                    if ack.get("status") != "ok":
                        raise BridgeError(f"Lua {operation}: {ack.get('error', 'unknown error')}")
                    return ack
            except (OSError, ValueError):
                pass
            time.sleep(0.05)
        raise BridgeError(f"Lua {operation} acknowledgement timed out; reload the mod and unpause the game")

    def reset(self, arena, reader):
        self.request("reset_local" if arena.allow_obstacles else "reset", arena)
        sample = reader.wait_for_update(reader.read(), timeout=self.timeout)
        if math.hypot(sample.x - arena.x, sample.y - arena.y) > 0.75 or abs(sample.z - arena.z) > 0.1:
            raise TelemetryError("reset acknowledged but player is not at the arena origin")
        if sample.health < 0.99:
            raise TelemetryError("reset did not restore health")
        return sample

    def close(self):
        if self._lock_file is not None:
            try:
                # Explicit cancellation is distinguishable from a transient read failure.
                atomic_text(self.lease, f"{self.session} 0\n")
            finally:
                self._lock_file.close()
                self._lock_file = None
