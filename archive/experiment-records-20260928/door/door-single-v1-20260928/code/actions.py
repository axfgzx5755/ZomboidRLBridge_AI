"""Keyboard and mouse actions for Project Zomboid.

Inputs are sent only while a Project Zomboid window is in the foreground.  This
prevents an experiment from typing into an editor or terminal by accident.
"""

from __future__ import annotations

import ctypes
import math
import threading
import sys
import time
from ctypes import wintypes
from enum import IntEnum
from typing import Iterable


if sys.platform != "win32":
    raise RuntimeError("Project Zomboid actions currently support Windows only")


class ActionError(RuntimeError):
    """Raised when an action cannot be executed safely."""


class Movement(IntEnum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    NORTH_WEST = 5
    NORTH_EAST = 6
    SOUTH_WEST = 7
    SOUTH_EAST = 8


_MOVEMENT_KEYS: dict[Movement, tuple[int, ...]] = {
    Movement.NONE: (),
    Movement.NORTH: (0x57,),  # W
    Movement.SOUTH: (0x53,),  # S
    Movement.WEST: (0x41,),  # A
    Movement.EAST: (0x44,),  # D
    Movement.NORTH_WEST: (0x57, 0x41),
    Movement.NORTH_EAST: (0x57, 0x44),
    Movement.SOUTH_WEST: (0x53, 0x41),
    Movement.SOUTH_EAST: (0x53, 0x44),
}
_ALL_MOVEMENT_KEYS = (0x57, 0x53, 0x41, 0x44)

_INPUT_KEYBOARD = 1
_INPUT_MOUSE = 0
_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_SCANCODE = 0x0008
_MAPVK_VK_TO_VSC = 0
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP = 0x0004

_ULONG_PTR = wintypes.WPARAM


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    )


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    )


class _INPUT_UNION(ctypes.Union):
    _fields_ = (("ki", _KEYBDINPUT), ("mi", _MOUSEINPUT))


class _INPUT(ctypes.Structure):
    _anonymous_ = ("union",)
    _fields_ = (("type", wintypes.DWORD), ("union", _INPUT_UNION))


_user32 = ctypes.WinDLL("user32", use_last_error=True)
_user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(_INPUT), ctypes.c_int)
_user32.SendInput.restype = wintypes.UINT


def foreground_window_title() -> str:
    """Return the title of the current foreground window."""
    window = _user32.GetForegroundWindow()
    length = _user32.GetWindowTextLengthW(window)
    buffer = ctypes.create_unicode_buffer(length + 1)
    _user32.GetWindowTextW(window, buffer, len(buffer))
    return buffer.value


def is_project_zomboid_foreground() -> bool:
    return "project zomboid" in foreground_window_title().lower()


def _require_game_focus() -> None:
    if not is_project_zomboid_foreground():
        title = foreground_window_title() or "<untitled>"
        raise ActionError(
            f"Project Zomboid is not the foreground window (foreground={title!r})"
        )


def _send_input(input_event: _INPUT) -> None:
    sent = _user32.SendInput(1, ctypes.byref(input_event), ctypes.sizeof(_INPUT))
    if sent != 1:
        error = ctypes.get_last_error()
        raise OSError(error, "SendInput failed")


def _key_event(virtual_key: int, *, key_up: bool) -> None:
    scan_code = _user32.MapVirtualKeyW(virtual_key, _MAPVK_VK_TO_VSC)
    flags = _KEYEVENTF_SCANCODE | (_KEYEVENTF_KEYUP if key_up else 0)
    event = _INPUT(
        type=_INPUT_KEYBOARD,
        union=_INPUT_UNION(
            ki=_KEYBDINPUT(
                wVk=0,
                wScan=scan_code,
                dwFlags=flags,
                time=0,
                dwExtraInfo=0,
            )
        ),
    )
    _send_input(event)


def _mouse_left_event(flags: int) -> None:
    event = _INPUT(
        type=_INPUT_MOUSE,
        union=_INPUT_UNION(
            mi=_MOUSEINPUT(0, 0, 0, flags, 0, 0)
        ),
    )
    _send_input(event)


def _release_keys(keys: Iterable[int]) -> None:
    for key in keys:
        _key_event(key, key_up=True)


def stop() -> None:
    """Release every movement key, including after an interrupted action."""
    _release_keys(_ALL_MOVEMENT_KEYS)


def move(direction: Movement | int, duration: float = 0.25) -> None:
    """Hold one of the eight movement directions for ``duration`` seconds."""
    direction = Movement(direction)
    if duration < 0:
        raise ValueError("duration must be non-negative")
    if direction is Movement.NONE:
        stop()
        return

    _require_game_focus()
    keys = _MOVEMENT_KEYS[direction]
    stop()
    try:
        for key in keys:
            _key_event(key, key_up=False)
        time.sleep(duration)
    finally:
        _release_keys(keys)


class HeldMovement:
    """Persistent movement with a bounded watchdog, including between policy steps."""

    def __init__(self, timeout=1.0):
        self.timeout = timeout
        self._keys = set()
        self._deadline = 0.0
        self._failure = None
        self._lock = threading.RLock()
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def _release(self):
        first_error = None
        for key in tuple(self._keys):
            try:
                _key_event(key, key_up=True)
                self._keys.discard(key)
            except OSError as error:
                first_error = first_error or error
        if first_error:
            raise first_error

    def check(self):
        with self._lock:
            if self._failure:
                raise ActionError(self._failure)
            if self._closed.is_set():
                raise ActionError("movement controller is closed")

    def _watch(self):
        while not self._closed.wait(0.02):
            with self._lock:
                if not self._keys:
                    continue
                try:
                    if not is_project_zomboid_foreground():
                        raise ActionError("game lost focus; movement stopped")
                    if time.monotonic() >= self._deadline:
                        raise ActionError("movement watchdog expired; restart the controller")
                except Exception as error:
                    self._failure = str(error)
                    try:
                        self._release()
                    except OSError:
                        pass  # Retry any failed releases on the next watchdog tick.

    def move(self, direction, duration):
        try:
            direction = Movement(direction)
            if not math.isfinite(duration) or duration < 0:
                raise ValueError("duration must be finite and nonnegative")
            with self._lock:
                self.check()
                _require_game_focus()
                wanted = set(_MOVEMENT_KEYS[direction])
                for key in self._keys - wanted:
                    _key_event(key, key_up=True)
                self._keys.intersection_update(wanted)
                for key in wanted - self._keys:
                    self._keys.add(key)  # Include partially failed key-down in cleanup.
                    _key_event(key, key_up=False)
                self._deadline = time.monotonic() + min(duration, 1.0) + self.timeout
            until = time.monotonic() + duration
            while time.monotonic() < until:
                self.check()
                time.sleep(min(0.02, max(0, until - time.monotonic())))
            self.check()
        except BaseException:
            self.stop()
            raise

    def stop(self):
        with self._lock:
            self._release()

    def close(self):
        try:
            self.stop()
        finally:
            self._closed.set()
            self._thread.join(timeout=1)


def _tap_key(virtual_key: int, duration: float = 0.05) -> None:
    _require_game_focus()
    try:
        _key_event(virtual_key, key_up=False)
        time.sleep(duration)
    finally:
        _key_event(virtual_key, key_up=True)


def move_north(duration: float = 0.25) -> None:
    move(Movement.NORTH, duration)


def move_south(duration: float = 0.25) -> None:
    move(Movement.SOUTH, duration)


def move_west(duration: float = 0.25) -> None:
    move(Movement.WEST, duration)


def move_east(duration: float = 0.25) -> None:
    move(Movement.EAST, duration)


def move_nw(duration: float = 0.25) -> None:
    move(Movement.NORTH_WEST, duration)


def move_ne(duration: float = 0.25) -> None:
    move(Movement.NORTH_EAST, duration)


def move_sw(duration: float = 0.25) -> None:
    move(Movement.SOUTH_WEST, duration)


def move_se(duration: float = 0.25) -> None:
    move(Movement.SOUTH_EAST, duration)


def interact(duration: float = 0.05) -> None:
    _tap_key(0x45, duration)  # E


def shove(duration: float = 0.05) -> None:
    _tap_key(0x20, duration)  # Space


def attack(duration: float = 0.05) -> None:
    _require_game_focus()
    try:
        _mouse_left_event(_MOUSEEVENTF_LEFTDOWN)
        time.sleep(duration)
    finally:
        _mouse_left_event(_MOUSEEVENTF_LEFTUP)

