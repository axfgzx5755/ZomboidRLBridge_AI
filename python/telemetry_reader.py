import json
from pathlib import Path

# Project Zomboid's getFileWriter stores files under the user Lua directory.
# Resolve from this script instead of hard-coding a particular Windows user.
ZOMBOID_USER_DIR = Path(__file__).resolve().parents[2]
STATE_PATH = ZOMBOID_USER_DIR / "Lua" / "pz_state.txt"


def read_state():
    if not STATE_PATH.exists():
        return None

    try:
        with STATE_PATH.open("r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            return None
        return json.loads(text)
    except Exception as e:
        print(f"[telemetry_reader] read error: {e}")
        return None


if __name__ == "__main__":
    import time

    while True:
        state = read_state()
        if state is not None:
            print(state)
        time.sleep(0.5)
