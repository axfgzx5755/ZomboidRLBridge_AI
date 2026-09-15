# ZomboidRL

Project Zomboid Build 42 telemetry and reinforcement-learning experiment.

## Layout

- `ZomboidRLBridge/`: Lua telemetry mod source
- `python/`: telemetry reader, Windows action controller, Gymnasium environment,
  reward logic, and tests
- `requirements.txt`: Python runtime dependencies

The live mod writes `pz_state.txt` through `getFileWriter`, which places it in
`%UserProfile%\Zomboid\Lua`. The Python reader resolves that location from the
project path.

## Setup and tests

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd python
..\.venv\Scripts\python.exe test_reward.py
..\.venv\Scripts\python.exe test_env.py
..\.venv\Scripts\python.exe test_live_controller.py
```

## Automatic controller

With Project Zomboid running and a character loaded:

```powershell
cd python
..\.venv\Scripts\python.exe live_controller.py --relative-x 10 --relative-y 0
```

The current controller is a deterministic navigation baseline. Its action
selection will later be replaced with a PPO policy after automated episode reset
and a training arena are implemented.

