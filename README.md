# ZomboidRL

Latest results (2026-09-28): the single-door policy reached 30/30 on its captured
scenario. The obstacle-navigation latest policy reached 83% on a fresh 500-episode
synthetic benchmark; the archived live evaluation for best-stage policy reached 5/5.
See [PROGRESS.md](PROGRESS.md) and the [categorized experiment archive](archive/experiment-records-20260928/README.md).

No-API local autonomous exploration (implemented, live testing pending):
[LOCAL_PLAY.md](LOCAL_PLAY.md).
The explorer opens supported doors and now retreats from nearby zombies; this
rule-based behavior is separate from the PPO navigation model.

Current development plan: [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md).
Next: connect the existing navigation and door policies, then train a separate
combat policy and add situation-based policy selection. The integrated runner
has not been implemented yet.

Single-door interaction training instructions:
[DOOR_TRAINING.md](DOOR_TRAINING.md). The captured single-door policy passed 30/30 evaluations.

Obstacle-aware navigation v2: see [NAVIGATION_V2.md](NAVIGATION_V2.md) for the
train/save/evaluate loop, synthetic curriculum, live setup and resume commands.
The saved latest obstacle policy scored 83% in a fresh 500-episode stage-2
synthetic benchmark; the same guide now includes a live evaluate-only workflow.
The older coordinate-only pipeline below remains available.

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

This controller is a deterministic navigation baseline. The PPO workflow below
provides trainable navigation with repeated arena resets.

## Baseline evaluation

With a character loaded, run from `python` and focus the game during the countdown:

```powershell
..\.venv\Scripts\python.exe evaluate.py --offset 10 0 --offset 0 0 --max-steps 100
..\.venv\Scripts\python.exe test_evaluate.py
```

Targets are relative to the position at the start of the run. This example goes
10 tiles along X and returns to the original position. Episodes run sequentially
without resetting the world or restoring health. The default route is the same
out-and-back route. Use an open area for the greedy navigation baseline.

Results append to `python/runs/evaluation.jsonl` (override with `--output`). Each
line records start/final telemetry, target, distance, reward, steps, duration, and
status: `success`, `max_steps`, `dead`, `error`, or `interrupted`. Death, input or
telemetry errors, and Ctrl+C end the run; movement keys are released at episode
exit. Startup requires a fresh telemetry update. Exit codes are 0 for all targets
reached, 1 for step limits, 2 for errors/death, and 130 for interruption.

## PPO training (Build 42.20.4, single-player)

Implemented: bounded arena episodes, acknowledged position/health reset, random
targets, PPO training/checkpoints/resume, and saved-policy evaluation. The arena
uses an existing open region of the map; it does not generate a new map or erase
objects. Training observations contain only position relative to the target and
health. This is a navigation task, not zombie combat or survival training.

### Live game quick start

1. Install dependencies with `.venv/Scripts/python.exe -m pip install -r requirements.txt`.
2. Enable ZomboidRLBridge and reload the game so both `Telemetry.lua` and the new
   `TrainingBridge.lua` are loaded. Both source and installed mod were updated in
   this workspace. For another installation, copy the mod to `%UserProfile%/Zomboid/mods`.
3. Load a **training save** with a living character on clear, level ground, outside
   a vehicle. Use normal game speed. Position/health and basic needs are changed
   by training resets. Choose a broad empty area; the default arena checks the
   square extending 8 tiles in each direction for loaded, free floor squares.
4. From `ZomboidRL/python`, capture the current position while the game is unpaused:

```powershell
..\.venv\Scripts\python.exe rl.py capture --output arena.json
```

5. Start training, then focus and unpause the game during the countdown:

```powershell
..\.venv\Scripts\python.exe rl.py train --live --arena arena.json --timesteps 10000 --output runs/live-01
```

Each episode restores the origin, health, hunger, thirst, fatigue, panic, stress,
pain, and endurance. God/ghost flags protect the living training character and
are restored to their previous values when the Python lease is cancelled or expires
(30 seconds after a crash, once game ticks resume). This is not a full save-state
rewind: time, inventory, clothing, weather, world objects, and nutrition are not
reset. A dead character must be loaded/replaced manually; training prevents ordinary
damage instead of attempting resurrection. Walls between free tiles and moving
obstacles are not fully validated; use an empty field and avoid overburdening.

The target is sampled 2–5 tiles from the origin. Reaching the goal ends the episode;
leaving the arena, death, or 100 steps also ends it. PPO automatically resets the
next episode. Focus loss, telemetry failure, or a rejected reset stops training
and saves `interrupted.zip` if the model was initialized. Ctrl+C does the same.
Keep one controller running at a time; PPO processes use an exclusive OS lock.

Files in the run directory:

- `final.zip` / `final.json`: policy and matching arena/observation configuration.
- `checkpoint_<steps>.zip` / `.json`: periodic recovery checkpoints.
- `interrupted.zip` / `.json`: last model when interrupted or a runtime error occurs.
- `episodes.monitor.csv`: episode reward, length, success, terminal reason.
- `progress.csv`: PPO optimization and rollout metrics.
- `config.json`: run inputs and seed.

Keep a model's `.json` next to its `.zip`. Resume into a **new** output directory:

```powershell
..\.venv\Scripts\python.exe rl.py train --live --resume runs/live-01/final.zip --timesteps 10000 --output runs/live-02
..\.venv\Scripts\python.exe rl.py evaluate --live --model runs/live-02/final.zip --episodes 10 --output runs/live-evaluation.jsonl
```

Resume adds the requested number of steps; PPO may round up to a full rollout.
Checkpoints retain learned weights/optimizer state, not an unfinished rollout or
an exact world snapshot. Evaluation uses deterministic actions and seeded goals,
reports success rate, and appends per-episode results. It uses the same protected
arena reset as training. Use normal game speed and the saved action duration when
comparing runs. Telemetry now arrives every 0.1 seconds; live movement speed still depends on
game ticks, action duration, and inference latency.

### Synthetic smoke test

Omitting `--live` uses a simple point-mass backend and never sends keyboard input
or reset commands to the game. This validates the ML pipeline; synthetic scores
do not measure in-game performance. Live training requires a captured arena.

```powershell
..\.venv\Scripts\python.exe rl.py train --timesteps 2048 --output runs/smoke
..\.venv\Scripts\python.exe rl.py evaluate --model runs/smoke/final.zip --episodes 5
..\.venv\Scripts\python.exe rl.py train --resume runs/smoke/final.zip --timesteps 256 --output runs/smoke-resumed
```

### Tests and troubleshooting

```powershell
..\.venv\Scripts\python.exe -m pip install -r ../requirements-dev.txt
..\.venv\Scripts\python.exe -m unittest test_training test_lua_bridge test_evaluate
..\.venv\Scripts\python.exe test_env.py
..\.venv\Scripts\python.exe test_reward.py
..\.venv\Scripts\python.exe test_live_controller.py
```

Lua tests execute the actual bridge using Lua 5.1 with mock game APIs. They do not
replace a live-game integration test. Reset acknowledgements time out if the mod
has not been reloaded, the game is paused, or there is no active character. An
obstructed-arena error includes the rejected tile; move to clearer ground and
capture a new arena file, or reduce radius and target distances together:

```powershell
..\.venv\Scripts\python.exe rl.py capture --radius 4 --min-distance 1 --max-distance 3 --output arena-small.json
```

PPO API reference: https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

### Testing with nearby obstacles

Reload the save after updating TrainingBridge.lua. Capture using:

```powershell
..\.venv\Scripts\python.exe rl.py capture --allow-obstacles --min-distance 1 --max-distance 2 --output arena-obstacles.json
..\.venv\Scripts\python.exe rl.py evaluate --live --arena arena-obstacles.json --model runs/navigation-next-01/final.zip --episodes 1 --countdown 15 --output runs/obstacle-evaluation.jsonl
```

Keep radius 8 for the saved model's observation scaling. This opt-in mode checks
only the reset tile and sampled goal tiles (up to 32 attempts). Other tiles may
contain obstacles. It does not check path connectivity or teach obstacle avoidance;
a free goal can still be unreachable. Use it to test reset and movement integration.
Results record allow_obstacles; episode step and boundary limits still apply.
Default captures retain full-arena validation.

### Offline reliability checks

New arena captures store integer X/Y tile coordinates, matching the observed
Build 42 teleport reset behavior. Recapture older fractional-origin arenas before
live evaluation; existing arena files and model metadata are not rewritten.
Policy evaluation now appends error/interrupted episode records, including error
type/message, last completed step, reward and last available telemetry, then closes
the environment and propagates the failure. A failed step's unobserved movement is
not counted as a completed step. Pausing still stops live evaluation on telemetry
timeout; this is not automatic pause/resume support.

Run from python: `..\.venv\Scripts\python.exe -m unittest test_policy_evaluate test_training test_lua_bridge test_evaluate`.
These checks do not require a running game or send keyboard input.

### Continuous movement for live PPO

Live PPO train/evaluate now holds shared movement keys across successive actions,
including while awaiting telemetry. Direction changes release only obsolete keys;
NONE, episode termination, reset and close release held keys. A background watchdog
checks focus every 20 ms and releases input if the next action does not arrive
within action duration (capped at one second) plus one second. A watchdog stop
ends the controller session rather than silently restarting input. PPO releases
keys before optimization between rollouts. Process termination/OS failure cannot
be handled by an in-process watchdog.

Reload the save to activate the installed Telemetry.lua update: snapshots every
0.1 seconds, routine coordinate console logging every 5 seconds. Existing standalone
baseline controllers still use their original timed key taps. Existing policy files
load, but changed live movement timing requires a new live evaluation; earlier
scores are not directly comparable. Synthetic backend movement is unchanged.

Offline input tests: `..\.venv\Scripts\python.exe -m unittest test_held_movement`.

## 이동·문 통합 실행

`hybrid_play.py`는 기존 이동 PPO와 문 PPO를 한 경로에서 차례로 실행합니다.
훈련용 위치 초기화는 하지 않습니다. 캐릭터가 직접 출발해 문을 열고,
실제 문 경계를 넘은 뒤 최종 목표까지 이동합니다. 현재 연결 단계에서는
좀비가 4.5타일 안에 들어오거나 체력이 0.8 아래로 내려가면 중단합니다.

1. Project Zomboid에서 모드가 로드되어 있고 게임이 일시정지되지 않았는지 확인합니다.
2. 보관된 `door-arena-03.json`에서 학습한 문과 같은 문을 닫고, 같은 접근 방향의
   열린 바닥에 캐릭터를 둡니다. 이동 정책의 접근 단계도 확인하려면 문에서 3~5타일 떨어지고,
   문 시작 타일에서 8타일 이내인 곳에 서세요.
3. `capture`는 캐릭터를 이동시키지 않고 현재 타일을 기록합니다. `--min-distance`와
   `--max-distance`는 위치를 정하는 옵션이 아니라 구역 설정값입니다. `python` 폴더에서 캡처하고,
   기존 파일은 덮어쓰지 않습니다.

```powershell
..\.venv\Scripts\python.exe rl.py capture --allow-obstacles --min-distance 2 --max-distance 5 --output hybrid-arena-01.json
```

4. 문을 지난 쪽의 비어 있는 바닥 타일을 고르고 그 타일 중심 좌표를 `X Y`에 넣습니다.
   목표는 문 학습 목표보다 최소 0.75타일 더 지나 있어야 하며, 새 구역 중심에서 8타일 안에 있어야 합니다.
5. 아래 명령을 실행합니다. 처음 실행은 단일 연결 확인으로 남기도록 새 출력 폴더를 씁니다.

```powershell
..\.venv\Scripts\python.exe hybrid_play.py --navigation-model ..\archive\experiment-records-20260928\obstacle-navigation\navigation-v2-obstacles-20260918\latest.zip --door-model ..\archive\experiment-records-20260928\door\door-single-v1-20260928\training\final.zip --door-scenario ..\archive\experiment-records-20260928\door\door-single-v1-20260928\scenario\door-arena-03.json --arena hybrid-arena-01.json --goal X Y --output runs\hybrid-01 --countdown 5
```

실행 전에 모델의 JSON 메타데이터와 관측·행동 크기를 검사합니다. 문이 잠겼거나,
열려 있거나, 바리케이드가 있거나, 다른 문이 보이면 제어를 시작하지 않습니다.
실행 중에는 포커스 상실·관측 오류·정체·스텝 한도 초과 때 멈춥니다.
실제 연결 성공을 확인하기 전까지 학습이나 30회 평가를 시작하지 마세요.
문 근처 마지막 접근은 지도 타일 경로로 정확한 시작 쪽 인접 타일에 맞춘 뒤 문 정책으로 전환합니다.

`runs\hybrid-01\summary.json`에는 성공 또는 중단 원인이 기록되고,
`trajectory.jsonl`에는 매 스텝의 단계·행동·위치·체력이 기록됩니다.
