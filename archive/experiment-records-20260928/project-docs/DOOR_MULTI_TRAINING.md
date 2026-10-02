# Multiple-door training

The single-door baseline and its 30/30 evaluation are preserved in `archive/door-single-v1-20260928`.
This workflow adds a scenario curriculum: each episode randomly selects one captured door and resets beside it.
The policy still receives the same 142-value observation and chooses among the same 10 actions.

## Capture and check additional doors

For each new ordinary, unlocked door, stand on one of the two door tiles and capture a distinct scenario:

```powershell
..\.venv\Scripts\python.exe door_train.py capture --output door-arena-04.json
```

Check each scenario before training. Use a fresh output folder each time:

```powershell
..\.venv\Scripts\python.exe door_train.py check --scenario door-arena-04.json --output runs/door-check-04
```

Repeat for another door, such as `door-arena-05.json`. The saved scenario fixes the start side, door orientation, and clear target tile. Check scenarios from different buildings and both door orientations when available.

## Train across the captured doors

Once every scenario passes its individual check, start from the archived successful model and provide all scenarios. Each episode will randomly choose one:

```powershell
..\.venv\Scripts\python.exe door_train.py train --scenario door-arena-03.json --scenario door-arena-04.json --scenario door-arena-05.json --model ..\archive\experiment-records-20260928\door\door-single-v1-20260928\training\final.zip --steps 10000 --episodes 30 --output runs/door-multi-01
```

The first scenario must be one the model was trained with. Add scenarios on later runs while keeping the original scenarios in the command, then resume from the latest `final.zip`.

## Evaluate

Evaluate across the same scenario set:

```powershell
..\.venv\Scripts\python.exe door_train.py evaluate --scenario door-arena-03.json --scenario door-arena-04.json --scenario door-arena-05.json --model runs/door-multi-01/final.zip --episodes 60 --seed 123 --output runs/door-multi-eval-01
```

The evaluation randomly chooses among the supplied scenarios each episode. For a meaningful held-out-door test, capture a door and do not include its scenario during training; evaluate separately with that scenario and the trained model.

Keep the game focused and unpaused during the 5-second countdown and throughout training/evaluation. A model trained on the original one-door scenario cannot be assessed on a new door until the new scenario is captured and passed through the check command.

### Capture limit

The bridge currently requires each reset destination to be within 50 tiles of the player's current loaded position. Capture training doors close enough that resets can move between them reliably (keep the whole group within roughly 50 tiles). The multi-scenario code changes the target door and arena at episode reset; live training on multiple different rooms still needs to be verified in game.
