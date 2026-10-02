# Door training archive — 2026-09-28

This folder preserves the completed single-door baseline.

- `training/final.zip` and `training/final.json`: successful 10,000-step PPO policy and metadata.
- `evaluation/evaluation.jsonl` and `evaluation/summary.json`: independent 30-episode result (100% success).
- `training/`: checkpoints, training configuration, episode history, progress, and final summary.
- `scenario/door-arena-03.json`: exact door/start/target configuration used.
- `code/`: source snapshot for the saved model's environment and training bridge.

The empty evaluation `progress.csv` was removed. Training outputs are retained for reproducibility.
