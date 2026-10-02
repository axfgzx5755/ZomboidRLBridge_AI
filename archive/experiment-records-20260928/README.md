# Project Zomboid RL experiment archive — 2026-09-28

This archive keeps the strongest results, model files, their matching metadata, evaluation data, and concise records for earlier failures. Project source code remains in `../../python` and the Lua mod folders.

## Categories

- `door/`: single-door baseline (10,496 PPO timesteps, 30/30 evaluation), its captured arena, code snapshot, and successful setup checks.
- `obstacle-navigation/`: strongest obstacle-navigation policies and full training/evaluation history. `best_stage_2.zip` is the selected benchmark policy; `latest.zip` is the final resume checkpoint.
- `benchmarks/`: two 100-episode seed runs comparing the navigation baseline and obstacle policies.
- `navigation/`: earlier navigation policies and their training/evaluation records.
- `live-validation/`: completed live checks and evaluations, plus failure status for interrupted live checks.
- `local-exploration/`: room/exploration output and run status.
- `implementation/`: final policies from implementation runs.
- `diagnostics/`: configuration and status for earlier failed/interrupted runs.
- `project-docs/`: progress, training and design guides as they stood at archive time.

## Best obstacle result

`obstacle-navigation/navigation-v2-obstacles-20260918/best_stage_2.zip` scored 85% over 100 synthetic obstacle episodes and 84% on a second 100-episode seed. `latest.zip` scored 84% and 84% on those same two benchmark sets. The separate live evaluation recorded 5/5 success for its own live model. Synthetic benchmark results and live game results measure different conditions.

Each policy ZIP is stored with its JSON metadata. Keep both together when resuming or evaluating a policy. Intermediate periodic checkpoints are excluded from the archive; the best policy, latest resume point, final reports, episode records, and training configuration are retained.



## Extended stage-2 benchmark (seed 900000)

A fresh 500-episode synthetic evaluation per model compared the baseline with both obstacle policies. The baseline scored 65.8% (329/500), best_stage_2 scored 78.6% (393/500), and latest scored 83.0% (415/500). The latest checkpoint had the highest score and lowest mean steps (13.446). Per-episode results and configuration are in benchmarks/nav-benchmark-stage2-500-20260928/.
