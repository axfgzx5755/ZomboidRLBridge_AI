# 장애물 이동 학습 v2

게임 없이 개발·모의 학습할 수 있고, 나중에 같은 학습 루프를 실제 게임에 연결할 수 있습니다.
기존 `rl.py`의 좌표 전용 모델은 유지되며 새 모델과 호환되지 않습니다.

## 구현한 범위

- 관측 134개: 목표 상대 좌표·거리, 체력, 정체 비율, 최근 이동량, 타일 내 위치,
  주변 5×5타일의 점유 여부와 북/동/남/서 통행 차단 여부.
- Lua에서 벽·문 경계와 창문을 검사합니다. 미로딩 타일은 막힌 것으로 취급합니다.
  문 열기, 창문 넘기, 전투, 계단 이동은 아직 행동에 포함하지 않습니다.
- 에피소드마다 연결된 지도를 탐색해 도달 가능한 목표 후보를 고릅니다.
  실제 게임의 차량·움직이는 물체·충돌 판정까지 경로 가능성을 보증하지는 않습니다.
- 목표 도달 보상, 접근 보상, 시간·정지 페널티. 30스텝 동안 의미 있는 거리 개선이
  없으면 정체로 종료합니다. 경계 이탈·사망·100스텝 제한도 적용합니다.
- 모의 환경은 빈 공간 → 산재한 장애물 → 장애물과 벽의 3단계입니다.
  각 구간 평가 성공률이 기본 80% 이상이면 다음 단계로 올라갑니다.
- 학습 → 구간 저장 → 별도 평가 → 최고 모델·최신 모델 저장을 반복합니다.
  평가는 결정적 행동과 별도 고정 시드를 사용하며 평가 경험은 PPO 학습 버퍼에 넣지 않습니다.
- 키 입력은 실제 PPO 환경에서 연속 유지하고, 종료·포커스 이탈·감시 시간 초과 시 해제합니다.
  최적화·체크포인트 저장 중에는 키를 해제합니다.

## 게임 없이 실행

아래 명령은 `ZomboidRL/python`에서 실행합니다. `--live`가 없으면 게임을 조작하지 않습니다.

```powershell
..\.venv\Scripts\python.exe navigation_train.py --cycles 10 --steps 10000 --episodes 20 --output runs/nav-v2-01
```

`--cycles`는 학습·평가 반복 횟수, `--steps`는 반복당 추가 학습량입니다.
PPO 특성상 실제 학습량은 256스텝 묶음으로 올림됩니다. 무제한 실행이 아니라
지정한 구간을 마치면 정상 종료합니다. 구간을 늘리거나 저장된 모델에서 재개할 수 있습니다.

```powershell
..\.venv\Scripts\python.exe navigation_train.py --resume runs/nav-v2-01/latest.zip --cycles 10 --steps 10000 --output runs/nav-v2-02
```

항상 새 출력 디렉터리를 사용합니다. 기존 모델 ZIP과 같은 이름의 JSON을 함께 보관하세요.
재개는 정책·최적화 상태와 난이도를 복원하며, 진행 중이던 에피소드·PPO 버퍼·난수 상태까지
정확히 복원하는 것은 아닙니다. 새 실행의 최고 모델은 해당 실행 내 평가 기준입니다.

## 실제 게임에 연결할 때

수정된 모드에는 `Navigation.lua`, `Telemetry.lua`, `TrainingBridge.lua`가 필요합니다.
이 작업 공간의 원본과 설치된 모드를 함께 갱신했습니다. 세이브를 다시 불러와야 적용됩니다.
살아 있는 캐릭터, 지상, 차량 밖, 훈련용 세이브, 기본 WASD, 정상 게임 속도를 사용합니다.

1. 위치를 캡처합니다. 주변 전체가 빈 공터일 필요는 없습니다.

```powershell
..\.venv\Scripts\python.exe rl.py capture --allow-obstacles --min-distance 1 --max-distance 3 --output arena-v2.json
```

2. 처음에는 짧게 실행해 입력 방향, 지형 관측, 리셋을 확인합니다. 카운트다운 동안 게임에 포커스를 둡니다.

```powershell
..\.venv\Scripts\python.exe navigation_train.py --live --arena arena-v2.json --cycles 1 --steps 256 --episodes 3 --output runs/nav-live-check
```

3. 확인 후 같은 모델을 이어서 장시간 학습합니다.

```powershell
..\.venv\Scripts\python.exe navigation_train.py --live --resume runs/nav-live-check/latest.zip --cycles 10 --steps 10000 --output runs/nav-live-01
```

실게임에서는 지도 자체를 생성하거나 난이도에 따라 바꾸지 않습니다. 현재 지역에서 리셋하고
목표를 바꿉니다. 장시간 학습에는 게임을 실행하고 포커스·일시정지 해제를 유지해야 합니다.
일시정지·통신 오류·포커스 이탈은 정상 에피소드 실패와 구분해 실행을 중단합니다.
자동으로 일시정지를 풀거나 오류를 무시하고 반복 재시도하지 않습니다.
원인을 해결한 뒤 `interrupted.zip`으로 새 출력 폴더에서 재개할 수 있습니다.

모의 모델을 실게임으로 옮기려면 `--resume <v2모델.zip> --transfer --live --arena <캡처.json>`을
명시합니다. 모델은 불러올 수 있지만 실제 동작과 성공률은 별도 검증이 필요합니다.
모의 이동 방향은 기본 등각투영 WASD를 가정합니다(W: -X/-Y, D: +X/-Y).
사용자 키 설정이나 게임 입력 방향이 다르면 먼저 방향을 확인하고 맞춰야 합니다.

## 저장 모델의 모의 성능 비교

같은 arena 설정의 모의 모델을 동일한 별도 시드로 비교할 수 있습니다.
실게임 입력은 발생하지 않습니다. 학습 중 구간 평가의 기본 시드(100042부터)와
다른 시드를 사용하며, 출력 폴더는 새 경로여야 합니다.

```powershell
..\.venv\Scripts\python.exe navigation_benchmark.py --models runs/navigation-v2-resume-check/latest.zip runs/navigation-v2-obstacles-20260918/best_stage_2.zip --episodes 100 --seed 200000 --stage 2 --output runs/nav-comparison-new
```

`summary.json`에 성공률, 종료 원인별 횟수, 평균 보상·스텝을 기록하며
`model_<번호>_episodes.jsonl`에 개별 에피소드 결과를 남깁니다.
2026-09-18 학습 및 비교 결과는 [PROGRESS.md](PROGRESS.md)에 기록했습니다.

## 학습 결과 파일

| 파일 | 내용 |
|---|---|
| `config.json` | 실행 설정 |
| `episodes.monitor.csv` | 학습 에피소드 보상·길이·종료 원인 |
| `progress.csv` | PPO 최적화 지표 |
| `evaluation.jsonl` | 평가 에피소드 위치·목표·성공·실패 또는 오류 |
| `cycles.jsonl` | 구간별 성공률·난이도·누적 학습량 |
| `checkpoint_<steps>.zip/.json` | 구간 중 복구용 저장 |
| `cycle_<number>.zip/.json` | 평가 전 구간 모델 |
| `best_stage_<stage>.zip/.json` | 난이도별 이번 실행의 최고 평가 모델 |
| `latest.zip/.json` | 마지막 완료 구간과 다음 난이도 |
| `interrupted.zip/.json` | 오류·Ctrl+C 시 가능한 경우 저장한 모델 |
| `status.json` | 완료·오류·중단 상태와 원인 |

## 검증 및 한계

```powershell
..\.venv\Scripts\python.exe -m unittest test_navigation test_held_movement test_policy_evaluate test_training test_lua_bridge test_evaluate
```

모의 환경은 타일 충돌과 벽을 포함한 간단한 이동 모델입니다. PZ의 애니메이션, 관성,
캐릭터 충돌 크기, 차량, 동적 장애물, 시간 진행을 정확하게 재현하지 않습니다.
5×5 관측만 사용하는 MLP이므로 큰 미로에서의 기억·전역 경로 탐색 능력을 보장하지 않습니다.
실게임 연동은 코드와 모의 API 테스트까지 완료했으며, 변경 후 게임 검증은 남아 있습니다.

지형 API 참고: [IsoGridSquare 공식 문서](https://projectzomboid.com/modding/zombie/iso/IsoGridSquare.html).
로컬 게임 Lua의 `isBlockedTo` 사용도 확인했습니다.

?? ??? ??? ?? ????? ??? ??? ?? ? arena ??? `--transfer`? ?????.
?? ??? ?? ??? ?? ????? ??? ??? ? ?? ???? ?? ?????????.

### Evaluate a saved model without training

Use `--evaluate-only` to run a fixed number of episodes and save the evaluation records and summary. For a new live-game area, capture an obstacle-aware arena with the model's 2–5 tile target range:

```powershell
..\.venv\Scripts\python.exe rl.py capture --allow-obstacles --min-distance 2 --max-distance 5 --output arena-v2-eval.json
```

Then evaluate the archived latest obstacle policy:

```powershell
..\.venv\Scripts\python.exe navigation_train.py --live --arena arena-v2-eval.json --resume ..\archive\experiment-records-20260928\obstacle-navigation\navigation-v2-obstacles-20260918\latest.zip --transfer --evaluate-only --episodes 20 --seed 700000 --countdown 5 --output runs/nav-live-eval-latest-01
```

This loads the policy and evaluates it without PPO updates. Use a fresh output directory. The run stores per-episode outcomes in `evaluation.jsonl`, aggregate results in `summary.json`, and completion or failure in `status.json`. Live evaluation moves the character in the game, so use a training save and an area you can safely reset.
