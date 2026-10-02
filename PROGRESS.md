# 진행 상황

## 2026-10-01: CNN 후속 검증 10개 seed 완료

- 이전과 같은 구조·PPO·학습량·증강 OFF로 새 학습 seed 81~90을 구조별 10회 실행. 모델당 32,768스텝, 시험 seed 2200000~2200499에서 500회 평가.
- 평균 성공률 MLP 16.82%, CNN 64.60%. paired 평균 차이 +47.78%p, 학습 seed 기반 t 95% 구간 +42.09~+53.47%p. 10쌍 모두 CNN이 높음. 이전 3개 seed와 합산하지 않음.
- 평균 학습 시간 MLP 20.64초, CNN 28.04초로 1.36배. 총 학습 655,360스텝, 평가 실행 10,000회, 고유 시험 seed 500개.
- 20개 모델의 수치 유한성, 스텝 수, 에피소드 seed·성공 집계 일치 검증 완료. 모델/로그와 PNG·SVG 그래프 보관. 학습 코드·설정은 변경하지 않음.
- 기존 162,816스텝 모델은 같은 시험 지형에서 416/500(83.2%). 실사용 모델 유지. 장기 동일 예산 비교와 문 실게임 검증은 아직 수행하지 않음.
- 결과: [CNN_CONFIRMATION.md](CNN_CONFIRMATION.md), `python/runs/cnn-confirm-10seeds-20260930/`. 그림 재현을 위해 개발 의존성에 Matplotlib 추가.

## 2026-09-30: CNN과 MLP 실측 비교

- PyTorch `GridCNN` 추가: 기존 5×5×5 격자에 3×3 합성곱 두 층, 48개 특징과 scalar 입력을 합쳐 기존 64×64 actor/critic에 전달. 파라미터는 MLP 26,250개, CNN 26,994개.
- 두 구조를 처음부터 각각 32,768스텝, seed 71/72/73에서 학습. 증강 OFF, 동일 PPO 설정·고정 커리큘럼. 모델별 미사용 stage-2 지형 500개에서 평가.
- 평균 성공률 MLP 22.07%, CNN 50.00%: +27.93%p. 평균 학습 시간 31.62초 대 44.97초(1.42배). CNN seed별 7.4/72.8/69.8%로 편차가 커 안정적인 개선은 미확정.
- 기존 162,816스텝 기준 모델은 같은 지형에서 421/500(84.2%). 예산이 달라 구조 효과 비교에는 포함하지 않았으며 실사용 모델 유지.
- 전체 unittest 83개 통과. 입력 채널 배열, CNN 가중치 갱신, 저장/로드, paired 통계 검증. 실험 통계 재현용 SciPy를 개발 의존성에 추가.
- 결과·재현: [CNN_COMPARISON.md](CNN_COMPARISON.md), `python/runs/cnn-vs-mlp-20260930/`, `python/runs/cnn-existing-reference-20260930/`.

## 2026-09-30: 학습 품질 점검·증강 대조 실험

- 기존 이동/문 PPO ZIP을 직접 점검: 각각 26,250/27,339개 파라미터, 가중치와 Adam 상태의 NaN/Inf 없음. 문 critic의 마지막 기록 explained variance는 -0.618로 별도 추적 필요.
- 원래 좌표계와 x/y 교환 좌표계를 에피소드 단위로 사용하는 선택적 증강 추가. 격자·벽 방향·문 방향·이동 행동을 함께 변환하며 PPO rollout의 log probability 대응을 보존.
- 두 학습기에 재개 시 적용되는 LR/epochs/target KL/entropy 옵션, 전후 모델 상태 기록, 마지막 학습 로그 저장 추가. 문 평가는 시나리오 균등 배정과 문별 지표/coverage, 이동은 cycle별 평가 요약 추가.
- 합성 이동에서 3개 학습 seed × 증강 OFF/ON × 4,096스텝 실험. 검증 100회 후 후보 선택, 독립 시험 300회. 기존 257/300(85.67%), 증강 평균 83.89%, 대조군 평균 82.67%. 기존 모델 대비 개선이 없어 기본 모델 유지.
- 문 실게임 재학습은 게임 프로세스가 없어 미실행. 문 데이터의 물리적 분리와 실제 학습 명령은 [TRAINING_QUALITY.md](TRAINING_QUALITY.md)에 정리.
- 전체 unittest 80개 통과. 증강 CLI 256스텝 학습/저장/평가 실행 확인; 이 짧은 실행의 평가 0/3은 성능 개선 근거가 아님. 기존 free-play Lua 테스트 모형에 누락된 `getZombieList()`를 추가했으며 게임 Lua 동작은 수정하지 않음.
- 산출물: `python/runs/quality-audit-20260930/`, `python/runs/quality-pilot-20260930/`, `python/runs/quality-cli-smoke-20260930/`.

## 현재 개발 방향 (2026-09-28)

- 실행 기준: [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md).
- 다음 작업: 기존 이동·문 정책의 로더와 훈련 리셋 없는 공통 실행 경로 구현.
- 이후: 전투 전용 정책 학습 → 상위 제어기의 상황별 정책 선택.
- 통합 실행기는 아직 미구현. 아래 날짜별 기록의 당시 다음 단계와 구분한다.
- 이전 AGENT_DESIGN.md는 내용이 같은 보관본을 확인한 뒤 삭제하고 새 계획서로 대체.
- 학습·평가 안내, 소스, 모델과 실행 기록은 계속 보존한다.

## 2026-09-26: 오프라인 검증 및 탐색 기억 수정

- 기존 unittest 68개 통과 후, 관측 범위 밖 건물에 연결된 문의 상태가 갱신되지 않는 오류를 새 회귀 테스트로 재현.
- `local_explorer.py`: 관측한 문의 열림/닫힘을 같은 세계의 모든 기존 건물 연결에 반영. 관측하지 않은 문은 마지막 상태 유지.
- `test_places.py`: 부분 관측 시 열림·다시 닫힘 반영, 다른 세계 기록 격리, 미관측 문 상태 유지 검증 추가.
- 수정 후 `python -m unittest discover`: 70개 통과. 독립 실행형 `test_env.py`, `test_reward.py`, `test_live_controller.py`도 통과.
- 원본/설치 Lua 6개 코드 내용 동일. TrainingBridge.lua의 해시 차이는 줄바꿈 차이뿐.
- 게임 프로세스가 없어 실게임 조작·문 학습·자율 탐색 평가는 실행하지 않음. 오프라인 통과는 실게임 성능 검증이 아님.
- 다음 단계: 세이브 재로드 후 `LOCAL_PLAY.md`의 1분 자율 탐색 점검. 문 PPO는 `DOOR_TRAINING.md`의 capture → check → 256스텝 학습 순서로 별도 검증.

아래의 테스트 미실행 표기는 당시 작업 기록이며, 최신 오프라인 검증 결과는 위와 같음.

## 2026-09-20: 방·건물 탐색 확장

- Places.lua로 관측 타일의 건물·방 정보를 추가하고 원본/설치 모드에 반영.
- 기존 방문 DB를 보존하며 방·건물 관측 및 문 상태 테이블을 추가.
- 미방문 방, 현재 건물 미방문 타일 등의 목적 우선순위로 목표 선택.
- exploration.json에 관측 범위 내 진행 상태 저장. 새 관측으로 완료 상태가 다시 미완료가 될 수 있음.
- test_places.py와 기존 모의 관측/브리지 테스트를 갱신. 사용자 요청에 따라 실행하지 않음.

## 2026-09-20: API 없는 로컬 자율 탐색 구현

- local_play.py: 자유 플레이 실행, 공유 제어 잠금, 시간 제한/무제한 모드, 포커스 대기, 종료 시 입력 해제.
- local_explorer.py: 주변 지형 경로 탐색, 문 선택, 방문 기록, 실패 경로 재시도 제한, SQLite 저장.
- ActionBridge.lua: 관측 및 지정 문 열기만 제공. 순간이동·건강 복구·보호 플래그 변경 없음.
- 원본 및 설치 모드에 반영. 사용자용 test_local_play.py 및 LOCAL_PLAY.md 추가.
- 외부 모델 API 및 PPO 학습 없이 실행. 기억 기반 탐색이며 전투·파밍·생존 자동화는 미구현.
- 사용자 지시에 따라 테스트와 게임 조작은 실행하지 않음. 실제 호환성·탐색 성능 검증 대기.

## 2026-09-20: 문 상호작용 코드 확장 (사용자 테스트 대기)

- `DoorTraining.lua`: 인접 단일 문 검색, 상태 조회, 제한된 열기, 리셋용 닫기.
- `door_env.py`: 142개 관측과 10개 행동, 최초 열기 보상, 실제 문틈 통과·목표 도달 판정.
- `door_train.py`: 캡처, 짧은 연결 점검, 실게임 PPO 학습·재개·평가.
- `test_door.py`: 사용자가 실행할 모의 환경·Lua 회귀 테스트 작성.
- 원본과 설치 모드에 반영. 기존 이동 모델과 별도의 door-v1 모델 사용.
- 사용자 요청에 따라 테스트·실게임 조작·학습은 실행하지 않음. 성공 여부 미검증.
- 실행 절차와 결과 판독: [DOOR_TRAINING.md](DOOR_TRAINING.md).

## 확인된 상태

- Lua 텔레메트리, 훈련 리셋, 주변 지형 관측 및 134개 관측값 기반 PPO v2 구현.
- 모의/실게임 학습, 평가, 체크포인트 저장, 재개 및 연속 키 입력 지원.
- `python/runs/live-check-03/status.json`: 1구간, 256스텝 완료.
- 같은 실행의 평가: 3회 모두 `out_of_bounds`, 성공률 0%. 실행 완료는 이동 학습 성공을 뜻하지 않음.
- `navigation-v2-wide-check-01`은 6구간에서 게임 포커스 미확보 오류로 중단됨.
- 확인 시점에 게임 프로세스가 없어 새 실게임 검증은 수행하지 않음.

## 이번 변경

- 원본과 설치된 `Telemetry.lua`에 동일하게 유한수 검증 적용.
- 숫자가 아닌 좌표, NaN, 무한대 좌표/체력을 정상 관측으로 발행하지 않고 `invalidPlayerValues`로 표시.
- 플레이어 부재/잘못된 값에 대한 매 틱 중복 로그 제거. 기존 10초 간격 경고 유지.
- 실제 Lua를 실행하는 회귀 테스트 추가: 정상 관측, 잘못된 값과 복구, 플레이어 부재 경고 제한.

## 후속 진행: 장애물 학습 확대 완료

사용자가 실게임 이동 정상 동작을 확인하여 재검증은 생략하고 모의 학습을 진행함.
이전 기록을 추가 확인한 결과 `navigation-v2-live-check-02`의 평가 3회는 모두 성공했음.
초기화된 모델인 `live-check-03`의 실패 결과만으로 기존 학습 모델의 성능을 판단하지 않음.

- 시작 모델: `python/runs/navigation-v2-resume-check/latest.zip` (61,696스텝).
- 장애물·벽 단계에서 20,000스텝 요청 × 5구간 추가 학습. 롤아웃 올림으로 실제 추가 101,120스텝, 누적 162,816스텝.
- 출력: `python/runs/navigation-v2-obstacles-20260918/`.
- 구간별 30회 평가 성공률: 83.3%, 90%, 90%, 70%, 73.3%.
- 동일한 별도 시드 200000~200099에서 각 모델을 100회 비교:

| 모델 | 성공 | 정체 종료 | 평균 스텝 |
|---|---:|---:|---:|
| 기존 모델 | 69% | 31회 | 18.31 |
| 추가 학습 최고 평가 모델 | 85% | 15회 | 16.08 |
| 추가 학습 마지막 모델 | 84% | 16회 | 13.22 |

최고 평가 모델은 `python/runs/navigation-v2-obstacles-20260918/best_stage_2.zip`이며
동명 JSON을 함께 보관해야 함. 이 모델은 구간 평가로 선택했으며 별도 평가에서 기존 모델보다
16%p 높은 성공률을 기록함. 모의 환경 결과이며 실게임 성공률을 의미하지 않음.
100회 평가의 85%와 84% 차이만으로 두 새 모델 간 확실한 우열을 주장하지 않음.

재현 가능한 비교 도구 `python/navigation_benchmark.py` 추가.
상세 결과는 `python/runs/navigation-v2-benchmark-20260918/summary.json` 및 모델별 JSONL에 저장.
기존 전체 테스트 46개 통과, 비교 도구는 실제 세 모델의 총 300회 평가로 실행 검증함.

다음 개선 대상은 남은 정체 실패 15회 분석과 우회 경로 학습 보강.

## 추가 테스트 (사용자 요청)

- 전체 unittest 46개 통과.
- 별도 시드 300000~300099, 장애물·벽 단계에서 기존/최고 모델 각 100회 모의 평가 완료.
- 기존 모델: 성공 65회, 정체 35회, 평균 20.29스텝.
- 추가 학습 최고 모델: 성공 84회, 정체 16회, 평균 14.80스텝.
- 이번 비교에서 성공률 19%p 향상. 앞선 별도 시드 평가와 합하면 기존 134/200(67%), 새 모델 169/200(84.5%).
- 결과: `python/runs/navigation-v2-benchmark-20260918-seed300000/summary.json`.
- 실게임 재검증은 앞서 정한 대로 생략.

## 실게임 검증 참고 절차 (사용자 확인으로 생략)

후속 요청으로 실게임 평가를 실행함: `navigation-v2-obstacles-20260918/best_stage_2.zip`,
현재 위치에서 캡처한 arena, 목표 거리 2~5타일, 시드 400000~400004.
5회 모두 성공(100%)했고 평가 종료 후 환경을 닫아 키 입력 및 훈련 lease를 정리함.
결과: `python/runs/navigation-v2-live-eval-20260918-130442/evaluation.jsonl`.
이 결과는 현재 지역에서의 짧은 이동 평가이며 복잡한 장애물 전반의 성공률을 의미하지 않음.

1. 훈련용 세이브를 다시 로드하여 변경된 모드를 적용.
2. 짧은 방향별 입력과 좌표 변화를 대조하여 WASD 방향 및 행동 시간을 확인.
3. 리셋 직후 위치와 목표까지의 거리 변화를 확인한 뒤 짧은 학습/평가 재실행.
4. 성공률 및 영역 이탈 빈도를 확인한 후 학습량 확대 여부 결정.

256스텝의 초기 정책 평가만으로 실패 원인을 입력 방향 또는 코드 오류라고 확정할 수 없음.
현재 보완은 관측 데이터의 신뢰성을 높이는 변경이며 영역 이탈 해결을 입증하지 않음.

## 2026-09-28 follow-up: archived results and independent evaluation

- Classified key policies and records under `archive/experiment-records-20260928/`; the archive README describes each category and keeps model metadata beside each ZIP.
- The strongest obstacle policy (`latest.zip`) scored 415/500 (83.0%) in a fresh synthetic stage-2 benchmark, seed 900000. The baseline scored 329/500 (65.8%); `best_stage_2.zip` scored 393/500 (78.6%).
- Per-episode records and summary: `python/runs/nav-benchmark-stage2-500-20260928/`; an archive copy is in `archive/experiment-records-20260928/benchmarks/`.
- Added `--evaluate-only` to `navigation_train.py` so a saved policy can be evaluated live without PPO training. Documented the capture and evaluation commands in `NAVIGATION_V2.md`.
- Python syntax compilation passed. Live evaluation of the latest obstacle policy in a newly captured area remains to be run by the user.

## 2026-09-28 capability preparation

- Extended the free-play ActionBridge snapshot with same-floor zombie coordinates within 8 tiles.
- The local explorer now temporarily flees toward the safest reachable path when a zombie is within 4.5 tiles, then resumes room exploration. Its supported unlocked-door behavior remains enabled.
- Updated both workspace and installed mod copies. Python syntax compilation passed; no game run or test suite was performed, so the new Lua observation and flee behavior need a short live check.
- The neural navigation PPO model is unchanged. Its next version should add a door interaction action and door-state observation; version the new observation/action schema and train from scratch rather than resuming v2 weights.

## 2026-09-28 navigation + door integration preparation

- Added `python/hybrid_play.py` to load the archived navigation and door PPO policies and run `approach_door → cross_door → navigate_goal` without calling either training environment's reset method.
- Added runtime checks for matching model schemas, observation/action dimensions, navigation arena scale, door scenario identity, and a final goal beyond the trained door target. The runner stops on nearby zombies, low health, invalid door state, focus/telemetry errors, or per-phase step limits.
- Documented the first-run capture and command in `README.md`; updated `DEVELOPMENT_PLAN.md` to separate implementation from pending live verification.
- `..\\.venv\\Scripts\\python.exe -m py_compile hybrid_play.py` passed. No test suite or live game run was performed.
- Next: capture a new radius-8 execution arena within 8 tiles of the trained door, choose a clear goal on the far side, then run one short integration attempt and inspect `summary.json` and `trajectory.jsonl`.

## 2026-09-28 first hybrid live attempt

- The first live attempt reached the door area but hit the approach phase's 100-step limit. The trajectory reached the trained start-side tile briefly, while the runner still required the exact door-target center to be within the door scenario's 0.35-tile goal radius.
- Changed the handoff condition to the door policy's real precondition: the player must occupy the exact adjacent tile on the trained start side. This avoids waiting for an unnecessarily precise center position and still prevents Lua from rejecting a non-adjacent interaction.
- The first run's output folder already existed, so the retry used a new path. Python syntax compilation passed after the fix.
- The next run (`python/runs/hybrid-02/`) succeeded end to end in 58 actions: navigation reached the trained door-side tile, the door policy opened and crossed the door, and navigation reached the final goal. `summary.json` records `status: success`, `opened_by_policy: true`, and `crossed_door: true`.
- The repeat run (`python/runs/hybrid-03/`) also succeeded end to end in 94 actions with the same door, arena, and goal.
- The third run (`python/runs/hybrid-04/`) opened and crossed the door, then stopped after 117 actions because the configured final goal `(2128.5, 6010.5)` was farther than the navigation policy could reliably reach. Its trajectory passed within 0.188 tiles of `(2128.5, 6011.5)` at step 85. The two successful runs also ended inside 0.75 tiles of `(2128.5, 6011.5)`.
- A fourth attempt (`python/runs/hybrid-05/`) again opened and crossed the door but stopped at the farther goal. Its trajectory came within 0.031 tiles of `(2128.5, 6011.5)` at step 54. All four trajectories entered the 0.75-tile completion radius around this nearer point.
- Attempt `hybrid-06` stopped in `approach_door`: the character started near the door, but the navigation policy circled without entering the required tile. The current action snapshot confirms the door is closed/unlocked and that the trained side tile is reachable in the local map.
- Added a short-range grid path to guide the final approach onto the exact trained door-side tile; the learned navigation policy still handles the farther approach. Python syntax compilation passed. Next use `(2128.5, 6011.5)` as the final goal and a fresh output folder to confirm clean termination. No retraining is needed.
