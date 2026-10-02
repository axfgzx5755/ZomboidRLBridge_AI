# 학습 품질 점검과 소량 데이터 증강

점검일: 2026-09-30. 범위: 이미 구현한 이동 PPO, 문 PPO, 두 정책의 연결 실행.
전투·아이템·장기 생존 학습은 이번 변경에 포함하지 않는다.

후속 요청으로 수행한 **CNN 대 MLP 구조 비교**는 [CNN_COMPARISON.md](CNN_COMPARISON.md)에
별도로 정리했다. 아래 증강 실험과 조건·목적이 다르므로 결과를 구분한다.

## 확인 결과

기존 코드는 사진의 7단계에 대체로 대응한다. 다만 **지도학습 분류기가 아니라 PPO 강화학습**이다.
학습/평가 분리, 모델 상태 기록, 증강과 대조 실험을 보강했다.
기존 보관 모델과 실행기의 기본 모델 경로는 유지했다.

| 사진 단계 | 현재 구현과 이번 보강 |
|---|---|
| 1. 패키지 가져오기 | NumPy, Gymnasium, PyTorch, Stable Baselines3. 확인한 환경: Torch 2.14.0+cpu, SB3 2.9.0 |
| 2. 학습/시험 데이터 | 관측→행동→보상→다음 관측을 수집한다. 정답 클래스 대신 return/advantage를 계산한다. 합성 이동은 지형·목표를 seed로 생성, 문은 실제 게임 관측을 사용한다. 검증 seed와 최종 시험 seed를 분리했다 |
| 3. 신경망 구조 | 별도 actor/critic MLP, 각각 64→64 Tanh. actor는 행동 확률, critic은 기대 누적 보상 1개 출력 |
| 4. 학습 변수 | PPO clipped policy loss + 0.5×value loss + entropy 항, Adam. 사진의 LR=0.8, BCE, 2000 epochs, SGD를 그대로 사용하지 않는다 |
| 5. 학습 | rollout 256개, minibatch 64개. 역전파·Adam 갱신은 SB3가 수행한다. 현재 정책으로 수집한 관측/행동/log probability의 대응을 유지한다 |
| 6. 평가 | 증강 없는 원래 좌표계, deterministic 행동. 성공률, 실패 원인, 스텝, 문 열기/통과율, 문별 결과와 Wilson 95% 구간 |
| 7. 저장/로드 | ZIP+동명 JSON, 설정, 학습 CSV, 에피소드 JSONL, 학습 전후 신경망 상태. 새 출력 폴더를 요구해 기존 결과를 보존한다 |

구현 근거: [SB3 PPO 2.9.0 공식 문서](https://stable-baselines3.readthedocs.io/en/v2.9.0/modules/ppo.html).
강화학습 데이터 증강의 연구 배경: [RAD 원 논문](https://arxiv.org/abs/2004.14990).
아래 x/y 변환은 이 저장소의 관측에 맞춘 구현이며, RAD의 성능을 재현했다고 주장하지 않는다.

## 실제 모델 상태

ZIP을 직접 읽고 가중치와 Adam 상태에 NaN/Inf가 없는지 검사했다.
결과: [audit.json](python/runs/quality-audit-20260930/audit.json).

| 항목 | 이동 | 문 |
|---|---:|---:|
| 입력 / 행동 수 | 134 / 9 | 142 / 10 |
| 학습 가능한 파라미터 | 26,250 | 27,339 |
| 누적 환경 스텝 | 162,816 | 10,496 |
| 가중치·optimizer 유한성 | 정상 | 정상 |
| 마지막 기록 explained variance | 0.783 | -0.618 |
| 마지막 기록 approximate KL | 0.0172 | 0.0102 |
| 마지막 기록 clip fraction | 0.124 | 0.068 |

기존 설정은 LR 0.0003, epochs 10, gamma 0.99, GAE lambda 0.95, clip 0.2,
gradient norm 상한 0.5. entropy 계수는 이동 0.01, 문 0.02이며 target KL 제한은 없었다.

문 critic의 음수 explained variance는 해당 rollout에서 보상 예측이 평균 상수 예측보다
좋지 않았음을 뜻한다. 이것만으로 행동 정책이 실패했다거나 신경망이 손상됐다고 판정하지 않는다.
기존 CSV는 SB3의 기록 시점 때문에 마지막 optimizer 갱신을 누락할 수 있다.
이번 코드에서는 학습 종료 후 남은 로그도 기록한다.

기존 문 평가는 같은 문·시작점 30/30, 평균 8.67스텝이다.
명목 Wilson 구간은 약 88.65~100%지만, 같은 조건의 반복은 새 문에서의 일반화 증거가 아니다.
저장 ZIP에는 과거 gradient가 저장되지 않는다. 새 학습의 `health_after.json`에만 마지막
clipping 후 gradient norm/유한성을 기록하며, 매 minibatch의 gradient 이력을 측정하지는 않는다.

## 각 데이터의 학습 방식

| 데이터/기능 | 학습 방식 | 해석과 제한 |
|---|---|---|
| 합성 이동 지형 | stage 0~2의 지형과 도달 가능한 목표를 생성해 PPO rollout 수집 | 기존부터 데이터 다양화가 있음. seed 분리만으로 실제 게임 전이를 보장하지 않음 |
| 실게임 이동 | 텔레메트리·벽 관측과 이동 보상으로 PPO 학습/평가 | 합성과 동일한 관측 구조, 실게임 별도 평가 필요 |
| 문 시나리오 JSON | 문·시작 타일·목표를 정의하는 설정 | JSON 1개가 학습 샘플 1개는 아님. reset 후 실제 행동 전이들이 학습 데이터 |
| 문 PPO | 최초 열기 +2, 실제 문 경계 통과 후 목표 도달 +10, 거리 개선과 실패 감점 | 파일명/seed를 바꿔도 같은 물리적 문이면 독립된 새 문 데이터가 아님 |
| 학습 CSV / 평가 JSONL | 학습 및 결과 측정 기록 | 과거 로그를 복사해 PPO 재학습 데이터로 넣지 않음 |
| 로컬 탐색 / hybrid 연결 | 규칙 기반 탐색·정책 전환, 이미 학습된 PPO 추론 | 실행 횟수가 늘어도 가중치는 갱신되지 않음 |

현재 `door-arena-01.json`은 (2042,6004), `door-arena-03.json`은 (2128,6014)이다.
두 문은 약 86.6타일 떨어져 있어 현재 bridge의 50타일 리셋 제한에 맞지 않는다.
이 두 파일을 바로 한 학습 명령에 섞지 않는다. 기존 문 주변에서 검증된 가까운 문을 확보해야 한다.

## 추가한 증강

`--augment-geometry`를 지정한 **학습에서만**, 에피소드마다 확률 50%로 x/y 좌표계를 교환한다.

- 목표 상대 좌표, 속도, 타일 내부 위치, 5×5 점유/벽 격자를 함께 변환한다.
- 벽 N/E/S/W 채널은 W/S/E/N으로 변환한다.
- 문 상대 좌표와 north/west 방향을 함께 변환한다.
- 정책이 선택한 이동을 원래 세계 방향으로 되돌려 실행한다. 정지와 문 열기는 그대로다.
- 보상, 성공 조건, 실제 게임 지도는 유지한다. 평가와 배포는 원래 좌표계를 사용한다.
- PPO의 과거 rollout을 2배 복제하지 않는다. 같은 물리 조건을 두 좌표 표현으로 경험하게 한다.

좌표 교환은 정확히 타일 경계에 있는 위치도 보존한다. 회전/부호 반전은 경계와 문 상태를
잘못 변환할 가능성이 있어 이번 구현에서는 제외했다. 이진 벽/문 값에 임의 노이즈도 넣지 않는다.
기존 입력/행동 schema와 MLP 크기를 유지해 모델을 이어서 학습할 수 있다.
새 건물·장애물 구조·반대 접근면 경험은 실제 시나리오를 더 확보해야 한다.

## 실행한 대조 실험

기존 이동 모델에서 각 실험을 시작했다. 학습 seed 43/44/45, 각각 추가 4,096스텝,
증강 OFF/ON을 같은 학습량으로 비교했다. 총 추가 학습량 24,576스텝.
두 조건 모두 LR 0.0001, epochs 5, target KL 0.015, entropy 0.01로 동일하게 설정했다.
이 설정은 이미 학습된 정책을 보수적으로 조정하기 위한 실험값이며 확정된 최적값은 아니다.

검증 100개 seed(1100000~1100099)로 후보를 먼저 선택했고, 이후 독립 시험
300개 seed(1200000~1200299)를 모든 후보에 동일하게 적용했다. 모두 합성 stage 2이다.

| 모델 | 검증 성공률 | 독립 시험 성공 | 시험 평균 스텝 |
|---|---:|---:|---:|
| 기존 기준 모델 | 81% | 257/300 (85.67%) | 13.03 |
| 대조군 seed 43 | 85% | 249/300 (83.00%) | 13.47 |
| 증강 seed 43 | 87% | 252/300 (84.00%) | 13.06 |
| 대조군 seed 44 | 81% | 246/300 (82.00%) | 13.99 |
| 증강 seed 44 | 82% | 250/300 (83.33%) | 13.98 |
| 대조군 seed 45 | 81% | 249/300 (83.00%) | 13.84 |
| 증강 seed 45 | 85% | 253/300 (84.33%) | 13.01 |

증강군 평균은 83.89%, 대조군 평균은 82.67%로 1.22%p 차이다.
동일한 시험 지형을 재사용했으므로 이를 900개의 독립 지형 평가로 취급하지 않는다.
3개 학습 seed의 작은 차이만으로 통계적으로 확실한 개선이나 데이터 절감 배수를 주장할 수 없다.
검증으로 선택된 `augmented_43`도 시험에서는 기존 모델보다 낮다.
**기존 기준 모델을 유지한다. 이번 실험에서는 기존 모델 대비 품질 향상이 입증되지 않았다.**

결과와 후보 모델: [summary.json](python/runs/quality-pilot-20260930/summary.json),
[선택 기록](python/runs/quality-pilot-20260930/selection.json).
시험 seed는 이제 확인한 데이터이므로 후속 모델 튜닝의 검증 데이터로 사용할 수는 있어도,
다음 최종 시험에는 새로운 seed 구간을 지정해야 한다.

## 메트릭스와 결과물 산출 구조

`관측/보상 → PPO rollout → advantage/return → minibatch 역전파 → ZIP 저장 → 독립 평가 → JSON 요약`

| 산출물 | 주요 내용 / 사용법 |
|---|---|
| `config.json` | 모델 출처, seed, 요청 스텝, 시나리오, 증강 여부, 명시적 PPO 설정 변경 |
| `episodes.monitor.csv` | 학습 중 episode 보상·길이·성공·종료 원인. 독립 평가 점수가 아님 |
| `progress.csv` | policy/value loss, entropy loss, approximate KL, clip fraction, explained variance, 실제 timestep |
| `health_before/after.json` | 네트워크 구조·파라미터 수·유한성·실제 PPO 설정, 사용 가능한 마지막 gradient |
| `final.zip` + `final.json` | optimizer를 포함한 모델과 schema/시나리오/증강 설정. 이동 cycle 모델도 같은 형태 |
| `evaluation.jsonl` | episode seed·보상·스텝·종료 원인·최종 위치. 문은 scenario index, door identity, 열기/통과 여부 |
| 문 `summary.json` | 성공률·Wilson 구간·실패 원인·평균/95백분위 스텝·열기/통과율·문별 결과·macro 평균·문 평가 coverage |
| 이동 `cycle_NNNN_evaluation.json` | 해당 cycle 평가 집계. 평가 전용 명령에서는 `summary.json` |
| `status.json` | 실행 완료/중단/오류. `complete`는 과제 성능 통과 판정이 아님 |

손실 하나만 낮추는 것을 목표로 하지 않는다. KL 급증, 높은 clip fraction, entropy 감소와
성공률 하락이 함께 나타나는지 확인한다. explained variance는 critic 진단에 사용한다.
문 열기율만 높고 통과율이 낮으면 통과 동작/충돌을, 통과율은 높고 성공률이 낮으면
반대편 목표 도달을 살핀다. hybrid는 별도 실행의 단계별 실패와 전체 목표 달성으로 평가한다.
문별 평가는 순서대로 균등 배정하며, 평가 횟수를 문 개수의 배수로 잡는다.

## 다음 문 학습 실행 순서

현재 게임 프로세스가 없어 이번 세션에서는 문 실게임 학습을 실행하지 않았다.
문 모델을 키우거나 이동 모델과 가중치를 합치지 않고, 기존 64×64 구조로 먼저 비교한다.

1. [문 준비/연결 점검](DOOR_TRAINING.md)의 게임 준비와 `check`를 수행한다.
2. 가까운 문들을 캡처했다면 물리적 문 단위로 학습/검증/최종 시험을 나눈다.
   같은 문의 반대편 캡처나 증강본을 시험용 새 문으로 세지 않는다.
3. 기존 모델로 검증용 문을 먼저 평가한다. 현재 03만 있다면 같은 문 회귀 점검만 가능하다.
4. 아래 짧은 추가 학습을 새 폴더에 실행한다. 실게임에서는 게임 포커스·일시정지 해제를 유지한다.
5. 기존/후보를 같은 검증 시나리오·같은 횟수로 비교한다. 개선이 없으면 기존 모델을 유지한다.
   최종 시험용 문은 설정을 결정한 뒤 한 번 평가하고, 별도로 hybrid 연결을 확인한다.

PowerShell, `ZomboidRL/python`에서 실행:

```powershell
# 기존 문만으로 증강 파이프라인을 먼저 확인하는 짧은 실행
..\.venv\Scripts\python.exe door_train.py train --scenario door-arena-03.json --model ..\archive\experiment-records-20260928\door\door-single-v1-20260928\training\final.zip --augment-geometry --learning-rate 0.0001 --epochs 5 --target-kl 0.015 --entropy-coef 0.02 --steps 2048 --episodes 30 --output runs/door-quality-01

# 추가 학습 없이 원래 좌표계에서 평가
..\.venv\Scripts\python.exe door_train.py evaluate --scenario door-arena-03.json --model runs/door-quality-01/final.zip --episodes 30 --seed 123 --output runs/door-quality-eval-01
```

위 명령은 **같은 문의 회귀/연결 확인용**이다. 새로운 문 학습이 준비되면 `--scenario`를 반복 지정하되
재개 모델이 배운 원래 시나리오도 포함한다. 검증/시험 문은 train 명령에 넣지 않는다.
증강 자체의 효과를 보려면 원래 ZIP에서 똑같이 시작해 `--augment-geometry`만 제외한 대조군도 실행한다.
평가 모드에서는 증강/학습 설정 옵션을 거부한다. 재개 학습 옵션은 저장 모델에 실제 반영되며,
미지정 옵션은 기존 모델의 값을 유지한다. `health_before.json`의 실제 설정을 확인한다.

## 오프라인 재현 명령

```powershell
..\.venv\Scripts\python.exe -m unittest discover
..\.venv\Scripts\python.exe quality_audit.py --models ..\archive\experiment-records-20260928\door\door-single-v1-20260928\training\final.zip ..\archive\experiment-records-20260928\obstacle-navigation\navigation-v2-obstacles-20260918\latest.zip --output runs/quality-audit-next
..\.venv\Scripts\python.exe quality_experiment.py --model ..\archive\experiment-records-20260928\obstacle-navigation\navigation-v2-obstacles-20260918\latest.zip --validation-seed 1300000 --test-seed 1400000 --output runs/quality-pilot-next
```

출력 경로는 새 이름으로 사용한다. 테스트는 좌표/벽/행동 변환, 실제 합성 충돌·보상·종료의
대응, 문 경계 통과, 균등 평가, 모델 재개 옵션, 학습/저장/로드를 확인한다.
실게임 일반화나 장기 학습 성능을 단위 테스트로 입증하지 않는다.
