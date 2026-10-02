# CNN과 MLP의 실제 학습 비교

**후속 검증 완료:** [새 seed 10개 확인 실험](CNN_CONFIRMATION.md)에서 MLP 16.82%,
CNN 64.60%, 평균 차이 +47.78%p(95% 구간 +42.09~+53.47%p)였다.
아래는 최초 3개 seed 탐색 실험의 기록이며 후속 결과와 합산하지 않는다.

2026-09-30. 이미 구현된 합성 이동 환경에서 직접 학습하고 평가했다.
실게임 문 열기, 게임 화면 이미지 인식, 전투 성능으로 확대 해석하지 않는다.

## 결과: 평균은 개선, 반복 간 편차가 큼

| 학습 seed | MLP 성공 / 500 | CNN 성공 / 500 | CNN − MLP |
|---|---:|---:|---:|
| 71 | 164 (32.8%) | 37 (7.4%) | -25.4%p |
| 72 | 105 (21.0%) | 364 (72.8%) | +51.8%p |
| 73 | 62 (12.4%) | 349 (69.8%) | +57.4%p |
| 세 seed 평균 | **22.07%** | **50.00%** | **+27.93%p** |

계산: `((7.4−32.8) + (72.8−21.0) + (69.8−12.4)) / 3 = 27.93%p`.

| 비용/진행 지표 | MLP | CNN |
|---|---:|---:|
| 모델당 학습 스텝 | 32,768 | 32,768 |
| 평균 학습 경과 시간 | 31.62초 | 44.97초 |
| 시간 비율 | 1.00 | 1.42 |
| 평가 평균 에피소드 길이 | 35.43스텝 | 22.13스텝 |

에피소드 길이는 성공과 실패를 모두 포함한다. 모든 모델의 가중치·optimizer·마지막
gradient 유한성 검사를 통과했다. CNN 첫 합성곱 가중치의 실제 갱신과 checkpoint
저장/로드도 테스트로 확인했다. 전체 unittest 83개 통과.

**이번 예산에서 관측된 평균은 CNN이 높다. 안정적인 우월성은 아직 확인하지 못했다.**
첫 seed에서는 CNN이 크게 낮았고, paired 학습 seed 차이의 t 95% 구간은
약 **-87.01~+142.88%p**다. 표본 3개의 선형 t 구간이라 가능한 확률 차이 범위를
넘을 정도로 넓으며, 신뢰할 만한 개선 크기를 추정하기에 반복 수가 부족하다는 의미다.
개별 지형 500회를 합쳐 학습 seed 편차를 감추지 않는다.

참고로 기존 **162,816스텝** 이동 MLP도 같은 시험 지형에서 별도로 평가했다:
**421/500 = 84.2%**, 평균 13.54스텝.
[기존 모델 평가](python/runs/cnn-existing-reference-20260930/summary.json).
학습량과 이력이 다르므로 이 수치를 구조 간 공정 비교 표에 섞지 않는다.
실사용 기준 모델은 그대로 유지한다.

다음 판단에 필요한 것은 고정된 구조/설정으로 학습 seed 반복 수를 늘려 초기화 민감도를
확인하는 일이다. 장기 성능을 비교하려면 두 구조에 동일하게 더 큰 학습 예산을 주어야 한다.
현재 결과는 그 추가 실험이나 실게임 문 학습 결과를 대신하지 않는다.

## 비교 조건

- 두 모델 모두 무작위 가중치에서 시작. 기존 학습 가중치는 사용하지 않음.
- 학습 seed 71, 72, 73으로 구조별 3회, 총 6개 모델을 학습.
- 모델마다 32,768스텝: stage 0에서 8,192, stage 1에서 8,192, stage 2에서 16,384.
- 승급 시점은 평가 점수와 무관하게 고정. 추가 데이터 증강은 두 모델 모두 OFF.
- 동일한 PPO 설정: Adam, LR 0.0003, rollout 256, batch 64, epochs 10,
  gamma 0.99, GAE lambda 0.95, clip 0.2, entropy 0.01, value 계수 0.5,
  gradient norm 상한 0.5, target KL 미지정.
- 최종 checkpoint만 평가. 시험 결과로 구조/설정을 바꾸거나 checkpoint를 선택하지 않음.
- 시험: stage 2의 새 seed 2100000~2100499, 동일한 500개 지형/목표,
  deterministic 행동. 환경·보상·관측 134개·행동 9개는 동일.
- CPU 1 thread, deterministic PyTorch 연산. 구조별 실행 순서를 번갈아 배치.

총 학습 전이는 196,608개, 평가 실행은 3,000회다. 평가 지형 500개는 모든 모델이
공유하므로 3,000개의 독립 지형으로 계산하지 않는다.

## 구조

| 항목 | 기존 방식 MLP | 비교용 CNN+MLP |
|---|---|---|
| 주변 지형 입력 | 5×5×5 격자를 펼친 125개 값 | 동일 값을 채널 5×높이 5×너비 5로 재배열 |
| 지형 처리 | 다른 9개 값과 함께 MLP 입력 | Conv2d(5→8, 3×3) → ReLU → Conv2d(8→8, 3×3) → ReLU → Flatten → Linear(200→48) → ReLU |
| 좌표·체력·속도 등 | 입력에 포함 | CNN 특징 48개와 나머지 9개를 연결 |
| actor / critic | 각각 64→64 Tanh | 각각 64→64 Tanh |
| 출력 | 행동 9개 확률 / 가치 1개 | 동일 |
| 파라미터 | 26,250 | 26,994 (+2.83%) |

CNN 특징 추출기는 actor와 critic이 공유한다. 이는 입력 전체를 그림으로 간주하는
방식이 아니라, 실제 공간 구조가 있는 주변 격자에만 합성곱을 적용하는 구조다.
코드의 `MlpPolicy` 이름은 SB3의 정책 인터페이스 이름이며, `features_extractor_class=GridCNN`에
실제 PyTorch `Conv2d` 두 층이 들어간다.
구현 방식은 [SB3 공식 custom feature extractor 문서](https://stable-baselines3.readthedocs.io/en/v2.9.0/guide/custom_policy.html)를 따른다.

## 계산 방법과 해석 범위

- seed별 성공률 = 해당 모델의 성공 에피소드 수 ÷ 500.
- 평균 차이(%p) = 세 seed의 `(CNN 성공률 − MLP 성공률)` 평균 × 100.
- 불확실성: 세 paired 학습 seed 차이에 대해 Student t 95% 구간을 계산한다.
  자유도 2인 작은 실험이며, 구간은 이번 시험 지형 500개를 고정한 조건에서의 값이다.
- 같은 seed 내에서 CNN만 성공/MLP만 성공한 지형 수도 기록한다.
  개별 binomial p 값은 참고용이며 다중 비교 보정이나 전체 일반화 검증을 대체하지 않는다.
- 시간은 PPO `learn()` 구간의 실제 경과 시간이다. 모델 초기화·평가·파일 저장은 제외한다.
  같은 스텝 수 비교이며 같은 벽시계 시간 예산 비교는 아니다.
- 이 실험은 **선택한 작은 CNN 구조와 기존 MLP가 동일한 설정/학습량에서 어떻게 다른지**를
  측정한다. 모든 CNN의 우월성, 두 구조 각각의 최적 성능, 장기 수렴 성능을 판정하지 않는다.

## 코드와 재현

- CNN: [grid_cnn.py](python/grid_cnn.py)
- 실험/paired 통계: [cnn_experiment.py](python/cnn_experiment.py)
- 입력 채널·역전파·저장/로드·통계 테스트: [test_grid_cnn.py](python/test_grid_cnn.py)
- 실행 설정: [config.json](python/runs/cnn-vs-mlp-20260930/config.json)
- 전체 비교: [summary.json](python/runs/cnn-vs-mlp-20260930/summary.json)
- 모델별 `final.zip`, `health_before/after.json`, `progress.csv`, `evaluation.jsonl`, `summary.json`은
  `python/runs/cnn-vs-mlp-20260930/{mlp,cnn}_{71,72,73}/`에 저장.

PowerShell, `ZomboidRL/python`에서:

```powershell
..\.venv\Scripts\python.exe -m pip install -r ..\requirements-dev.txt
..\.venv\Scripts\python.exe -m unittest test_grid_cnn
..\.venv\Scripts\python.exe cnn_experiment.py --output runs/cnn-vs-mlp-reproduce
```

재현은 같은 시험 seed를 사용한다. 이 결과를 보고 설정을 튜닝한다면 최종 검증 때
`--test-seed`로 새로운 구간을 지정한다. 기존 실게임 실행기의 모델 경로는 바꾸지 않았다.
