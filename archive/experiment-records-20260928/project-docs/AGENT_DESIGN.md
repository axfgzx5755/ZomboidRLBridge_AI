# 자율 플레이와 성장 설계

업데이트: API 비용 없는 규칙 기반 자율 탐색을 구현했다. 사용 방법은 [LOCAL_PLAY.md](LOCAL_PLAY.md).
local_play.py, local_explorer.py, ActionBridge.lua가 실행·판단·관측을 담당한다.
방문 기록과 실패 재시도 제한을 SQLite에 저장한다. 테스트는 사용자 실행 대기 상태다.
아래 아스트라/모방학습 설계는 여전히 미구현이며 현재 로컬 실행기는 이를 필요로 하지 않는다.

상태: 설계 문서. 아스트라 호출, 자율 실행 관리자, 경험 저장소는 아직 구현되지 않았다.
현재 실행 가능한 학습기는 기존 PPO 이동·문 학습기다.

## 실행 과정

사용자 목표 → 화면·게임 상태 관측 → 관련 경험 검색 → 아스트라 행동 선택
→ 로컬 실행기 → 실제 결과 확인 → 경험 저장 → 다음 판단.

아스트라는 문 열기, 가까운 좌표로 이동 같은 짧은 목표를 선택한다.
로컬 실행기는 키 입력과 게임 API를 담당한다. 시간 제한·포커스 상실·중단 시 입력을 해제한다.
자유 플레이에는 훈련용 순간이동·무적·건강 복구가 없는 별도 실행 경로가 필요하다.
아직 play.py 실행 명령이나 모델 API 연결은 없다.

## 성장 과정과 원리

1. 경험 기억: 관측, 행동, 실제 결과, 시각, 세이브 및 대상 식별자를 저장한다.
   예를 들어 문이 잠겨 있음을 확인하면 같은 시도를 반복하는 대신 다른 출입구를 찾는다.
   현재 상황과 관련된 기록을 다음 판단의 입력에 포함하는 방식이다. 모델 가중치 업데이트는 아니다.
   상태는 변할 수 있으므로 과거 기억보다 최신 관측을 우선하고 다른 세이브의 기록은 분리한다.
2. 성공 절차 재사용: 접근 → 열림 확인 → 통과 확인처럼 검증된 행동 순서를 재사용한다.
   적용 조건과 각 단계의 결과를 확인하며 한 번의 성공을 모든 장소에 일반화하지 않는다.
3. 로컬 정책 학습: 필요하면 성공 기록으로 모방학습을 하고 별도 훈련 환경에서 PPO로 개선한다.
   PPO는 관측·행동·보상을 이용해 행동 선택 확률과 가치 예측을 업데이트한다.
   현재 ZIP에 저장되는 것은 이 로컬 정책이며 아스트라 자체의 가중치가 아니다.
   모방학습과 자율 플레이 기록의 자동 학습 연결은 아직 구현되지 않았다.

로그가 늘어나는 것만으로 실력이 향상되지는 않는다.
같은 조건에서 성공률, 반복 실패 횟수, 완료 시간, 호출 수를 비교한다.
새 장소에서도 평가하고 모델의 성공 주장 대신 게임 상태로 성공 여부를 확인한다.
첫 목표는 문 하나 자율 통과, 이후 여러 출입구 탐색과 아이템 상호작용으로 확장한다.

## 파일 정리 조사 (2026-09-20)

삭제 후보: 프로젝트 내부 examplemod(미사용 예제 모드), python/__pycache__(재생성 가능한 캐시).
이 두 경로의 재귀 삭제 명령은 자동 승인 검토에서 blocked by policy로 거부되었다.
따라서 이번 작업에서는 파일을 삭제하지 않았다.

보존 대상: 실게임 세이브, 가상환경, 설치 RL 모드, 학습 결과, 테스트 및 현재 의존 모듈.
implementation-smoke, implementation-resume, navigation-v2-smoke 결과는 후속 실행의
config.json에 학습 출처로 참조되어 있어 더미로 취급하지 않는다.
기존 모듈도 door_train → door_env/navigation/training_env 등의 의존성이 남아 있다.
새 실행 경로 구현 후 공통 기능을 분리하고 참조가 사라진 파일을 추가 정리해야 한다.
사용자 요청에 따라 테스트·학습·게임 조작은 실행하지 않았다.

## Next capability bundle

The current 134-input PPO model remains obstacle navigation only. The separate local explorer already opens supported unlocked doors and now also senses nearby zombies and retreats when one is within 4.5 tiles. It does not update or use PPO weights for either behavior.

The next training-model expansion should add door state and an interaction action to the observation/action space, followed by a multi-door synthetic curriculum and separate live evaluation. Windows, stairs, inventory and combat need their own game-state actions and reset/evaluation support before they can be safely trained. Do not resume a v2 checkpoint after changing its observation or action schema; train a versioned v3 policy.
