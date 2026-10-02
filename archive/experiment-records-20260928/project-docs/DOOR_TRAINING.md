# 문 열기·통과 학습: 직접 실행 안내

코드 확장 완료. **2026-09-26 오프라인 회귀 테스트 통과. 실제 게임 점검·문 학습은 아직 미검증입니다.**
아래 명령을 사용자가 순서대로 실행하고 결과를 확인해야 합니다.

## 추가된 기능

- 8방향 이동, 정지, 지정 문 열기의 10개 행동. 관측 142개.
- 문 위치·방향·열림·잠김·인접 여부, 이번 에피소드의 열기·통과 기록.
- 닫힌 문을 실제로 열면 최초 1회 +2, 문틈을 통과해 반대편 목표에 도달하면 +10.
  열기 반복은 추가 보상이 없으며 불필요한 상호작용에는 감점.
- 리셋 시 캐릭터 위치·건강·기본 욕구를 복구하고 해당 문을 닫음.
- 체크포인트, 중단 모델 저장, 재개, 별도 평가 및 결과 JSONL.

이번 단계는 **실게임의 일반 단일 문 하나**를 대상으로 합니다. 합성 학습 모드는 없습니다.
기존 navigation-v2 모델은 입력·행동 크기가 달라 사용할 수 없습니다.
처음에는 새 모델을 학습하고 이후 door-v1 모델끼리 이어 학습합니다.

문 상호작용은 E키 입력 대신 Lua에서 지정한 문에 `ToggleDoor(player)`를 호출합니다.
캐릭터가 문 양옆 두 타일 중 하나에 있을 때만 열기를 시도합니다.
잠금 해제, 바리케이드 제거, 이중문·차고문, 창문은 이번 과제에 포함되지 않습니다.
API 참고: [Project Zomboid IsoDoor 공식 문서](https://projectzomboid.com/modding/zombie/iso/objects/IsoDoor.html).
현재 설치된 Build에서의 실제 호환성은 아래 게임 점검으로 확인해야 합니다.

## 1. 터미널 준비와 오프라인 테스트

PowerShell에서:

```powershell
cd C:\Users\axfgz\Zomboid\ZomboidRL\python
..\.venv\Scripts\python.exe -m pip install -r ..\requirements-dev.txt
..\.venv\Scripts\python.exe -m unittest test_door test_navigation test_lua_bridge test_held_movement test_telemetry_lua
```

기존 `.venv`가 없다면 프로젝트 폴더 `ZomboidRL`에서 `python -m venv .venv`를 먼저 실행합니다.
테스트는 모의 Python/Lua 객체를 사용하며 게임을 조작하지 않습니다.
끝에 `OK`가 나와야 합니다. 실패하면 게임 학습으로 넘어가지 말고 실패 출력을 공유하세요.
이 테스트의 통과만으로 실제 게임 호환성이나 학습 성능이 입증되지는 않습니다.

## 2. 게임 준비

1. `ZomboidRLBridge`를 활성화하고 **세이브를 다시 불러옵니다**.
   이번 작업에서 원본과 설치 모드 양쪽에 `DoorTraining.lua` 및 bridge 변경을 반영했습니다.
2. 싱글플레이 훈련용 세이브에서 지상층의 살아 있는 캐릭터를 준비합니다.
   기존 훈련 리셋처럼 위치·건강·욕구가 바뀌고, 훈련 중 god/ghost 보호가 적용됩니다.
3. 잠기지 않고 바리케이드가 없는 일반 단일 문 하나를 고릅니다.
   직접 한 번 열고 닫을 수 있는지 확인합니다. 주변 좀비·가구가 없는 장소가 좋습니다.
4. 문 바로 옆 타일에 서세요. 두 개 이상의 문과 동시에 인접한 위치는 피하세요.
   문 반대편으로 2타일 정도 이동할 공간을 확보합니다.
5. 기본 WASD, 정상 게임 속도, 일시정지 해제를 유지합니다. 다른 제어 스크립트는 종료합니다.

이후 각 명령에는 기본 5초 카운트다운이 있습니다.
**카운트다운 중 게임 창을 클릭하고 일시정지를 해제하세요.**
실행 도중 터미널로 돌아오면 포커스 오류로 중단될 수 있습니다.

## 3. 문과 시작 위치 캡처

```powershell
..\.venv\Scripts\python.exe door_train.py capture --output door-arena-01.json
```

성공하면 `Captured ...`와 문 좌표·목표 좌표가 표시됩니다.
이 단계는 인접 문을 읽고 설정 파일을 만듭니다. 문을 열거나 캐릭터를 리셋하지 않습니다.
`expected exactly one adjacent ordinary door`가 나오면 문 바로 옆으로 이동해 다시 실행하세요.
여기서 인접은 문 경계를 공유하는 두 타일 중 하나라는 뜻입니다. 대각선이나 한 타일 떨어진 위치는 제외합니다.
문을 직접 닫고 문 중앙 바로 앞으로 이동하면 위치를 잡기 쉽습니다.
보완된 오류에는 `player tile`(현재 타일)과 주변 문별 `stand on ... or ...`(허용 타일)이 표시됩니다.
`no IsoDoor within 2 tiles`라면 더 가까이 이동하거나 일반 건물 문으로 바꿔 보세요.
제작 문·모드 문 등은 현재 지원하지 않는 종류일 수 있습니다. 오류 안내 변경도 세이브 재로드 후 적용됩니다.
이미 만들어진 파일을 덮어쓰지 않으므로 재캡처 시 `door-arena-02.json`처럼 새 이름을 사용하세요.

## 4. 짧은 문 열기 점검 — 먼저 이것부터

```powershell
..\.venv\Scripts\python.exe door_train.py check --scenario door-arena-01.json --output runs/door-check-01
```

자동으로 **시작 위치 리셋 → 문 닫힘 → 문 열기 1회 → 다시 리셋·문 닫힘**을 수행합니다.
출력의 `PASS`와 `runs/door-check-01/check.json`을 확인하세요.

- `initial.door_open`: `false`
- `opened.door_open`, `opened.opened_by_agent`: `true`
- `reset.door_open`, `reset.opened_by_agent`: `false`

이 점검은 문 열기·리셋 연결만 확인합니다. 문 통과나 학습 성능 검증은 다음 단계입니다.
문은 마지막 리셋 후 닫힌 상태로 남습니다. 원래 문 상태를 복원하는 기능은 없습니다.

## 5. 짧은 학습 실행

```powershell
..\.venv\Scripts\python.exe door_train.py train --scenario door-arena-01.json --steps 256 --episodes 3 --output runs/door-smoke-01
```

실게임에서 256스텝 학습 후 3회 평가합니다. 무작위 초기 정책이므로 성공률 0%도 가능합니다.
우선 문 리셋, 움직임, 문 열기 시도, 로그 저장과 정상 종료를 확인하세요.
`status.json`의 `complete`는 실행 완료를 뜻하며 과제 학습 성공을 뜻하지 않습니다.

이상이 없다면 저장된 모델에서 학습을 늘립니다:

```powershell
..\.venv\Scripts\python.exe door_train.py train --scenario door-arena-01.json --model runs/door-smoke-01/final.zip --steps 10000 --episodes 10 --output runs/door-train-01
```

요청 스텝은 PPO의 256스텝 단위로 올림될 수 있습니다. 게임 포커스를 유지해야 합니다.
같은 문·시작점 반복 학습이며 다른 문이나 복잡한 맵에 대한 일반화를 보장하지 않습니다.

## 6. 학습을 추가하지 않고 평가

```powershell
..\.venv\Scripts\python.exe door_train.py evaluate --scenario door-arena-01.json --model runs/door-train-01/final.zip --episodes 10 --seed 123 --output runs/door-eval-01
```

학습 직후 평가와 다른 시드로 실행합니다. 문과 목표 자체는 고정되어 있으므로 새로운 지형 평가는 아닙니다.
`summary.json`의 성공률과 `evaluation.jsonl`의 다음 필드를 확인하세요:

- `opened_by_agent`: 정책이 문 열기에 성공했는지
- `crossed_door`: 문이 열린 뒤 실제 문틈을 시작 방향에서 반대편으로 통과했는지
- `is_success`: 위 두 조건과 반대편 목표 도달까지 충족했는지
- `terminal_reason`: `success`, `stalled`, `max_steps`, `out_of_bounds`, `dead`, `error` 등

문을 열었다고만 성공하지 않으며, 옆길로 돌아 목표에 도착해도 문 통과가 없으면 성공하지 않습니다.
통과 판정은 연속 관측 위치 사이의 선분을 사용하므로 비정상적인 순간이동이 없는 환경에서 실행하세요.

## 중단·재개·오류

- 매 실행의 출력 폴더는 새 이름이어야 합니다. 재실행 시 `-02`, `-03` 등을 붙이세요.
- Ctrl+C로 중단할 수 있습니다. 게임 포커스를 잃어도 오류로 중단하고 키를 해제합니다.
- 학습 모델이 준비된 뒤 중단되면 `interrupted.zip`과 같은 이름의 JSON을 저장합니다.
  `--model runs/해당폴더/interrupted.zip`으로 새 출력 폴더에 재개하세요.
- 모델 ZIP과 동명 JSON, 원래 `door-arena-01.json`을 함께 보관하세요.
- `acknowledgement timed out`: 모드 재로드, 일시정지, 캐릭터 로딩 여부를 확인합니다.
- `use an unlocked...`: 잠금·바리케이드·파손·문 종류를 확인합니다.
- `unsupported training door: ...`: 표시된 이유(locked, locked-by-key, barricaded, destroyed, double, garage)를 확인합니다.
- `Object tried to call nil in suitable`: 기존 문 속성 검사에서 Build 42에 맞지 않는 `Is()` 호출이 원인이었습니다.
  `has()`로 수정했으며, 세이브를 다시 불러온 뒤 캡처하세요. 수정 회귀 테스트는
  `..\.venv\Scripts\python.exe -m unittest test_door`로 직접 실행할 수 있습니다.
- `clear the target tile...`: 문 반대편 목표 타일의 가구·장애물을 치우거나 다른 문을 캡처합니다.
- `door did not close`: 문을 막는 물체나 다른 캐릭터를 확인합니다.
- 게임/Lua 오류 시 `C:\Users\axfgz\Zomboid\console.txt`의 관련 오류와
  출력 폴더의 `status.json`, `check.json` 또는 `evaluation.jsonl`을 공유하면 원인을 확인할 수 있습니다.

게임 상태 전체를 되감지는 않습니다. 훈련 종료 시 키와 lease는 정리하지만 위치·욕구·문 상태는 그대로 남습니다.
