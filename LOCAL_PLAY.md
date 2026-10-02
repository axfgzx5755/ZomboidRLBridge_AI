# API 비용 없는 로컬 자율 탐색

최신 검증 (2026-09-26): 오프라인 unittest 70개와 독립 실행형 검사 3개 통과.
관측한 문의 상태가 관측 범위 밖 건물 연결에도 갱신되도록 수정했습니다.
실게임 자율 탐색 검증은 대기 중입니다. 아래의 테스트 미실행 표기는 최초 구현 당시 기록입니다.

## 방·건물 탐색 확장

Places.lua에서 관측 범위 안의 로드된 타일에 건물·방 식별자와 방 이름을 붙입니다.
기존 SQLite에 places/portals 테이블을 추가하며 기존 방문 기록을 유지합니다.
닫힌 접근 가능 문을 우선 처리하고, 이동 목표는 다음 우선순위로 선택합니다:
미방문 방 → 현재 건물의 미방문 타일 → 미방문 건물 → 기타 미방문 타일.
관측 범위 내 경로를 사용하므로 먼 건물까지의 전역 경로 계획은 아직 없습니다.

매 50스텝과 종료 시 실행 폴더의 exploration.json에 방·건물 진행률을 저장합니다.
known_tiles_covered는 해당 방의 **관측된 이동 가능 타일**을 모두 방문했다는 뜻입니다.
known_area_covered는 알려진 방의 위 조건을 충족하고 알려진 닫힌 문이 없다는 뜻입니다.
새 타일·방·닫힌 문이 발견되면 다시 incomplete가 될 수 있습니다.
잠금·바리케이드 등으로 닫힌 문도 미완료 이유에 포함합니다.
지상층과 주변 관측에 한정하며 건물 전체·윗층·숨겨진 방 탐색 완료를 뜻하지 않습니다.
문 기록은 마지막 관측 기준이므로 범위 밖에서 변경되면 다시 관측하기 전까지 반영되지 않습니다.

사용자가 실행할 확장 회귀 테스트:

    ..\.venv\Scripts\python.exe -m unittest test_places test_local_play test_door test_navigation test_lua_bridge test_held_movement

OK 확인 후 **세이브를 다시 불러와** Places.lua와 ActionBridge.lua 변경을 적용하세요.

    ..\.venv\Scripts\python.exe local_play.py --world RL_Test-character1 --minutes 2 --output runs/local-rooms-01
    Get-Content runs/local-rooms-01/exploration.json

동일한 --world를 유지하면 기존 방문 기록을 활용합니다. 테스트·게임 실행은 작성자가 수행하지 않았습니다.
API 근거: [RoomDef](https://projectzomboid.com/modding/zombie/iso/RoomDef.html),
[BuildingDef](https://projectzomboid.com/modding/zombie/iso/BuildingDef.html).

구현 완료, **사용자 테스트 대기**. 이번 작업에서 테스트·학습·게임 조작은 실행하지 않았습니다.
OpenAI API, API 키, 아스트라 호출, 로컬 언어 모델 설치가 필요하지 않습니다.
Python에서 규칙과 경로 탐색으로 판단하고 SQLite에 방문·실패 기록을 저장합니다.

## 현재 기능과 한계

- 주변 9×9 타일의 실제 게임 지형을 읽고 이동 가능한 경로를 계산합니다.
- 덜 방문한 곳을 목표로 이동하고 접근 가능한 일반 문을 열어 탐색합니다.
- 막힌 이동은 잠시 제외하고 다른 경로를 찾습니다. 문 열기 실패도 일정 시간 재시도를 억제합니다.
- 같은 세이브 식별자로 재실행하면 방문 기록을 재사용합니다.
- 포커스를 잃으면 키를 놓고 대기합니다. 게임 복귀 시 3초 후 새 관측으로 재개합니다.
- 기본 5분 실행, --minutes 0은 시간 제한 없이 실행합니다.

이 버전은 **탐색 자동화**입니다. 전투·아이템 줍기·음식 섭취·수면·장기 생존·사망 후 재시작은 미구현입니다.
화면 이미지 대신 Lua 상태를 관측합니다. 지상층 싱글플레이, 차량 밖 캐릭터, 기본 WASD를 전제로 합니다.
PC가 켜져 있고 게임이 전면에서 정상 속도로 실행되어야 합니다.
기본 체력 80% 미만, 사망, 지원하지 않는 상태, 지속적인 통신 오류에는 중단합니다.
신경망 가중치를 업데이트하지 않으며 PPO 모델도 이 실행 경로에서 사용하지 않습니다.

## 1. 사용자가 실행할 오프라인 테스트

    cd C:\Users\axfgz\Zomboid\ZomboidRL\python
    ..\.venv\Scripts\python.exe -m pip install -r ..\requirements-dev.txt
    ..\.venv\Scripts\python.exe -m unittest test_local_play test_door test_navigation test_lua_bridge test_held_movement

OK를 확인한 다음 실제 게임 점검으로 넘어가세요. 테스트는 모의 객체로 동작하며 게임 키를 누르지 않습니다.

## 2. 짧은 실제 게임 점검

설치 모드와 원본에 ActionBridge.lua를 추가했습니다. 게임 세이브를 다시 불러와 적용하세요.
처음에는 좀비가 없고 문 앞뒤 공간이 확보된 훈련용 세이브에서 확인하세요.
자유 플레이는 순간이동·체력 복구·god/ghost 보호를 켜지 않습니다.
문 열기에는 Lua의 지정 문 상호작용을 사용합니다.
다른 이동·PPO 프로세스는 종료하세요. 공유 잠금으로 동시 조작을 막습니다.

    ..\.venv\Scripts\python.exe local_play.py --world RL_Test-character1 --minutes 1 --output runs/local-check-01

15초 카운트다운 동안 게임을 클릭하고 일시정지를 해제하세요.
step=... action=move/open 출력, 캐릭터 이동, 문 열림을 확인하세요.
게임에서 다른 창으로 전환하면 입력이 멈추고, 다시 게임으로 돌아오면 재개해야 합니다.
status.json의 time_limit은 지정 시간이 끝났다는 뜻이며 학습 성공률을 뜻하지 않습니다.

## 3. 시간 제한 없이 실행

짧은 점검이 정상 동작한 뒤 실행하세요.

    ..\.venv\Scripts\python.exe local_play.py --world RL_Test-character1 --minutes 0 --output runs/local-explore-01

Ctrl+C로 종료합니다. 필요하면 터미널로 전환한 다음 Ctrl+C를 누르세요.
매번 새로운 출력 폴더 이름을 사용합니다. --world에는 **세이브와 캐릭터를 구분할 고유 이름**을 사용하세요.
새 세이브·캐릭터에서는 다른 이름을 지정해야 과거 세계의 기억이 섞이지 않습니다.
세이브 전환을 자동 식별하지 않으므로 세이브를 바꾸기 전에 실행기를 종료하세요.

## 기록과 성장 원리

runs/local-memory.sqlite3에 방문 횟수, 실패 경로의 재시도 제한 시간, 최근 행동 결과를 저장합니다.
현재 관측으로 이동 가능성을 먼저 판단하고 방문 횟수가 적은 목표를 선택합니다.
이동이 반복해서 진행되지 않으면 해당 방향을 60초, 실패한 문은 120초 제외합니다.
환경이 바뀔 수 있어 실패 기록을 영구 금지로 사용하지 않습니다.
세이브별 최근 사건 10,000개를 유지하며 방문 횟수는 누적합니다.

이는 **기억 기반 적응**입니다. 경험이 자동으로 새로운 전투 기술이나 언어 추론 능력을 만들지는 않습니다.
반복 실패가 줄어드는지, 더 많은 지역을 탐색하는지는 실제 게임에서 확인해야 합니다.
일반적인 생존 능력의 향상은 아직 주장할 수 없습니다.

## 오류와 복구

- no fresh action reply: 모드 재로드·일시정지 해제를 확인하세요. 최대 15회 관측 재시도 후 중단합니다.
- another training process owns the game: 기존 학습/제어 프로그램을 종료하세요.
- low_health: 기본 체력 기준에 따라 종료했습니다. 수동으로 상태를 확인하세요.
- 지형에 끼임: 현재 경로를 잠시 제외하여 다른 방향을 시도합니다. 탈출을 보장하지 않습니다.
- 게임 재시작·사망·세이브 교체: 직접 캐릭터를 준비한 뒤 새 실행으로 시작하세요.
- 실행 폴더의 status.json과 게임 console.txt의 오류를 함께 확인할 수 있습니다.

API 키 환경변수가 설정되어 있어도 사용하지 않으며 외부 모델 서버에 관측을 보내지 않습니다.

## Zombie proximity response (added 2026-09-28)

ActionBridge now reports same-floor zombies within 8 tiles. The local explorer gives immediate danger priority: when a zombie is closer than 4.5 tiles, it follows a reachable route that maximizes the closest distance along that route and moves away. It then returns to room exploration when the threat is farther away.

This is a simple reactive retreat, not trained combat or a guarantee of escape. It does not attack, kite by sight, detect zombies beyond the loaded observation range, or account for their future movement. Load the updated `ZomboidRLBridge` mod before trying it. Use a disposable training save first.
