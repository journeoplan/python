# Breath & Vital Sensing — Claude Code Project Memory

## 프로젝트 개요

WiFi CSI 및 카메라 기반 생체신호 감지 시스템.
카메라 호흡 감지(Phase 1) → WiFi CSI 호흡 감지(Phase 2) → 자세 추정(Phase 3) 순서로 개발한다.

**현재 단계**: Phase 1 — 카메라 기반 호흡 감지

---

## 디렉토리 구조

```
python/
├── CLAUDE.md                  ← 이 파일 (Claude Code 프로젝트 메모리)
├── .claude/
│   ├── skills/                ← 재사용 가능한 태스크 정의
│   │   ├── add-module.md
│   │   ├── run-test.md
│   │   ├── signal-debug.md
│   │   └── new-sensor.md
│   └── settings.json
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── config.py              ← 전역 설정 (FPS, 주파수 범위 등)
│   │
│   ├── sensors/               ← 입력 소스 (카메라 / WiFi CSI)
│   │   ├── __init__.py
│   │   ├── base_sensor.py     ← 추상 베이스 클래스
│   │   ├── camera_sensor.py   ← MediaPipe 기반 카메라
│   │   └── wifi_sensor.py     ← WiFi CSI (Phase 2에서 구현)
│   │
│   ├── signal/                ← 신호 처리 파이프라인
│   │   ├── __init__.py
│   │   ├── filter.py          ← Butterworth, Kalman 필터
│   │   ├── fft_analyzer.py    ← FFT + 피크 감지
│   │   └── smoother.py        ← BPM 평활화
│   │
│   ├── detectors/             ← 감지 로직
│   │   ├── __init__.py
│   │   ├── breath_detector.py ← 호흡 BPM
│   │   └── pose_detector.py   ← 자세 추정 (Phase 3)
│   │
│   └── ui/
│       ├── __init__.py
│       ├── overlay.py         ← OpenCV 오버레이
│       └── dashboard.py       ← 실시간 그래프
│
├── tests/
│   ├── test_filter.py
│   ├── test_fft.py
│   └── test_camera_sensor.py
│
├── scripts/
│   ├── run_camera.py          ← 카메라 모드 실행 진입점
│   ├── run_wifi.py            ← WiFi 모드 실행 진입점
│   └── collect_data.py        ← 학습 데이터 수집
│
└── data/
    ├── recordings/            ← 녹화된 세션
    └── models/                ← 학습된 모델 가중치
```

---

## 핵심 설계 원칙

### 센서 추상화
`BaseSensor` 추상 클래스를 반드시 상속해야 한다.
카메라와 WiFi CSI는 동일한 인터페이스로 신호 처리 파이프라인에 데이터를 공급한다.

```python
# 모든 센서가 지켜야 할 인터페이스
class BaseSensor(ABC):
    def start(self) -> None: ...
    def read(self) -> np.ndarray: ...   # shape: (n_channels,)
    def stop(self) -> None: ...
    def fps(self) -> float: ...
```

### 신호 처리 파이프라인 (불변 원칙)
```
raw signal → DC 제거 → Butterworth 밴드패스 → FFT → 피크 감지 → BPM
```
이 파이프라인은 카메라 / WiFi CSI 양쪽에서 동일하게 사용한다.
`src/signal/` 모듈을 절대 센서별로 분기하지 말 것.

### 주파수 범위 상수 (config.py에서만 수정)
```python
BREATH_HZ_MIN = 0.1   # 6 BPM
BREATH_HZ_MAX = 0.6   # 36 BPM
HEART_HZ_MIN  = 0.8   # 48 BPM
HEART_HZ_MAX  = 2.5   # 150 BPM
```

---

## 개발 규칙

### Python 스타일
- Python 3.11+ 사용
- 타입 힌트 필수: `def func(x: np.ndarray) -> float:`
- docstring: Google 스타일
- 라인 길이: 100자
- 포매터: `black`, 린터: `ruff`

### 테스트
- 모든 `src/signal/` 함수는 단위 테스트 필수
- `pytest tests/ -v` 로 실행
- 신호 처리 함수는 합성 데이터(사인파)로 검증

### 커밋 메시지
```
feat: 카메라 센서 어깨 Y좌표 추출 추가
fix: FFT 피크 감지 엣지케이스 수정
refactor: 필터 파라미터 config.py로 이동
test: Butterworth 필터 주파수 응답 테스트 추가
```

### 브랜치 전략
```
main          ← 안정 버전
dev           ← 통합 브랜치
feat/phase-1  ← 카메라 기능 개발
feat/phase-2  ← WiFi CSI 기능 개발
```

---

## 환경 설정

```bash
# 가상환경
python -m venv venv
source venv/bin/activate   # macOS/Linux

# 의존성 설치
pip install -r requirements.txt

# 실행
python scripts/run_camera.py
```

---

## 현재 알려진 이슈 / 제약사항

- MediaPipe Pose는 어깨가 프레임에 절반 이상 보여야 정상 동작
- 호흡 BPM은 최소 10초 데이터 수집 후 안정화됨
- WiFi CSI는 Intel 5300 NIC 또는 Nexmon 펌웨어 필요 (Phase 2)
- 심박 감지는 SNR이 낮아 카메라 방식으로는 정확도 제한 있음

---

## Phase 전환 체크리스트

### Phase 1 → Phase 2 전환 전 완료 조건
- [ ] 카메라 호흡 감지 정확도 ±2 BPM 이내
- [ ] 단위 테스트 커버리지 80% 이상
- [ ] `BaseSensor` 인터페이스 안정화
- [ ] 신호 처리 파이프라인 문서화 완료

---

## 참고 논문 / 프로젝트

- RF-Pose (MIT CSAIL): https://rfpose.csail.mit.edu
- WiVi (MIT): 벽 뒤 사람 감지
- linux-80211n-csitool: Intel 5300 CSI 추출
- Nexmon CSI: https://github.com/seemoo-lab/nexmon_csi
- ESP32-CSI-Tool: https://github.com/StevenMHernandez/ESP32-CSI-Tool
