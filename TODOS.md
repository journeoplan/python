# TODOS

Phase 1 EXPANSION 리뷰에서 도출된 작업 목록 (2026-03-14).

## 즉시 구현 (이번 작업)

- [x] **Git 초기화** + `.gitignore` 설정
- [x] **Python 3.12 업그레이드** — venv 재생성 (3.9.6 → 3.12.13)
- [x] **SensorOutput 데이터클래스** — `BaseSensor.read() -> SensorOutput` 통일
- [x] **App 오케스트레이터** — `src/app.py` BreathMonitorApp
- [x] **MedianSmoother NaN 필터링** — `update()`에서 isnan 체크
- [x] **FFT magnitude 임계값** — all-zero 신호 → None
- [x] **폰트 경로 크로스 플랫폼** — OS별 폴백 + cv2.putText 폴백
- [x] **FaceMesh 선택적 적용** — 기본 OFF, F키 토글
- [x] **ApneaDetector 구현** — 호흡 패턴 4단계 분류
- [x] **자동 FPS 교정** — sensor.fps로 ButterworthFilter fs 동적 조정
- [x] **FPS 표시 UI** — overlay.py에 draw_fps() 추가
- [x] **OpenCV 윈도우 X 버튼 감지** — cv2.getWindowProperty() 체크
- [x] **테스트 80%+ 커버리지 달성** (로직 코드 97%+)
- [x] **DRY 수정** — make_sine() → conftest.py
- [x] **Git 워크플로우 정책** — 브랜치 전략, PR 규칙, 머지 체크리스트

## 연기 (TODOS)

### P1 — 높은 우선순위
- [ ] **사운드 알림** — 무호흡/이상 호흡 시 경고음 재생 (macOS afplay, Linux simpleaudio). ApneaDetector 완성 후 통합. 노력: S

### P2 — 중간 우선순위
- [ ] **카메라 재연결 로직** — 프레임 손실 시 N초 대기 후 재연결 시도. SafeHome 장시간 모니터링 필수. 노력: M
- [ ] **세션 리포트** — 종료 시 세션 요약 (총 시간, 평균/최대/최소 BPM, 이상 감지 횟수) 출력 + JSON 저장. 노력: S
- [ ] **실시간 대시보드 (dashboard.py)** — matplotlib 기반 별도 윈도우 그래프. Phase 2 SafeHome React 대시보드와 통합 시 구현. 노력: M
- [ ] **녹화 재생 모드 (replay.py)** — CSV 로드 후 오프라인 분석/디버깅. 테스트와 디버깅에 유용. 노력: S
- [ ] **호흡 가이드 라인** — BPM 맞춰 화면에 호흡 가이드 표시. 명상/호흡 훈련 UX. 노력: S
