# 전역 설정 — 이 파일에서만 상수를 수정할 것
FPS: int = 30

# 호흡 주파수 범위
BREATH_HZ_MIN: float = 0.1   # 6 BPM
BREATH_HZ_MAX: float = 0.6   # 36 BPM

# 심박 주파수 범위 (Phase 3)
HEART_HZ_MIN: float = 0.8    # 48 BPM
HEART_HZ_MAX: float = 2.5    # 150 BPM

# 신호 버퍼
WINDOW_SEC: int = 10
BUFFER_SIZE: int = FPS * WINDOW_SEC

# BPM 평활화
SMOOTHER_WINDOW: int = 30

# MediaPipe 설정
MEDIAPIPE_MODEL_COMPLEXITY: int = 1
MEDIAPIPE_MIN_DETECTION_CONFIDENCE: float = 0.6
MEDIAPIPE_MIN_TRACKING_CONFIDENCE: float = 0.6
